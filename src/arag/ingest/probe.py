"""Corpus probe: measure the document mess before writing a line of chunking code.

This module does **no** ingest work. It answers one question per document — *what is
actually wrong with this PDF?* — so that every downstream decision (whether to run OCR at
all, whether the clause hierarchy is parseable, how to handle tables) is driven by a
measurement instead of an assumption about what insurance PDFs look like.

The failure class it exists to catch is the one that quietly ruins RAG over real
documents: **a PDF that extracts text successfully but extracts it wrongly.** A scanned
page with no text layer fails loudly and is easy. A page whose fonts emit a single
ligature codepoint, or whose columns interleave, produces fluent text that a naive
pipeline will happily embed, retrieve and cite.

## Heuristics corrected against the real corpus

Three of this probe's first-draft heuristics were **wrong**, each caught by checking its
output against the actual PDFs rather than trusting it. They are recorded here because the
corrections are the reason the current numbers can be trusted:

1. **Space ratio does not detect fused text.** A table extracts one short cell per line,
   so its space ratio is naturally low while its text is perfectly clean. The first
   version flagged ``nivabupa-rise`` p34 — a discount-rate table — as corrupt. Real
   fusion produces abnormally *long* tokens, so token length is what is measured now.

2. **Two-column layout is not a defect.** ``star-comprehensive-2025`` has 32 two-column
   pages and every one extracts in correct reading order, because the content stream emits
   one whole column before the other. Only *interleaving* is damage, so the probe counts
   left/right transitions and reports the subset that actually alternates.

3. **Column-transition counting must exclude tables.** A table's cells are emitted in row
   order, which alternates left/right by construction and looks exactly like splicing. Of
   22 pages initially flagged as spliced, 18 were tables. Excluding table regions leaves
   **4** genuinely spliced prose pages, all in one document.

The uncorrected version of this file over-reported broken pages by roughly 15x.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import pymupdf

from arag.obs import get_logger, span

log = get_logger(__name__)

# --- thresholds, named so the report can be read without reading the code -------------

# Below this many extracted characters a page has no usable text layer. A blank page and
# a scanned page both land here; image coverage separates them.
MIN_CHARS_FOR_TEXT_LAYER = 80

# Fraction of page area covered by images above which a page is image-dominant.
IMAGE_DOMINANT_RATIO = 0.55

# Token fusion. See correction 1 in the module docstring: length, not space ratio.
FUSED_TOKEN_LEN = 30
FUSED_TOKEN_SHARE = 0.03

# Column detection.
COLUMN_BLOCK_SHARE = 0.25
COLUMN_GUTTER_TOL = 4.0
MIN_BLOCKS_FOR_COLUMN_TEST = 6

# Header/footer bands excluded from column analysis: a running header spans the full page
# width and would register as a column-spanning block on every page.
HEADER_BAND = 60.0
FOOTER_BAND = 40.0

# A full-bleed image with almost no text at the front or back of a document is a cover,
# not a scanned content page. Verified against star-comprehensive-2025 p1, which is one
# 595x842 image plus the words "Policy Wordings".
COVER_MAX_CHARS = 30

CID_PATTERN = re.compile(r"\(cid:\d+\)")
REPLACEMENT_CHAR = "�"

# Typographic ligatures, U+FB00..U+FB06 (ff fi fl ffi ffl ft st). The highest-impact defect
# found in this corpus. "benefit" typeset with an fi-ligature extracts as a string
# containing U+FB01 — a single codepoint that is NOT the letters f and i — so a BM25 query
# for "benefit" cannot match it and the embedder sees an out-of-vocabulary token.
LIGATURES = re.compile("[\ufb00-\ufb06]")
# Written as escapes, not literals: these characters are invisible or confusable in a
# source file, and a reviewer cannot tell a NO-BREAK SPACE from a SPACE by looking.
SOFT_HYPHEN = "\u00ad"
NBSP = "\u00a0"

HEADING_PATTERNS: dict[str, re.Pattern[str]] = {
    # 3.1, 4.2.b, 1.2.14 — the numbering a policy wording hangs its structure on.
    "numbered_clause": re.compile(r"^\s{0,6}(\d+(?:\.\d+){0,3})[\.\)]?\s+\S"),
    "section_roman": re.compile(r"^\s{0,6}SECTION\s+[IVXLCDM]+\b", re.IGNORECASE),
    "section_numeric": re.compile(r"^\s{0,6}SECTION\s+\d+\b", re.IGNORECASE),
    "all_caps": re.compile(r"^\s{0,6}[A-Z][A-Z0-9 \-&/(),\.]{6,80}$"),
    "annexure": re.compile(r"^\s{0,6}(ANNEXURE|SCHEDULE|APPENDIX)\b", re.IGNORECASE),
    "alpha_subclause": re.compile(r"^\s{0,6}\(([a-z]{1,3}|[ivxl]{1,5})\)\s+\S"),
}

DEVANAGARI = re.compile("[ऀ-ॿ]")


@dataclass
class PageProbe:
    number: int
    chars: int = 0
    words: int = 0
    has_text_layer: bool = False

    image_count: int = 0
    image_area_ratio: float = 0.0
    image_dominant: bool = False
    likely_cover: bool = False

    blocks: int = 0
    two_column: bool = False
    # Transitions counted with table regions EXCLUDED. See correction 3.
    column_switches: int = 0
    prose_blocks_outside_tables: int = 0
    reading_order_broken: bool = False
    table_driven_switching: bool = False

    table_candidates: int = 0
    table_error: str | None = None

    cid_artefacts: int = 0
    replacement_chars: int = 0
    ligature_chars: int = 0
    soft_hyphens: int = 0
    nbsp_chars: int = 0
    space_ratio: float = 0.0
    long_token_share: float = 0.0
    fused_text: bool = False

    devanagari_chars: int = 0
    rotation: int = 0
    heading_hits: dict[str, int] = field(default_factory=dict)
    max_font_size: float = 0.0
    body_font_size: float = 0.0

    @property
    def needs_ocr(self) -> bool:
        """No usable text, mostly picture, and not a cover: OCR is the only way in.

        The cover exclusion matters. Without it this fires on every glossy title page and
        the OCR budget gets spent rendering company logos.
        """
        return not self.has_text_layer and self.image_dominant and not self.likely_cover

    @property
    def ligature_contaminated(self) -> bool:
        return self.ligature_chars > 0

    @property
    def suspect(self) -> bool:
        """Text was extracted, but it is probably wrong.

        The dangerous category: worse than a page that fails loudly, because a naive
        pipeline embeds and cites it without complaint.
        """
        return self.has_text_layer and (
            self.cid_artefacts > 0
            or self.replacement_chars > 3
            or self.fused_text
            or self.reading_order_broken
            or self.ligature_contaminated
        )


@dataclass
class DocumentProbe:
    source_id: str
    path: Path
    ok: bool = True
    error: str | None = None

    pages: int = 0
    encrypted: bool = False
    needs_password: bool = False
    has_outline: bool = False
    outline_entries: int = 0
    producer: str | None = None
    page_probes: list[PageProbe] = field(default_factory=list)

    # --- aggregates the report table is built from ---

    @property
    def total_chars(self) -> int:
        return sum(p.chars for p in self.page_probes)

    @property
    def cover_pages(self) -> list[int]:
        return [p.number for p in self.page_probes if p.likely_cover]

    @property
    def pages_without_text(self) -> list[int]:
        return [p.number for p in self.page_probes if not p.has_text_layer]

    @property
    def pages_needing_ocr(self) -> list[int]:
        return [p.number for p in self.page_probes if p.needs_ocr]

    @property
    def blank_pages(self) -> list[int]:
        """Empty pages that are neither OCR candidates nor covers.

        Covers are subtracted explicitly: reporting one page as both "cover excluded from
        OCR" and "empty page" is two lines of noise describing one benign fact.
        """
        return sorted(
            set(self.pages_without_text) - set(self.pages_needing_ocr) - set(self.cover_pages)
        )

    @property
    def suspect_pages(self) -> list[int]:
        return [p.number for p in self.page_probes if p.suspect]

    @property
    def two_column_pages(self) -> list[int]:
        """Pages laid out in two columns. Informational — most extract correctly."""
        return [p.number for p in self.page_probes if p.two_column]

    @property
    def reading_order_broken_pages(self) -> list[int]:
        """Two-column prose pages whose extraction order genuinely interleaves."""
        return [p.number for p in self.page_probes if p.reading_order_broken]

    @property
    def table_driven_switch_pages(self) -> list[int]:
        return [p.number for p in self.page_probes if p.table_driven_switching]

    @property
    def fused_text_pages(self) -> list[int]:
        return [p.number for p in self.page_probes if p.fused_text]

    @property
    def cid_pages(self) -> list[int]:
        return [p.number for p in self.page_probes if p.cid_artefacts > 0]

    @property
    def ligature_pages(self) -> list[int]:
        return [p.number for p in self.page_probes if p.ligature_contaminated]

    @property
    def ligature_chars(self) -> int:
        return sum(p.ligature_chars for p in self.page_probes)

    @property
    def soft_hyphens(self) -> int:
        return sum(p.soft_hyphens for p in self.page_probes)

    @property
    def nbsp_chars(self) -> int:
        return sum(p.nbsp_chars for p in self.page_probes)

    @property
    def table_candidates(self) -> int:
        return sum(p.table_candidates for p in self.page_probes)

    @property
    def pages_with_tables(self) -> list[int]:
        return [p.number for p in self.page_probes if p.table_candidates > 0]

    @property
    def devanagari_chars(self) -> int:
        return sum(p.devanagari_chars for p in self.page_probes)

    @property
    def median_chars_per_page(self) -> float:
        values = [p.chars for p in self.page_probes]
        return statistics.median(values) if values else 0.0

    @property
    def heading_totals(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for page in self.page_probes:
            for name, count in page.heading_hits.items():
                totals[name] = totals.get(name, 0) + count
        return dict(sorted(totals.items(), key=lambda kv: -kv[1]))

    @property
    def dominant_heading_pattern(self) -> str | None:
        totals = self.heading_totals
        return next(iter(totals), None) if totals else None

    @property
    def scanned_fraction(self) -> float:
        return len(self.pages_needing_ocr) / self.pages if self.pages else 0.0

    def breakages(self) -> list[str]:
        """What is actually wrong, worst first.

        Ordered so silent corruption precedes loud failure, because the silent kind is
        what reaches production and gets cited in an answer.
        """
        out: list[str] = []
        if not self.ok:
            return [f"UNREADABLE: {self.error}"]
        if self.needs_password:
            out.append("password-protected: no content reachable")
        if self.ligature_pages:
            out.append(
                f"LIGATURE CONTAMINATION: {self.ligature_chars} ligature codepoint(s) on "
                f"{len(self.ligature_pages)} page(s). 'benefit' extracts with U+FB01, a "
                f"single char that is not f+i, so BM25 cannot match it and the embedder "
                f"sees an unknown token. Needs Unicode NFKC normalisation at ingest: "
                f"{_compact(self.ligature_pages)}"
            )
        if self.cid_pages:
            out.append(
                f"broken font encoding on {len(self.cid_pages)} page(s) (CID artefacts) — "
                f"extraction yields confident garbage: {_compact(self.cid_pages)}"
            )
        if self.fused_text_pages:
            out.append(
                f"fused tokens on {len(self.fused_text_pages)} page(s) "
                f"(>{FUSED_TOKEN_SHARE:.0%} of tokens over {FUSED_TOKEN_LEN} chars): "
                f"{_compact(self.fused_text_pages)}"
            )
        if self.reading_order_broken_pages:
            out.append(
                f"INTERLEAVED prose columns on {len(self.reading_order_broken_pages)} "
                f"page(s), table regions excluded — clauses from different columns are "
                f"spliced and read fluently while being wrong: "
                f"{_compact(self.reading_order_broken_pages)}"
            )
        if self.soft_hyphens:
            out.append(
                f"{self.soft_hyphens} soft hyphen(s) — split words survive extraction and "
                "break exact matching"
            )
        if self.pages_needing_ocr:
            out.append(
                f"{len(self.pages_needing_ocr)} page(s) have no text layer and are "
                f"image-dominant — OCR required: {_compact(self.pages_needing_ocr)}"
            )
        if self.blank_pages:
            out.append(f"{len(self.blank_pages)} page(s) empty: {_compact(self.blank_pages)}")
        if self.devanagari_chars:
            out.append(
                f"{self.devanagari_chars} Devanagari character(s) — DESIGN section 13 "
                "assumption 2 (English-only corpus) does not hold here"
            )
        if not self.dominant_heading_pattern:
            out.append("no heading pattern detected — clause hierarchy is not parseable")
        if not out:
            out.append("no blocking defects found")
        return out

    def notes(self) -> list[str]:
        """Observations that are not defects but change how ingest must be written."""
        out: list[str] = []
        benign = sorted(set(self.two_column_pages) - set(self.reading_order_broken_pages))
        if benign:
            out.append(
                f"{len(benign)} two-column page(s) extract in CORRECT order — layout alone "
                f"is not damage: {_compact(benign)}"
            )
        if self.table_driven_switch_pages:
            out.append(
                f"{len(self.table_driven_switch_pages)} page(s) alternate columns only "
                f"inside table regions — row-major cell order, not splicing: "
                f"{_compact(self.table_driven_switch_pages)}"
            )
        if self.pages_with_tables:
            out.append(
                f"{self.table_candidates} table candidate(s) on "
                f"{len(self.pages_with_tables)}/{self.pages} page(s) — the highest-value "
                f"and most fragile content in this domain"
            )
        if self.cover_pages:
            out.append(f"cover page(s) excluded from OCR: {_compact(self.cover_pages)}")
        if self.nbsp_chars:
            out.append(f"{self.nbsp_chars} non-breaking space(s) — normalise at ingest")
        if self.has_outline:
            out.append(f"PDF outline present with {self.outline_entries} entries")
        return out


def _compact(numbers: list[int], limit: int = 12) -> str:
    """Render a page list as ranges: [1,2,3,7,9,10] -> '1-3, 7, 9-10'."""
    if not numbers:
        return "-"
    ranges: list[str] = []
    start = prev = numbers[0]
    for n in numbers[1:]:
        if n == prev + 1:
            prev = n
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = n
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    if len(ranges) > limit:
        return ", ".join(ranges[:limit]) + f", +{len(ranges) - limit} more"
    return ", ".join(ranges)


def _body_blocks(page: pymupdf.Page, info: dict[str, Any]) -> list[dict[str, Any]]:
    """Text blocks excluding the running header and footer bands."""
    height = page.rect.height
    return [
        b
        for b in info.get("blocks", [])
        if b.get("type") == 0
        and b.get("lines")
        and b.get("bbox")
        and b["bbox"][1] > HEADER_BAND
        and b["bbox"][3] < height - FOOTER_BAND
    ]


def _column_profile(
    blocks: list[dict[str, Any]],
    page_width: float,
    table_rects: list[pymupdf.Rect],
) -> tuple[bool, int, int]:
    """Return (is_two_column, switches_outside_tables, prose_block_count).

    ``switches`` counts left<->right transitions in the order the PDF emits blocks, with
    every block overlapping a detected table removed first. A correctly-ordered two-column
    page emits one whole column then the other, giving exactly **one** switch. Interleaved
    prose alternates and gives many.

    Excluding tables is essential and was learned the hard way: table cells are emitted
    row-major, which alternates left/right by construction. Without the exclusion this
    function reports 22 spliced pages on the real corpus; with it, 4.
    """
    if len(blocks) < MIN_BLOCKS_FOR_COLUMN_TEST:
        return False, 0, 0

    mid = page_width / 2
    left = right = spanning = 0
    sequence: list[str] = []

    for block in blocks:
        x0, _, x1, _ = block["bbox"]
        if x1 <= mid + COLUMN_GUTTER_TOL:
            left += 1
            side = "L"
        elif x0 >= mid - COLUMN_GUTTER_TOL:
            right += 1
            side = "R"
        else:
            spanning += 1
            continue
        if not any(pymupdf.Rect(block["bbox"]).intersects(r) for r in table_rects):
            sequence.append(side)

    total = len(blocks)
    is_two_column = (
        left / total >= COLUMN_BLOCK_SHARE
        and right / total >= COLUMN_BLOCK_SHARE
        and spanning / total < 0.35
    )
    switches = sum(1 for a, b in pairwise(sequence) if a != b)
    return is_two_column, switches, len(sequence)


def probe_page(page: pymupdf.Page, number: int, total_pages: int = 0) -> PageProbe:
    out = PageProbe(number=number, rotation=page.rotation)
    text = page.get_text("text") or ""
    out.chars = len(text.strip())
    out.words = len(text.split())
    out.has_text_layer = out.chars >= MIN_CHARS_FOR_TEXT_LAYER

    if text:
        out.space_ratio = text.count(" ") / len(text)
        tokens = text.split()
        if tokens:
            long_tokens = sum(1 for tok in tokens if len(tok) > FUSED_TOKEN_LEN)
            out.long_token_share = long_tokens / len(tokens)
            out.fused_text = out.has_text_layer and out.long_token_share >= FUSED_TOKEN_SHARE
    out.cid_artefacts = len(CID_PATTERN.findall(text))
    out.replacement_chars = text.count(REPLACEMENT_CHAR)
    out.ligature_chars = len(LIGATURES.findall(text))
    out.soft_hyphens = text.count(SOFT_HYPHEN)
    out.nbsp_chars = text.count(NBSP)
    out.devanagari_chars = len(DEVANAGARI.findall(text))

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or len(stripped) > 120:
            continue
        for name, pattern in HEADING_PATTERNS.items():
            if pattern.match(line):
                out.heading_hits[name] = out.heading_hits.get(name, 0) + 1

    # Tables are found FIRST: their bounding boxes are an input to column analysis.
    table_rects: list[pymupdf.Rect] = []
    try:
        found = page.find_tables()
        out.table_candidates = len(found.tables)
        table_rects = [pymupdf.Rect(t.bbox) for t in found.tables]
    except Exception as exc:
        # The most failure-prone call in the probe. Recorded, never raised, so one bad
        # page cannot end the corpus scan.
        out.table_error = f"{type(exc).__name__}: {exc}"

    page_area = abs(page.rect.get_area()) or 1.0
    info = page.get_text("dict")
    blocks = info.get("blocks", [])
    out.blocks = len(blocks)

    image_area = 0.0
    sizes: list[float] = []
    for block in blocks:
        if block.get("type") == 1:
            out.image_count += 1
            x0, y0, x1, y1 = block.get("bbox", (0, 0, 0, 0))
            image_area += max(0.0, x1 - x0) * max(0.0, y1 - y0)
        else:
            for line in block.get("lines", []):
                for spn in line.get("spans", []):
                    size = float(spn.get("size", 0.0))
                    if size > 0:
                        sizes.append(size)

    out.image_area_ratio = min(1.0, image_area / page_area)
    out.image_dominant = out.image_area_ratio >= IMAGE_DOMINANT_RATIO
    near_edges = number <= 2 or bool(total_pages and number >= total_pages - 1)
    out.likely_cover = bool(out.image_dominant and out.chars < COVER_MAX_CHARS and near_edges)

    if sizes:
        out.max_font_size = max(sizes)
        out.body_font_size = statistics.median(sizes)

    body = _body_blocks(page, info)
    out.two_column, out.column_switches, out.prose_blocks_outside_tables = _column_profile(
        body, page.rect.width, table_rects
    )
    out.reading_order_broken = out.two_column and out.column_switches > 1

    # A page that alternates columns only inside tables. Reported as a note so nobody has
    # to re-derive the table-cell-order discovery from scratch.
    if table_rects:
        _, raw_switches, _ = _column_profile(body, page.rect.width, [])
        out.table_driven_switching = bool(
            out.two_column and raw_switches > 1 and out.column_switches <= 1
        )
    return out


def probe_document(source_id: str, path: Path, *, max_pages: int | None = None) -> DocumentProbe:
    result = DocumentProbe(source_id=source_id, path=path)
    with span("ingest.probe"):
        try:
            doc = pymupdf.open(path)
        except Exception as exc:
            result.ok = False
            result.error = f"{type(exc).__name__}: {exc}"
            log.error("probe_open_failed", source_id=source_id, error=result.error)
            return result

        with doc:
            result.encrypted = bool(doc.is_encrypted)
            result.needs_password = bool(doc.needs_pass)
            result.producer = (doc.metadata or {}).get("producer")
            result.pages = doc.page_count
            try:
                toc = doc.get_toc()
                result.outline_entries = len(toc)
                result.has_outline = bool(toc)
            except Exception:
                result.has_outline = False

            if result.needs_password:
                result.ok = False
                result.error = "password required"
                return result

            limit = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
            for index in range(limit):
                try:
                    result.page_probes.append(
                        probe_page(doc[index], index + 1, total_pages=doc.page_count)
                    )
                except Exception as exc:
                    log.error(
                        "probe_page_failed", source_id=source_id, page=index + 1, error=str(exc)
                    )
                    result.page_probes.append(
                        PageProbe(number=index + 1, table_error=f"page failed: {exc}")
                    )

    log.info(
        "probe_done",
        source_id=source_id,
        pages=result.pages,
        chars=result.total_chars,
        ocr_pages=len(result.pages_needing_ocr),
        spliced_pages=len(result.reading_order_broken_pages),
        ligature_chars=result.ligature_chars,
        tables=result.table_candidates,
    )
    return result
