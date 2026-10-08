"""Table extraction, stitching, and serialisation.

Tables are the load-bearing ingest problem in this corpus, not an edge case: spike S5
measured **165 table candidates on 131 of 197 pages**, including 41 of 46 annexure pages.
In Indian health insurance the tables *are* the answers — room-rent tiers, sub-limits,
waiting-period grids, discount schedules. Everything else is prose about them.

Three rules, implemented separately because they fail separately.

## T1 — stitch tables across page boundaries

Tables on 41 of 46 consecutive pages means tables necessarily span page breaks. An
unstitched page-spanning table becomes two half-tables, each individually plausible and
neither complete: a sum-insured band that continues overleaf silently loses its upper
rows, and a query for the highest band retrieves a table that looks authoritative and
does not contain the answer.

## T2 — the header row reaches every derived chunk, or ingest fails

A header-less table chunk is not degraded retrieval. It is a number with no meaning and
no detection path downstream. Given ``| Cataract | 25,000 |`` with no header, the
generator cannot know whether 25,000 is a per-eye limit, a per-policy-year limit, or a
deductible — and it will pick one and sound confident. So ``HeaderlessTableError`` is
raised at ingest rather than logged: minutes of ingest time against a wrong coverage
figure nobody can detect.

## T3 — two serialisations, and the eval picks

Both are built because the choice is an empirical question, not a preference:

* **Markdown** keeps the whole table in one chunk. Structure is preserved; the text is
  mostly pipes and digits, which gives a dense embedder little semantic signal and BM25
  almost nothing to match a natural-language query against.
* **Row-level natural language** turns each row into a sentence carrying its header names
  inline: *"For a sum insured of Rs 5,00,000, the room rent limit per day is 1% of sum
  insured."* Should retrieve far better; loses the surrounding table's structure.

Expectation going in is that markdown retrieves badly. That expectation is recorded here
so the measurement can contradict it. Whichever wins is indexed; the original table is
kept as answer context either way.

Note that T2 is satisfied *structurally* by both: markdown chunks include the header row,
and row sentences name the header inline. A chunk cannot exist without its header, which
is a stronger guarantee than remembering to attach one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pymupdf

from arag.ingest.normalise import normalise_text
from arag.obs import get_logger, span

log = get_logger(__name__)

# --- stitching thresholds -------------------------------------------------------------

# A table whose bbox reaches within this many points of the bottom text margin is a
# candidate for continuing overleaf.
BOTTOM_MARGIN_TOL = 90.0
# ...and one starting within this distance of the top text margin may be its continuation.
TOP_MARGIN_TOL = 140.0
# Column geometry must line up this closely for two fragments to be the same table.
COLUMN_X_TOL = 24.0

# Chunk sizing. Characters, not tokens: the embedder is not chosen per-serialisation and a
# character budget is deterministic and dependency-free.
MAX_TABLE_CHUNK_CHARS = 1400
MIN_ROWS_PER_CHUNK = 1

_WS = re.compile(r"\s+")


class HeaderlessTableError(RuntimeError):
    """A table produced a chunk with no header. Ingest stops (T2).

    Deliberately an exception. A header-less table chunk yields a confidently wrong
    numeric answer with no way to detect it downstream, so this must be impossible to
    ignore rather than a warning in a long ingest log.
    """

    def __init__(self, table_id: str, detail: str) -> None:
        super().__init__(
            f"table {table_id} would produce a chunk with no header: {detail}. "
            "A table row without its header is a number with no meaning - the generator "
            "cannot tell a per-eye limit from a per-year limit and will pick one. "
            "Fix the extraction or exclude the table explicitly; do not index it."
        )
        self.table_id = table_id


def _clean(value: Any) -> str:
    """Normalise one cell. Cells are frequently None or ragged whitespace."""
    if value is None:
        return ""
    text, _ = normalise_text(str(value), context="<cell>", strict_numerics=False)
    return _WS.sub(" ", text).strip()


@dataclass
class TableFragment:
    """One table as found on a single page, before stitching."""

    source_id: str
    page: int
    index_on_page: int
    bbox: tuple[float, float, float, float]
    col_xs: tuple[float, ...]
    header: tuple[str, ...]
    header_external: bool
    rows: tuple[tuple[str, ...], ...]
    page_height: float

    @property
    def fragment_id(self) -> str:
        return f"{self.source_id}:p{self.page}:t{self.index_on_page}"

    @property
    def col_count(self) -> int:
        return max(len(self.header), max((len(r) for r in self.rows), default=0))

    @property
    def reaches_bottom(self) -> bool:
        return self.bbox[3] >= self.page_height - BOTTOM_MARGIN_TOL

    @property
    def starts_at_top(self) -> bool:
        return self.bbox[1] <= TOP_MARGIN_TOL


@dataclass
class ExtractedTable:
    """A logical table, possibly assembled from fragments on consecutive pages."""

    table_id: str
    source_id: str
    pages: tuple[int, ...]
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    stitched_from: int = 1
    header_repeats_dropped: int = 0
    caption: str | None = None
    clause_id: str | None = None

    @property
    def col_count(self) -> int:
        return len(self.header)

    @property
    def spans_pages(self) -> bool:
        return len(self.pages) > 1

    @property
    def has_header(self) -> bool:
        return any(cell.strip() for cell in self.header)

    def context_label(self) -> str:
        """Human-readable provenance, prepended to every chunk so a retrieved row can be
        traced without consulting metadata.
        """
        pages = (
            f"pages {self.pages[0]}-{self.pages[-1]}"
            if self.spans_pages
            else f"page {self.pages[0]}"
        )
        bits = [self.source_id, pages]
        if self.clause_id:
            bits.append(f"clause {self.clause_id}")
        if self.caption:
            bits.append(self.caption)
        return ", ".join(bits)


# --------------------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------------------


def extract_fragments(source_id: str, page: pymupdf.Page, page_number: int) -> list[TableFragment]:
    """Find every table on one page. Never raises: a page whose tables cannot be parsed
    is recorded as having none, because one bad page must not end a corpus ingest.
    """
    try:
        found = page.find_tables()
    except Exception as exc:
        log.warning("table_find_failed", source_id=source_id, page=page_number, error=str(exc))
        return []

    out: list[TableFragment] = []
    for idx, table in enumerate(found.tables):
        try:
            raw_rows = table.extract()
        except Exception as exc:
            log.warning(
                "table_extract_failed",
                source_id=source_id,
                page=page_number,
                table=idx,
                error=str(exc),
            )
            continue

        rows = [tuple(_clean(c) for c in row) for row in raw_rows]

        header: tuple[str, ...] = ()
        external = False
        hdr = getattr(table, "header", None)
        if hdr is not None and getattr(hdr, "names", None):
            header = tuple(_clean(n) for n in hdr.names)
            external = bool(getattr(hdr, "external", False))

        # When the header is not external it IS the first extracted row, so drop the
        # duplicate. Keeping it would make the header appear as a data row in every
        # row-level serialisation.
        if header and not external and rows and _rows_match(rows[0], header):
            rows = rows[1:]
        if not header and rows:
            header = rows[0]
            rows = rows[1:]

        col_xs = tuple(round(float(x), 1) for x in sorted({c[0] for c in (table.cells or []) if c}))

        out.append(
            TableFragment(
                source_id=source_id,
                page=page_number,
                index_on_page=idx,
                bbox=tuple(float(v) for v in table.bbox),  # type: ignore[arg-type]
                col_xs=col_xs,
                header=header,
                header_external=external,
                rows=tuple(r for r in rows if any(cell for cell in r)),
                page_height=float(page.rect.height),
            )
        )
    return out


def _rows_match(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    norm = lambda row: tuple(c.lower() for c in row if c)  # noqa: E731
    return bool(norm(a)) and norm(a) == norm(b)


# --------------------------------------------------------------------------------------
# T1: stitching
# --------------------------------------------------------------------------------------


def _is_continuation(prev: TableFragment, nxt: TableFragment) -> bool:
    """Is ``nxt`` the continuation of ``prev`` from the previous page?

    Four conditions, all necessary. Column geometry is the discriminating one: two
    unrelated tables on consecutive pages routinely satisfy the page-adjacency and
    column-count tests, and only their x-positions distinguish them.
    """
    if nxt.page != prev.page + 1:
        return False
    if not prev.reaches_bottom or not nxt.starts_at_top:
        return False
    if prev.col_count != nxt.col_count or prev.col_count == 0:
        return False

    if prev.col_xs and nxt.col_xs and len(prev.col_xs) == len(nxt.col_xs):
        if any(abs(a - b) > COLUMN_X_TOL for a, b in zip(prev.col_xs, nxt.col_xs, strict=True)):
            return False
    else:
        # Fall back to the table's own left/right edges when per-column geometry is
        # unavailable. Weaker, so it is used only as a fallback.
        if abs(prev.bbox[0] - nxt.bbox[0]) > COLUMN_X_TOL:
            return False
        if abs(prev.bbox[2] - nxt.bbox[2]) > COLUMN_X_TOL:
            return False
    return True


def stitch(fragments: list[TableFragment]) -> list[ExtractedTable]:
    """Merge page-spanning fragments into logical tables (T1).

    A continuation fragment's own header is dropped when it repeats the parent's — policy
    wordings usually repeat the header on each page, and keeping it would inject a header
    row into the middle of the data.
    """
    ordered = sorted(fragments, key=lambda f: (f.page, f.bbox[1]))
    tables: list[ExtractedTable] = []
    current: ExtractedTable | None = None
    previous: TableFragment | None = None

    for frag in ordered:
        if current is not None and previous is not None and _is_continuation(previous, frag):
            rows = list(frag.rows)
            dropped = 0
            if rows and _rows_match(rows[0], current.header):
                rows = rows[1:]
                dropped = 1
            elif frag.header and _rows_match(frag.header, current.header):
                dropped = 1
            elif frag.header and not frag.header_external:
                # A continuation whose first row is data, not a repeated header: the
                # extractor mistook it for a header, so put it back.
                rows = [frag.header, *rows]

            current = ExtractedTable(
                table_id=current.table_id,
                source_id=current.source_id,
                pages=(*current.pages, frag.page),
                header=current.header,
                rows=(*current.rows, *(tuple(r) for r in rows)),
                stitched_from=current.stitched_from + 1,
                header_repeats_dropped=current.header_repeats_dropped + dropped,
                caption=current.caption,
                clause_id=current.clause_id,
            )
            tables[-1] = current
            previous = frag
            continue

        current = ExtractedTable(
            table_id=frag.fragment_id,
            source_id=frag.source_id,
            pages=(frag.page,),
            header=frag.header,
            rows=frag.rows,
        )
        tables.append(current)
        previous = frag

    return tables


# --------------------------------------------------------------------------------------
# stitch classification
# --------------------------------------------------------------------------------------

# A stitched region's "header" separates cleanly by length on this corpus: real column
# headers measure at most 57 characters, mis-detected prose paragraphs measure 313 to
# 2286. The threshold sits in the middle of a 250-character gap, which is why it can be
# a constant rather than a tuned parameter.
#
# The first version of this classifier used 45 characters and was WRONG: it flagged
# legitimately verbose column names ("ReAssure+ is triggered...", 50 chars) as prose,
# reporting 4 genuine grids where there are 7. The wide gap is what makes the corrected
# threshold safe, and tests/test_ingest_tables.py pins both the threshold and the counts
# so the classification cannot drift back.
PROSE_HEADER_CHARS = 250


class StitchClass(StrEnum):
    """What a page-spanning "table" actually is.

    Reported rather than acted on. A spurious stitch is not deleted, because the prose it
    contains is still real content that the prose chunker indexes; the classification
    exists so a stitch count is not quietly overstated, and so a labeller can tell which
    stitched tables are worth targeting.
    """

    DATA_GRID = "data_grid"
    CAPTION_HEADER = "caption_header"
    SPURIOUS = "spurious"

    @property
    def is_reliable(self) -> bool:
        return self is StitchClass.DATA_GRID


def classify_stitch(table: ExtractedTable) -> tuple[StitchClass, str]:
    """Classify a stitched table, with the reason.

    * ``SPURIOUS`` - a header cell of 250+ characters is a paragraph, so ``find_tables``
      detected a prose region and the merge is meaningless.
    * ``CAPTION_HEADER`` - a single header cell is the table's caption, not its column
      names. T2 passes (a header exists) but the header is uninformative, so a row
      sentence reads "For Table - B2 27%: ..." and is unretrievable. This is a real
      weakness in the T2 guarantee, not a rounding error.
    * ``DATA_GRID`` - two or more short column names. Trustworthy.
    """
    cells = [c for c in table.header if c.strip()]
    if not cells:
        return StitchClass.SPURIOUS, "no header cells at all"

    longest = max(len(c) for c in cells)
    if longest >= PROSE_HEADER_CHARS:
        return (
            StitchClass.SPURIOUS,
            f"longest header cell is {longest} chars - that is a paragraph, not a column name",
        )
    if len(cells) < 2:
        return (
            StitchClass.CAPTION_HEADER,
            f"single header cell {cells[0][:40]!r} is a caption, not column names - T2 "
            "passes but row sentences built from it are unretrievable",
        )
    return StitchClass.DATA_GRID, f"{len(cells)} column names, longest {longest} chars"


def classify_stitches(
    tables: list[ExtractedTable],
) -> dict[StitchClass, list[tuple[ExtractedTable, str]]]:
    """Group every stitched table by class. Single-page tables are not classified."""
    out: dict[StitchClass, list[tuple[ExtractedTable, str]]] = {c: [] for c in StitchClass}
    for table in tables:
        if table.stitched_from <= 1:
            continue
        cls, why = classify_stitch(table)
        out[cls].append((table, why))
    return out


# --------------------------------------------------------------------------------------
# T3: serialisation
# --------------------------------------------------------------------------------------


def to_markdown(table: ExtractedTable, rows: tuple[tuple[str, ...], ...] | None = None) -> str:
    """Whole-table markdown. The header is always emitted, satisfying T2 structurally."""
    body = table.rows if rows is None else rows
    width = table.col_count or max((len(r) for r in body), default=0)
    if width == 0:
        return ""

    def line(cells: tuple[str, ...]) -> str:
        padded = list(cells) + [""] * (width - len(cells))
        return "| " + " | ".join(c.replace("|", "/") for c in padded[:width]) + " |"

    out = [f"Table from {table.context_label()}", "", line(table.header)]
    out.append("|" + "---|" * width)
    out.extend(line(row) for row in body)
    return "\n".join(out)


def _is_data_row(row: tuple[str, ...]) -> bool:
    """A data row's first non-empty cell carries a value (a digit); a header row's does not.

    Used to recover a caption-as-header: find_tables sometimes takes a single-cell TITLE row
    as the header (e.g. "Delivery and New Born"), pushing the real column names - "Sum
    Insured", "Normal Delivery", "Delivery by Ceasarean Section" - down into data rows. The
    row serialiser then labels every column "value" and the axes (sum-insured band, delivery
    type) vanish from the sentence, so a query for "caesarean delivery" matches nothing.
    Measured on star-comprehensive-2025 p14 (h-10) and p14 vaccination (h-09).
    """
    first = next((c.strip() for c in row if c.strip()), "")
    return any(ch.isdigit() for ch in first)


def _recover_caption_header(
    header: list[str], rows: list[tuple[str, ...]]
) -> tuple[list[str], list[tuple[str, ...]], str | None]:
    """If the header is a single-cell caption, promote and merge the real column-name rows.

    Returns (header, remaining_rows, recovered_caption). Only fires when the header has at
    most one non-empty cell AND the leading rows are header-like (no value in the first
    cell), so an ordinary table is untouched. Multi-row headers (a column group over a
    sub-header, as in Normal/Caesarean under "Limit for Delivery") are merged column-wise so
    both levels survive into the sentence.
    """
    if sum(1 for c in header if c.strip()) > 1:
        return header, rows, None
    caption = next((c.strip() for c in header if c.strip()), None)
    hdr_rows: list[tuple[str, ...]] = []
    rest = list(rows)
    while rest and not _is_data_row(rest[0]):
        hdr_rows.append(rest.pop(0))
    if not hdr_rows:
        return header, rows, None
    ncols = max(len(header), *(len(r) for r in hdr_rows))
    merged = [
        " ".join(r[i].strip() for r in hdr_rows if i < len(r) and r[i].strip())
        for i in range(ncols)
    ]
    return merged, rest, caption


def to_row_sentences(table: ExtractedTable) -> list[str]:
    """One natural-language sentence per row, header names inline.

    The header is woven into the sentence rather than prefixed, which is what makes this
    serialisation retrievable: a query for "room rent limit for 5 lakh sum insured"
    shares real tokens with "For a sum insured of Rs 5,00,000, the room rent limit per day
    is ...", and shares almost nothing with "| 5,00,000 | 1% |".
    """
    if not table.has_header:
        raise HeaderlessTableError(table.table_id, "no header row available")

    header, rows_list, recovered = _recover_caption_header(list(table.header), list(table.rows))
    label = table.context_label()
    if recovered and recovered.lower() not in label.lower():
        label = f"{label}, {recovered}"
    sentences: list[str] = []
    for row in rows_list:
        cells = list(row) + [""] * (len(header) - len(row))
        pairs = [
            (header[i].strip(), cells[i].strip())
            for i in range(min(len(header), len(cells)))
            if cells[i].strip()
        ]
        if not pairs:
            continue

        subject_name, subject_value = pairs[0]
        rest = pairs[1:]
        lead = f"For {subject_name} {subject_value}" if subject_name else f"For {subject_value}"

        if rest:
            clauses = "; ".join(f"{name or 'value'} is {value}" for name, value in rest)
            sentence = f"{lead}: {clauses}."
        else:
            sentence = f"{lead}."
        sentences.append(f"[{label}] {sentence}")
    return sentences


# --------------------------------------------------------------------------------------
# T2: chunk derivation
# --------------------------------------------------------------------------------------


@dataclass
class TableChunk:
    """A chunk derived from a table. Cannot exist without a header (T2)."""

    chunk_id: str
    table_id: str
    source_id: str
    pages: tuple[int, ...]
    text: str
    serialisation: str
    header: tuple[str, ...]
    row_span: tuple[int, int]
    clause_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not any(cell.strip() for cell in self.header):
            raise HeaderlessTableError(self.table_id, f"chunk {self.chunk_id} has no header")


def table_chunks(
    table: ExtractedTable,
    *,
    serialisation: str = "markdown",
    max_chars: int = MAX_TABLE_CHUNK_CHARS,
) -> list[TableChunk]:
    """Derive chunks from a table. Raises ``HeaderlessTableError`` rather than emitting a
    chunk whose rows have lost their header.

    Markdown mode keeps the table whole when it fits and otherwise splits into row groups
    **repeating the header in each group** — which is the only correct way to split a
    table, and the reason "never split a table" was too coarse a rule to implement.
    """
    if not table.has_header:
        raise HeaderlessTableError(table.table_id, "table has no header row")
    if not table.rows:
        return []

    if serialisation == "row_nl":
        row_chunks: list[TableChunk] = []
        for i, sentence in enumerate(to_row_sentences(table)):
            row_chunks.append(
                TableChunk(
                    chunk_id=f"{table.table_id}#r{i}",
                    table_id=table.table_id,
                    source_id=table.source_id,
                    pages=table.pages,
                    text=sentence,
                    serialisation="row_nl",
                    header=table.header,
                    row_span=(i, i),
                    clause_id=table.clause_id,
                    meta={"spans_pages": table.spans_pages},
                )
            )
        return row_chunks

    if serialisation != "markdown":
        raise ValueError(f"unknown serialisation {serialisation!r}")

    whole = to_markdown(table)
    if len(whole) <= max_chars:
        return [
            TableChunk(
                chunk_id=f"{table.table_id}#whole",
                table_id=table.table_id,
                source_id=table.source_id,
                pages=table.pages,
                text=whole,
                serialisation="markdown",
                header=table.header,
                row_span=(0, len(table.rows) - 1),
                clause_id=table.clause_id,
                meta={"spans_pages": table.spans_pages, "split": False},
            )
        ]

    # Too large: split into row groups, header repeated in every group.
    chunks: list[TableChunk] = []
    group: list[tuple[str, ...]] = []
    start = 0
    overhead = len(to_markdown(table, rows=()))

    for i, row in enumerate(table.rows):
        candidate = [*group, row]
        rendered = overhead + sum(len(" | ".join(r)) + 4 for r in candidate)
        if rendered > max_chars and len(group) >= MIN_ROWS_PER_CHUNK:
            chunks.append(_group_chunk(table, group, start, i - 1, len(chunks)))
            group, start = [row], i
        else:
            group = candidate

    if group:
        chunks.append(_group_chunk(table, group, start, len(table.rows) - 1, len(chunks)))
    return chunks


def _group_chunk(
    table: ExtractedTable,
    rows: list[tuple[str, ...]],
    first: int,
    last: int,
    ordinal: int,
) -> TableChunk:
    return TableChunk(
        chunk_id=f"{table.table_id}#g{ordinal}",
        table_id=table.table_id,
        source_id=table.source_id,
        pages=table.pages,
        text=to_markdown(table, rows=tuple(rows)),
        serialisation="markdown",
        header=table.header,
        row_span=(first, last),
        clause_id=table.clause_id,
        meta={"spans_pages": table.spans_pages, "split": True},
    )


# --------------------------------------------------------------------------------------
# document-level entry point
# --------------------------------------------------------------------------------------


@dataclass
class TableExtractionReport:
    source_id: str
    fragments: int = 0
    tables: int = 0
    stitched: int = 0
    page_spanning: int = 0
    # Detected stitches, split by what they actually are. Reporting `stitched` alone
    # overstates the result: 16 detected, only 7 trustworthy data grids.
    stitch_classes: dict[str, int] = field(default_factory=dict)
    headerless: list[str] = field(default_factory=list)
    rows: int = 0
    header_repeats_dropped: int = 0

    @property
    def ok(self) -> bool:
        return not self.headerless


def extract_tables(
    source_id: str, doc: pymupdf.Document, *, max_pages: int | None = None
) -> tuple[list[ExtractedTable], TableExtractionReport]:
    """Extract and stitch every table in a document.

    Headerless tables are collected in the report rather than raised here: the caller
    decides whether to fail the whole ingest or exclude specific tables. Chunk derivation
    is where the hard stop lives, because that is where an unusable chunk would be created.
    """
    report = TableExtractionReport(source_id=source_id)
    fragments: list[TableFragment] = []

    with span("ingest.tables"):
        limit = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
        for i in range(limit):
            fragments.extend(extract_fragments(source_id, doc[i], i + 1))
    report.fragments = len(fragments)

    tables = stitch(fragments)
    report.tables = len(tables)
    report.stitched = sum(1 for t in tables if t.stitched_from > 1)
    report.page_spanning = sum(1 for t in tables if t.spans_pages)
    report.rows = sum(len(t.rows) for t in tables)
    report.header_repeats_dropped = sum(t.header_repeats_dropped for t in tables)
    report.headerless = [t.table_id for t in tables if not t.has_header]
    report.stitch_classes = {
        cls.value: len(entries) for cls, entries in classify_stitches(tables).items()
    }

    log.info(
        "tables_extracted",
        source_id=source_id,
        fragments=report.fragments,
        tables=report.tables,
        stitched=report.stitched,
        stitch_classes=report.stitch_classes,
        headerless=len(report.headerless),
    )
    return tables, report
