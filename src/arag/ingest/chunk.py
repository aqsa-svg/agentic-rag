"""Clause-aware prose chunking: where a chunk *ends* is a correctness decision.

## Why this module exists separately from the table chunker

Tables fail loudly - a row without its header is obviously meaningless, so
``HeaderlessTableError`` can stop ingest. Prose fails silently. A prose chunk that ends in
the wrong place is well-formed, retrievable, embeds fine, and returns a confidently wrong
answer. There is no downstream detector for it. So the boundary rules live here, with
tests that assert on the boundaries themselves rather than on chunk counts.

## R2: the split that must never happen

``def.hospital`` on page 4 of ``star-comprehensive-2025`` qualifies a facility two ways,
joined by a single word:

    ...registered as a hospital with the local authorities under the Clinical
    Establishments (Registration and Regulation) Act, 2010 ... **Or** complies with all
    minimum criteria as under: i. has qualified nursing staff ... v. maintains daily
    records ...

Split that definition at any fixed character budget and one chunk carries only the second
limb. Retrieved alone it reads as the complete test, so a facility that qualifies under
the *first* limb is answered "not a hospital" - with a correct-looking citation to the
right page of the right document. That is the worst failure this corpus can produce.

Hence two hard rules, which take precedence over ``max_chars``:

1. A definition is emitted **whole**, however long it is.
2. A clause is split **only** at sub-clause boundaries. A single sub-clause that is
   already over budget is emitted whole and counted in ``ChunkReport.oversized_split``.

``max_chars`` is therefore a target, not a limit. That is deliberate: an over-budget chunk
costs context window, a mis-split one costs a wrong answer.

## R2 also means: a table region is not a hole in the prose

Two tables in ``irdai-master-circular-2024`` (pages 10 and 11) have no detectable header
and are *refused* by the table chunker. If the prose chunker also skipped those page
regions - the obvious "don't index a table twice" optimisation - their content would be in
neither index, and the gap would be invisible: retrieval simply returns nothing for those
pages and it looks like a relevance problem. So this chunker never subtracts a region from
a page, ``table_regions`` is an input it *checks coverage against* rather than excludes,
and ``ChunkReport.pages_uncovered`` must be empty for every document in the corpus.

The cost is deliberate redundancy: text inside a table bbox is indexed both as a table
chunk and as prose. Duplication is recoverable at rank time; a gap is not recoverable at
all.

## What this chunker is not

v1 baseline: line-based, no cross-page layout model, and no de-duplication of the running
header PyMuPDF emits at the top of every page. Both are visible in the output and
measurable. A silently mis-split definition is neither, which is where the effort went.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from arag.eval.concepts import slug_term, validate_clause_id
from arag.ingest.chunk_types import Chunk, ChunkKind
from arag.ingest.clause_index import ID_PATTERNS, PAGE_NUMBER, ids_on_page
from arag.ingest.normalise import normalise_text
from arag.obs import get_logger, span
from arag.retrieval.filtering import meta_from_source
from arag.retrieval.types import DocSpan

log = get_logger(__name__)

# Characters, not tokens - matching MAX_TABLE_CHUNK_CHARS in tables.py. A character budget
# is deterministic and needs no tokeniser, and the embedder is not chosen yet.
MAX_PROSE_CHUNK_CHARS = 1200

# Which ID_PATTERNS open a clause. Only the line-ANCHORED ones qualify: `compound`
# ("II - Section 6 m.") and `excl_code` ("Code Excl 02") are *searched* in clause_index
# because they trail a clause title mid-line, so anchoring a chunk boundary to them would
# open a clause in the middle of a sentence. They stay useful as identifiers; they are not
# usable as boundaries.
HEADING_PATTERNS: tuple[str, ...] = ("section", "numbered")
DEFINITION_PATTERN = "definition"

# A line that recurs on this share of a document's pages (and on at least this many) is
# running furniture - a header or footer - not content structure.
#
# This is not cosmetic. "Product Name: ReAssure 2.0 | Product UIN: NBHHLIP27054V032627" is
# the page header on all 34 pages of nivabupa-reassure2 AND it matches the definition
# pattern, so without this filter it opens a bogus `def.product_name` on every page of the
# definitions section - five chunks in that one document, each claiming the same clause id
# and swallowing the real numbered definitions that follow it. Furniture is left in the
# chunk text (removing it would drop characters the coverage ratio is meant to account
# for); it is only barred from opening a clause.
RUNNING_LINE_MIN_PAGES = 3
RUNNING_LINE_SHARE = 0.5

# "I - DEFINITIONS", "STANDARD DEFINITIONS", "2.1. Specific Definitions:" - the heading
# that opens the definitions section. Anchored at BOTH ends: "Definitions of the terms
# used in this policy" is prose about definitions, not the section heading, and treating
# it as one would emit def.* chunks for every "Note:" line in the rest of the document.
DEFINITIONS_HEADING = re.compile(
    r"^\s{0,8}(?:(?:[IVXLCDM]+|\d{1,2}(?:\.\d{1,2})*)[\s.)\-]*)?"
    r"(?:standard|specific|general)?\s*definitions?\s*:?\s*$",
    re.IGNORECASE,
)

# An enumerated sub-clause: "A.", "b)", "i.", "(iv)", "3.". The ONLY legal split points
# inside a clause.
#
# Bullet glyphs are deliberately excluded. In this corpus's extracted text the enumerator
# and the text it introduces land on separate lines ("ii.\t" then "\x07Which would have
# otherwise required..."), so treating a bullet as a boundary would cut between a marker
# and its own content - a split mid-sub-clause, exactly what R2 forbids.
SUBCLAUSE_MARKER = re.compile(r"^[ \t]{0,10}\(?(?:[ivxlcdm]{1,5}|[A-Za-z]|\d{1,2})[.)][ \t]")


class PdfPage(Protocol):
    """The one PyMuPDF page method this module calls."""

    def get_text(self, option: str = "text", /) -> str: ...


class PdfDocument(Protocol):
    """Structural type for ``pymupdf.Document``.

    A Protocol rather than an import. PyMuPDF ships no type stubs, so importing it here
    would need a mypy override for this module, and every boundary test would need a real
    PDF on disk. With a Protocol the R2 tests are driven by a three-line fake document,
    which is what makes them readable enough to trust.
    """

    @property
    def page_count(self) -> int: ...

    def __getitem__(self, index: int, /) -> PdfPage: ...


@dataclass
class ChunkReport:
    """What chunking did, in numbers a reviewer can check against the corpus.

    ``pages_uncovered`` is the load-bearing field: it is the only evidence that no page of
    the corpus fell through the gap between the table chunker and this one.
    """

    source_id: str
    prose_chunks: int = 0
    definition_chunks: int = 0
    pages_covered: set[int] = field(default_factory=set)
    pages_uncovered: list[int] = field(default_factory=list)
    chars_in: int = 0
    chars_out: int = 0
    # Chunks emitted ABOVE max_chars because the only split point available was inside a
    # sub-clause, or because they are definitions. R2 winning over the size budget,
    # counted rather than hidden.
    oversized_split: int = 0

    @property
    def ok(self) -> bool:
        return not self.pages_uncovered

    @property
    def chunks(self) -> int:
        return self.prose_chunks + self.definition_chunks


@dataclass(frozen=True)
class _Line:
    """One normalised source line, still carrying the page it came from."""

    page: int
    text: str


@dataclass(frozen=True)
class _Heading:
    clause_id: str | None
    depth: int
    pattern: str
    is_definition: bool


@dataclass
class _Block:
    """One clause or definition, before the size budget is applied."""

    kind: ChunkKind
    clause_id: str | None
    depth: int
    pattern: str
    section_path: tuple[str, ...]
    lines: list[_Line] = field(default_factory=list)
    # Indices into ``lines`` at which a sub-clause starts. The only legal split points.
    subclause_at: list[int] = field(default_factory=list)


def _joined_len(lines: list[_Line]) -> int:
    r"""Length of ``"\n".join(...)`` without building it.

    Used for the packing decision so packing stays O(lines) instead of O(lines**2). The
    emitted chunk's size is measured with ``len()`` on the real text, which can be shorter
    by the whitespace ``_render`` strips off the ends.
    """
    if not lines:
        return 0
    return sum(len(line.text) for line in lines) + len(lines) - 1


def _render(lines: list[_Line]) -> str:
    return "\n".join(line.text for line in lines).strip()


def _running_lines(lines: list[_Line]) -> frozenset[str]:
    """Lines that repeat across most pages: the running header and footer."""
    pages_by_text: dict[str, set[int]] = {}
    pages: set[int] = set()
    for line in lines:
        pages.add(line.page)
        stripped = line.text.strip()
        if stripped:
            pages_by_text.setdefault(stripped, set()).add(line.page)
    threshold = max(RUNNING_LINE_MIN_PAGES, int(len(pages) * RUNNING_LINE_SHARE))
    return frozenset(text for text, seen in pages_by_text.items() if len(seen) >= threshold)


def _recognise(
    text: str, *, in_definitions: bool, definitions_depth: int, furniture: frozenset[str]
) -> _Heading | None:
    """What clause or definition, if any, this line opens."""
    # A running page number ("3 / 47") appears on every page of this corpus and is
    # indistinguishable from a bare clause number. clause_index excludes it for the same
    # reason; without this, every page would open a clause named after its own number.
    if PAGE_NUMBER.match(text):
        return None
    if text.strip() in furniture:
        return None

    if in_definitions:
        match = ID_PATTERNS[DEFINITION_PATTERN].match(text)
        if match:
            try:
                term_id = validate_clause_id(f"def.{slug_term(match.group(1))}")
            except ValueError:
                # A term that will not canonicalise cannot be cited, so it does not get to
                # be a definition. The line stays prose and is still indexed.
                return None
            # A defined term sits one level below the definitions section that contains it,
            # so a sibling definition ends it. In a document that *numbers* its definitions
            # (nivabupa: "2.1.24. Medical Advice means...") that sibling arrives as a
            # numbered heading, and treating a definition as infinitely deep instead made
            # one definition swallow the next four.
            return _Heading(term_id, definitions_depth + 1, DEFINITION_PATTERN, is_definition=True)

    for name in HEADING_PATTERNS:
        match = ID_PATTERNS[name].match(text)
        if not match:
            continue
        raw = match.group(1)
        depth = len(raw.split(".")) if name == "numbered" else 1
        try:
            clause_id: str | None = validate_clause_id(raw)
        except ValueError:
            # Still a boundary, just an uncitable one. Folding the text into the PRECEDING
            # clause was the alternative and it is worse: the chunk would then carry a
            # clause_id that does not govern its content, which is a mis-citation rather
            # than a missing one.
            clause_id = None
        return _Heading(clause_id, depth, name, is_definition=False)
    return None


def _clause_ids_for(text: str, heading_clause_id: str | None) -> tuple[str, ...]:
    """Every citable identifier this chunk's text carries, heading id included.

    A clause has more than one name. The block that opens with ``3.`` on
    ``star-comprehensive-2025`` p32 also carries ``Code Excl 03`` — and the IRDAI code is
    the PORTABLE key, the one a labeller is told to prefer because it survives a version
    bump while the list number does not.

    Carrying only the heading id is silent-wrongness instance 9: a span labelled
    ``excl.03`` could not match the chunk that contains excl.03's text, so a correct
    retrieval scored zero and every metric reported a *retrieval* failure. Measured before
    this fix: **0 of 1,024 chunks carried an ``excl.NN`` id**, while 3 of the 7 golden items
    keyed spans on one.

    The extraction is ``clause_index.ids_on_page`` — the same function that builds
    ``clause_index.json``, deliberately not a second copy. Two implementations of "what
    identifiers does this text contain?" would drift, and the drift would reappear as
    exactly this bug: the labeller's index and the retriever's chunks disagreeing about
    what a clause is called.

    The heading id stays first and remains ``span.clause_id``, because it is the chunk's
    provenance — what the chunk *is* — while the others are additional names it answers to.
    """
    ids: list[str] = []
    if heading_clause_id:
        ids.append(heading_clause_id)
    for clause_id, _pattern in ids_on_page(text):
        if clause_id not in ids:
            ids.append(clause_id)
    return tuple(ids)


def _closes(current: _Block, heading: _Heading) -> bool:
    """Does ``heading`` end the open block?

    "A clause runs from its heading to the next heading of the same-or-higher level" - so a
    DEEPER heading does not close a clause, it becomes a sub-clause boundary inside it.
    That is what keeps clause 4 together with 4.1 and 4.2, instead of emitting three chunks
    of which the first is a bare title.

    Definitions need no special case here: they carry the depth of a definition-section
    sibling, and the enumerated limbs that R2 protects ("i.", "ii.", "a.", "b.") are
    sub-clause markers rather than clause headings, so they never reach this function.
    """
    if heading.is_definition:
        return True
    return heading.depth <= current.depth


def _document_lines(source_id: str, doc: PdfDocument) -> tuple[list[_Line], int]:
    """Every page's text, normalised, as lines tagged with the page they came from.

    ``strict_numerics=True``: this is corpus content on its way to the index, so the N2
    numeric-invariance guard runs and a footnote marker fusing onto a monetary value stops
    ingest. ``NumericCorruptionError`` is deliberately allowed to propagate - a corrupted
    sub-limit is not something to log and carry on from.
    """
    lines: list[_Line] = []
    chars_in = 0
    for index in range(doc.page_count):
        page_no = index + 1
        raw = doc[index].get_text("text") or ""
        chars_in += len(raw)
        text, _ = normalise_text(raw, context=f"{source_id} p{page_no}", strict_numerics=True)
        lines.extend(_Line(page_no, line) for line in text.splitlines())
    return lines, chars_in


def _blocks(lines: list[_Line]) -> list[_Block]:
    """Group lines into clauses and definitions."""
    blocks: list[_Block] = []
    current: _Block | None = None
    # (depth, clause_id) of the enclosing headings, for ChunkMeta.section_path.
    stack: list[tuple[int, str]] = []
    furniture = _running_lines(lines)
    in_definitions = False
    definitions_depth = 1

    for line in lines:
        opens_definitions = (
            bool(DEFINITIONS_HEADING.match(line.text)) and line.text.strip() not in furniture
        )
        heading = _recognise(
            line.text,
            in_definitions=in_definitions,
            definitions_depth=definitions_depth,
            furniture=furniture,
        )

        if opens_definitions:
            # The section heading is a clause boundary in its own right, whether or not it
            # carries a number - "I - DEFINITIONS" carries none.
            numbered = heading if heading is not None and not heading.is_definition else None
            heading = _Heading(
                clause_id=numbered.clause_id if numbered else None,
                depth=numbered.depth if numbered else 1,
                pattern="definitions_section",
                is_definition=False,
            )

        if heading is None:
            if current is None:
                current = _Block(ChunkKind.PROSE, None, 1, "unheaded", tuple(c for _, c in stack))
                blocks.append(current)
            elif SUBCLAUSE_MARKER.match(line.text):
                current.subclause_at.append(len(current.lines))
            current.lines.append(line)
            continue

        if current is not None and not _closes(current, heading):
            # A deeper heading is a sub-clause of the open clause, not a new clause.
            current.subclause_at.append(len(current.lines))
            current.lines.append(line)
            if opens_definitions:
                in_definitions, definitions_depth = True, heading.depth
            continue

        if opens_definitions:
            in_definitions, definitions_depth = True, heading.depth
        elif in_definitions and not heading.is_definition and heading.depth <= definitions_depth:
            in_definitions = False

        while stack and stack[-1][0] >= heading.depth:
            stack.pop()
        section_path = tuple(clause for _, clause in stack)
        if heading.clause_id and not heading.is_definition:
            stack.append((heading.depth, heading.clause_id))

        current = _Block(
            kind=ChunkKind.DEFINITION if heading.is_definition else ChunkKind.PROSE,
            clause_id=heading.clause_id,
            depth=heading.depth,
            pattern=heading.pattern,
            section_path=section_path,
            lines=[line],
        )
        blocks.append(current)

    return blocks


def _split(block: _Block, max_chars: int) -> list[list[_Line]]:
    """Cut an over-budget clause at sub-clause boundaries, and nowhere else (R2).

    Segments - the lead-in, then one per sub-clause - are packed greedily. A segment is
    never subdivided, so no piece can begin or end inside a sub-clause; a segment that is
    over budget on its own becomes its own over-budget piece.
    """
    bounds = sorted({i for i in block.subclause_at if 0 < i < len(block.lines)})
    if not bounds:
        return [block.lines]

    segments: list[list[_Line]] = []
    start = 0
    for bound in [*bounds, len(block.lines)]:
        if bound > start:
            segments.append(block.lines[start:bound])
        start = bound

    pieces: list[list[_Line]] = []
    current: list[_Line] = []
    current_len = 0
    for segment in segments:
        segment_len = _joined_len(segment)
        if current and current_len + 1 + segment_len > max_chars:
            pieces.append(current)
            current, current_len = list(segment), segment_len
        else:
            # Joined length is additive: concatenating two groups costs their two lengths
            # plus the one newline between them. Kept incremental so packing is linear.
            current_len = current_len + 1 + segment_len if current else segment_len
            current.extend(segment)
    if current:
        pieces.append(current)
    return pieces


def _pieces_of(block: _Block, max_chars: int) -> list[list[_Line]]:
    """The line groups this block becomes. A definition is always one group (R2)."""
    if block.kind is ChunkKind.DEFINITION or _joined_len(block.lines) <= max_chars:
        return [block.lines]
    return _split(block, max_chars)


def chunk_document(
    source_id: str,
    doc: PdfDocument,
    *,
    max_chars: int = MAX_PROSE_CHUNK_CHARS,
    table_regions: list[tuple[int, tuple[float, float, float, float]]] | None = None,
    source: object | None = None,
) -> tuple[list[Chunk], ChunkReport]:
    """Chunk one document's prose and definitions.

    ``table_regions`` is ``(page, bbox)`` pairs as produced by the table extractor. They
    are NOT excluded from the text: ``get_text("text")`` returns the whole page including
    everything inside a table bbox, so page coverage *is* region coverage for this
    extractor, and the bboxes serve only to record which chunks fall on a page carrying a
    declared table. Passing them is how a caller asks "did the prose fallback cover the
    tables you refused?" and gets the answer in ``pages_uncovered``.

    Raises ``normalise.NumericCorruptionError`` if normalising a page would change a
    numeric value.
    """
    report = ChunkReport(source_id=source_id)
    region_pages = {page for page, _bbox in table_regions or ()}
    chunks: list[Chunk] = []

    with span("ingest.chunk"):
        lines, report.chars_in = _document_lines(source_id, doc)

        for block in _blocks(lines):
            pieces = _pieces_of(block, max_chars)
            for part, piece in enumerate(pieces):
                text = _render(piece)
                if not text:
                    continue
                pages = tuple(sorted({line.page for line in piece}))
                definition = block.kind is ChunkKind.DEFINITION
                marker = "d" if definition else "s"
                chunks.append(
                    Chunk(
                        chunk_id=f"{source_id}:p{pages[0]}:{marker}{len(chunks)}",
                        source_id=source_id,
                        kind=block.kind,
                        text=text,
                        span=DocSpan(
                            document_id=source_id, page=pages[0], clause_id=block.clause_id
                        ),
                        pages=pages,
                        meta=meta_from_source(
                            source,
                            section_path=block.section_path,
                            is_table=False,
                            clause_ids=_clause_ids_for(text, block.clause_id),
                        ),
                        detail={
                            "pattern": block.pattern,
                            "part": part,
                            "parts": len(pieces),
                            "oversized": len(text) > max_chars,
                            "covers_table_region": bool(region_pages.intersection(pages)),
                        },
                    )
                )

    for chunk in chunks:
        report.pages_covered.update(chunk.pages)
        report.chars_out += chunk.char_len
        if chunk.kind is ChunkKind.DEFINITION:
            report.definition_chunks += 1
        else:
            report.prose_chunks += 1
        if chunk.detail["oversized"]:
            report.oversized_split += 1
    report.pages_uncovered = [
        page for page in range(1, doc.page_count + 1) if page not in report.pages_covered
    ]

    log.info(
        "chunked",
        source_id=source_id,
        prose=report.prose_chunks,
        definitions=report.definition_chunks,
        oversized=report.oversized_split,
        pages_uncovered=report.pages_uncovered,
        chars_in=report.chars_in,
        chars_out=report.chars_out,
    )
    if report.pages_uncovered:
        log.warning(
            "chunk_pages_uncovered",
            source_id=source_id,
            pages=report.pages_uncovered,
            table_region_pages=sorted(region_pages.intersection(report.pages_uncovered)),
        )
    return chunks, report
