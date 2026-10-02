"""The corpus build, and the measurement that decides how tables are serialised.

Two things live here because they are the same computation run twice.

**The build.** ``chunk_document`` produces prose and definitions; ``extract_tables`` plus
``table_chunks`` produce the table chunks. Neither is the corpus. This module is where
they are combined into one addressable chunk store with one index over it, and where the
build reports what it *covered*. Coverage is the point, not decoration: 2 tables in
``irdai-master-circular-2024`` (p10, p11) are refused for having no header, and their
pages must still arrive through prose chunking (requirement R2). A silently dropped page
is undetectable downstream - a query about it retrieves plausible neighbours and the
answer is confidently wrong with a real citation attached - so ``pages_uncovered`` is
computed against an *independent* page census (the probe report's 197 pages), never
against the chunker's own idea of which pages it walked. Auditing the chunker with the
chunker's own count would pass a chunker that skipped a page and also forgot to count it.

**The T3 harness.** Markdown and row-level-NL table serialisations are built and measured
against each other. On this corpus row_nl emits roughly 8x the table chunks markdown does
(1735 vs 222 measured), so the two arms could plausibly hand the generator very
differently sized context windows at one fixed ``top_k``. Whether they actually do is a
measurement, not an inference, and it is the measurement that matters: a comparison run on
retrieval quality alone would pick a serialisation without knowing what it costs.
``*_tokens_at_k`` is the field that puts the bill next to the benefit, and on the first
full run it contradicted the chunk counts - 1393 vs 1450 tokens at k=6, a ratio of 0.96
against a chunk ratio of 7.8. That contradiction is the reason the harness exists rather
than a rule of thumb about chunk sizes.

Every collaborating module is imported defensively. That is not hypothetical caution:
PyMuPDF (and therefore ``arag.ingest.tables``) lives in the ``offline`` extra and is
absent from the PR job and from the deployed function on purpose. An unguarded import
would make this module - and the tests that pin its arithmetic - unimportable wherever the
extra is not installed. A missing collaborator is instead a named ``RuntimeError`` raised
by the function that needs it, at the point of use, saying which module is absent.
"""

from __future__ import annotations

import asyncio
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from arag.ingest.chunk_types import Chunk, ChunkKind
from arag.ingest.manifest import Manifest
from arag.obs import get_logger, span
from arag.retrieval.filtering import meta_from_source
from arag.retrieval.protocols import Retriever
from arag.retrieval.types import DocSpan

log = get_logger(__name__)

# The two T3 arms, mapped to the chunk kind each produces. A dict rather than a second
# enum because ``ChunkKind`` already distinguishes them; a third spelling of the same two
# values is a synchronisation bug waiting to happen.
TABLE_KINDS = {
    "markdown": ChunkKind.TABLE_MARKDOWN,
    "row_nl": ChunkKind.TABLE_ROW_NL,
}
SERIALISATIONS = tuple(TABLE_KINDS)

# A rough chars-per-token proxy, used because the real tokeniser is not a v1 dependency:
# the online arm runs ``tokenizers`` and the offline arm does not, and pulling it in here
# just to size a context window would make this harness un-runnable in the PR job. It is
# fit for purpose because the number it feeds is a *ratio* between two serialisations of
# the same corpus, where a constant multiplicative bias cancels.
CHARS_PER_TOKEN = 4

# Latency is reported to microsecond resolution. Anything finer is measurement noise on a
# wall clock, and full float repr in a report makes two runs impossible to eyeball.
LATENCY_DECIMALS = 3

# The page census lives beside the manifest, written by ``arag-ingest probe``.
PROBE_REPORT_NAME = "probe_report.json"


# --------------------------------------------------------------------------------------
# collaborators owned by other modules
#
# Every assumption this module makes about somebody else's signature is in this section
# and nowhere else, so a signature disagreement is a one-place fix rather than a hunt.
# --------------------------------------------------------------------------------------

PYMUPDF_MODULE = "pymupdf"
TABLES_MODULE = "arag.ingest.tables"
CHUNKER_MODULE = "arag.ingest.chunk"
LEXICAL_INDEX_MODULE = "arag.index.lexical"
LEXICAL_RETRIEVER_MODULE = "arag.retrieval.lexical"

_MISSING_MODULES: dict[str, str] = {}

try:
    import pymupdf
except ImportError as exc:  # pragma: no cover - depends on whether the extra is installed
    _MISSING_MODULES[PYMUPDF_MODULE] = str(exc)

try:
    from arag.ingest.tables import ExtractedTable, TableChunk, extract_tables, table_chunks
except ImportError as exc:  # pragma: no cover - depends on whether the extra is installed
    _MISSING_MODULES[TABLES_MODULE] = str(exc)

try:
    from arag.ingest.chunk import chunk_document
except ImportError as exc:  # pragma: no cover - depends on what has landed
    _MISSING_MODULES[CHUNKER_MODULE] = str(exc)

try:
    from arag.index.lexical import Bm25Index
except ImportError as exc:  # pragma: no cover - depends on what has landed
    _MISSING_MODULES[LEXICAL_INDEX_MODULE] = str(exc)

try:
    from arag.retrieval.lexical import LexicalRetriever
except ImportError as exc:  # pragma: no cover - depends on what has landed
    _MISSING_MODULES[LEXICAL_RETRIEVER_MODULE] = str(exc)


def _require(*modules: str) -> None:
    """Fail with a named, actionable error when a collaborator is absent.

    ``RuntimeError`` rather than re-raising ``ImportError`` on purpose: an ImportError
    surfacing from inside ``build_corpus`` reads as "arag is broken" and sends the reader
    to this file, when the fact is that a *different* module, or an uninstalled extra, is
    what is missing.
    """
    missing = [(name, _MISSING_MODULES[name]) for name in modules if name in _MISSING_MODULES]
    if not missing:
        return
    detail = "; ".join(f"{name} ({why})" for name, why in missing)
    raise RuntimeError(
        f"corpus build needs modules that are not importable: {detail}. "
        f"{PYMUPDF_MODULE} and {TABLES_MODULE} come from the 'offline' extra "
        "(pip install -e '.[offline]'), which is deliberately absent from the PR job and "
        "from the deployed function; the other arag modules ship with the package. This "
        "module owns none of them."
    )


def _as_chunk(table_chunk: TableChunk, kind: ChunkKind, source: object | None = None) -> Chunk:
    """Lift a ``TableChunk`` into the ``Chunk`` the index and the eval both key on.

    ``TableChunk`` is the table chunker's own record and carries table-shaped fields
    (``row_span``, ``header``) that nothing downstream filters on, so they move into
    ``detail`` - the diagnostic surface - rather than into ``ChunkMeta``, which is the
    retrieval-filterable one.
    """
    return Chunk(
        chunk_id=table_chunk.chunk_id,
        source_id=table_chunk.source_id,
        kind=kind,
        text=table_chunk.text,
        span=DocSpan(
            document_id=table_chunk.source_id,
            page=table_chunk.pages[0],
            clause_id=table_chunk.clause_id,
        ),
        pages=table_chunk.pages,
        meta=meta_from_source(source, is_table=True),
        detail={
            "table_id": table_chunk.table_id,
            "serialisation": table_chunk.serialisation,
            "row_span": table_chunk.row_span,
            "header": list(table_chunk.header),
            **table_chunk.meta,
        },
    )


def _table_chunks_for(
    source_id: str,
    tables: Sequence[ExtractedTable],
    serialisation: str,
    source: object | None = None,
) -> list[Chunk]:
    """Serialise every *usable* table, skipping the header-less ones.

    Skipping is an explicit exclusion, which ``tables.py`` sanctions ("exclude the table
    explicitly; do not index it"). The alternative - letting ``HeaderlessTableError``
    propagate - would abort a build over 2 refused tables while 134 usable ones and all
    the prose are fine, and it is the prose fallback (R2), not a dead build, that makes
    the refused pages answerable.
    """
    kind = TABLE_KINDS[serialisation]
    chunks: list[Chunk] = []
    for table in tables:
        if not table.has_header:
            continue
        chunks.extend(
            _as_chunk(derived, kind, source)
            for derived in table_chunks(table, serialisation=serialisation)
        )
    return chunks


def _open_document(pdf_path: Path) -> pymupdf.Document:
    """This module's only call into PyMuPDF, isolated deliberately.

    PyMuPDF's ``Document.__init__`` carries no annotations, so any call to it is an
    untyped call and this module needs ``disallow_untyped_calls = false`` in the mypy
    overrides - the same relaxation ``arag.ingest.probe`` and ``arag.ingest.tables``
    already carry. Confining the untyped surface to this one line bounds what that
    relaxation is able to hide: everything else in the module stays fully checked.
    """
    return pymupdf.open(pdf_path)


def _document_chunks(
    source_id: str, pdf_path: Path, serialisation: str, source: object | None = None
) -> tuple[list[Chunk], int]:
    """Every chunk for one document, plus how many of its tables were refused.

    ``table_regions`` is deliberately not passed to ``chunk_document``: ``extract_tables``
    discards fragment bboxes when it stitches, so recovering them would mean a second
    ``find_tables()`` pass over every page - the expensive half of ingest - to set a
    diagnostic flag. What R2 needs is page-level, and ``get_text("text")`` returns the
    whole page including everything inside a table bbox, so page coverage *is* region
    coverage for this extractor (see ``chunk_document``'s own note).
    """
    with _open_document(pdf_path) as doc:
        tables, table_report = extract_tables(source_id, doc)
        prose_chunks, prose_report = chunk_document(source_id, doc, source=source)

    chunks = [*prose_chunks, *_table_chunks_for(source_id, tables, serialisation, source)]

    # The R2 alarm, stated over the pages that actually matter. `pages_uncovered` below
    # catches the same hole from the census side; this line says *why* a page is missing.
    refused_pages = {page for table in tables if not table.has_header for page in table.pages}
    orphaned = sorted(refused_pages - prose_report.pages_covered)
    if orphaned:
        log.error(
            "refused_table_pages_not_covered_by_prose",
            source_id=source_id,
            pages=orphaned,
            detail="R2 violated: a refused table's numbers are in the corpus with no text",
        )

    log.info(
        "document_chunked",
        source_id=source_id,
        serialisation=serialisation,
        prose_chunks=prose_report.prose_chunks,
        definition_chunks=prose_report.definition_chunks,
        table_chunks=len(chunks) - len(prose_chunks),
        tables=table_report.tables,
        tables_refused=len(table_report.headerless),
        pages_uncovered=prose_report.pages_uncovered,
    )
    return chunks, len(table_report.headerless)


def _build_lexical_index(chunks: Mapping[str, Chunk]) -> Bm25Index:
    index = Bm25Index()
    for chunk in chunks.values():
        index.add(chunk)
    index.finalise()
    return index


def _lexical_retriever(corpus: CorpusIndex) -> Retriever:
    retriever: Retriever = LexicalRetriever(corpus.index, corpus.chunks)
    return retriever


# --------------------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildReport:
    """What one build of the corpus produced, and what it failed to cover.

    Frozen because it is a measurement. A report that can be edited after the run is a
    number nobody can cite, and this one is cited: ``pages_uncovered`` is the evidence for
    R2 and ``total_chunks`` is half of the T3 cost argument.
    """

    serialisation: str
    documents: int
    prose_chunks: int
    definition_chunks: int
    table_chunks: int
    total_chunks: int
    total_chars: int
    index_bytes: int
    build_seconds: float
    tables_refused: int
    # Only sources with a gap appear, so an empty dict means full page coverage and
    # ``if report.pages_uncovered:`` is the R2 check. Listing every source with an empty
    # list would make the failure state indistinguishable from the healthy one at a glance.
    pages_uncovered: dict[str, list[int]]

    @property
    def fully_covered(self) -> bool:
        return not self.pages_uncovered

    @property
    def mean_chunk_chars(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return round(self.total_chars / self.total_chunks, 1)


@dataclass
class CorpusIndex:
    """A built corpus: the chunks by id, the index over them, and how it was built.

    Not frozen - it owns a live ``Bm25Index``, and declaring the whole aggregate immutable
    while holding a mutable index would be a claim nothing enforces.
    """

    chunks: dict[str, Chunk]
    index: Bm25Index
    serialisation: str
    report: BuildReport


@dataclass(frozen=True)
class QueryMeasurement:
    """Per-serialisation query cost. ``tokens_at_k`` is the field the T3 decision turns on."""

    queries: int
    top_k: int
    latency_ms: dict[str, float]
    tokens_at_k: int


@dataclass(frozen=True)
class SerialisationComparison:
    """The T3 result: two builds of the same corpus, measured side by side."""

    markdown: BuildReport
    row_nl: BuildReport
    # per-serialisation query measurements
    markdown_latency_ms: dict[str, float]
    row_nl_latency_ms: dict[str, float]
    markdown_tokens_at_k: int  # tokens sent to the generator at top_k
    row_nl_tokens_at_k: int
    top_k: int
    queries: int

    @property
    def table_chunk_ratio(self) -> float:
        """row_nl table chunks per markdown table chunk: the size of the T3 effect.

        Measured 1735/222 = 7.8 on this corpus. This, not ``chunk_ratio``, is the ratio
        the serialisation actually controls - the prose and definition chunks are produced
        identically in both arms, so including them measures the corpus rather than the
        decision.
        """
        if self.markdown.table_chunks == 0:
            return 0.0
        return self.row_nl.table_chunks / self.markdown.table_chunks

    @property
    def chunk_ratio(self) -> float:
        """row_nl chunks per markdown chunk over the whole corpus. Measured 2537/1024 = 2.5.

        Reported alongside ``table_chunk_ratio`` because the two answer different
        questions and quoting the wrong one misleads in a predictable direction: 802 of
        these chunks are prose and definitions, identical in both arms, which dilutes a
        7.8x difference in the table region down to 2.5x across the corpus. Index size and
        build time scale with this figure; the serialisation choice is visible in the
        other one.
        """
        if self.markdown.total_chunks == 0:
            return 0.0
        return self.row_nl.total_chunks / self.markdown.total_chunks

    @property
    def token_ratio(self) -> float:
        """row_nl context per markdown context at the same ``top_k``.

        Independent of both chunk ratios, and the only one of the three that is a cost.
        More, smaller chunks at a fixed k does not have to mean more tokens: on this
        corpus at k=6 it measured 1393 vs 1450, a ratio of 0.96 - so the "8x the chunks
        means 8x the context bill" inference that the chunk counts invite is wrong by
        roughly a factor of 8. Measuring at k is what makes the difference visible.
        """
        if self.markdown_tokens_at_k == 0:
            return 0.0
        return self.row_nl_tokens_at_k / self.markdown_tokens_at_k


# --------------------------------------------------------------------------------------
# measurement primitives (pure, and therefore the part with hand-computed tests)
# --------------------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile: the smallest observed value at or above rank ``q``.

    ``index = ceil(q * n) - 1`` over the sorted samples. Nearest-rank rather than linear
    interpolation because every value returned is a latency that actually happened; an
    interpolated p95 is a number no query ever took, which is awkward to defend in a
    report. Note the consequence: ``percentile(x, 0.5)`` is the lower median for even
    ``n``, not the average of the middle pair.
    """
    if not values:
        raise ValueError("percentile of an empty sample is undefined")
    if not 0.0 < q <= 1.0:
        raise ValueError(f"q must be in (0, 1], got {q}")
    ordered = sorted(values)
    rank = math.ceil(q * len(ordered))
    return ordered[min(max(rank, 1), len(ordered)) - 1]


def latency_summary(samples_ms: Sequence[float]) -> dict[str, float]:
    """mean / p50 / p95 over latency samples, in milliseconds.

    Raises on an empty sample rather than returning zeros: a report reading "mean 0.0 ms"
    is indistinguishable from an impossibly fast index, and would be believed.
    """
    if not samples_ms:
        raise ValueError("no latency samples to summarise")
    return {
        "mean": round(statistics.fmean(samples_ms), LATENCY_DECIMALS),
        "p50": round(percentile(samples_ms, 0.5), LATENCY_DECIMALS),
        "p95": round(percentile(samples_ms, 0.95), LATENCY_DECIMALS),
    }


def estimate_tokens(texts: Iterable[str]) -> int:
    """Estimated generator tokens for ``texts``: total characters // 4.

    ``/4`` is a rough chars-per-token proxy, chosen because the real tokeniser is not a
    dependency at v1 (see :data:`CHARS_PER_TOKEN`). Total-then-divide, not
    divide-then-total, so that splitting one chunk in two does not change the estimate.
    """
    return sum(len(text) for text in texts) // CHARS_PER_TOKEN


def index_size_bytes(index: Bm25Index) -> int:
    """Bytes of the index's JSON serialisation - the figure that decides where it can ship.

    ``ensure_ascii=False`` because this measures the payload actually written to disk;
    escaping would charge 6 bytes for every non-ASCII character that UTF-8 stores in 2 or
    3, inflating the number for a Devanagari or rupee-bearing term for no reason. Sorted
    keys so two builds of the same corpus produce the same byte count.
    """
    return len(json.dumps(index.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8"))


def read_page_census(manifest_path: Path) -> dict[str, int]:
    """Pages per document, read from the probe report beside the manifest.

    The census is deliberately *not* derived from the chunker or from the PDFs at build
    time. It is the corpus's independently measured page count (197 across 6 documents),
    which is what makes ``pages_uncovered`` an audit rather than a tautology.
    """
    path = manifest_path.parent / PROBE_REPORT_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"no page census at {path}. Page coverage (R2) cannot be verified without one, "
            "and an unverified build would report zero uncovered pages whether or not the "
            "refused table regions on irdai-master-circular-2024 p10-p11 were chunked. "
            "Run `arag-ingest probe` first."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            f"{path} should be a list of per-document probes, got {type(raw).__name__}"
        )

    census: dict[str, int] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: expected probe objects, got {type(entry).__name__}")
        source_id = entry.get("source_id")
        pages = entry.get("pages")
        if not isinstance(source_id, str) or not isinstance(pages, int):
            raise ValueError(f"{path}: probe entry missing 'source_id'/'pages': {entry!r}")
        census[source_id] = pages
    return census


def uncovered_pages(
    chunks: Iterable[Chunk], page_census: Mapping[str, int]
) -> dict[str, list[int]]:
    """Census pages that no chunk claims, per source. Sources with no gap are omitted."""
    covered: dict[str, set[int]] = defaultdict(set)
    for chunk in chunks:
        covered[chunk.source_id].update(chunk.pages)

    for source_id, pages in covered.items():
        total = page_census.get(source_id)
        if total is None:
            # A chunk from a document the census does not know about. Its coverage cannot
            # be checked, and staying quiet would report it as fully covered.
            log.warning("source_absent_from_page_census", source_id=source_id)
        elif any(page > total for page in pages):
            log.warning(
                "chunk_pages_beyond_document",
                source_id=source_id,
                page_count=total,
                pages=sorted(page for page in pages if page > total),
            )

    gaps: dict[str, list[int]] = {}
    for source_id, total in page_census.items():
        missing = sorted(set(range(1, total + 1)) - covered.get(source_id, set()))
        if missing:
            gaps[source_id] = missing
    return gaps


def check_serialisation(serialisation: str) -> None:
    if serialisation not in SERIALISATIONS:
        raise ValueError(
            f"unknown serialisation {serialisation!r}, expected one of {SERIALISATIONS}"
        )


def build_report(
    chunks: Sequence[Chunk],
    *,
    serialisation: str,
    documents: int,
    page_census: Mapping[str, int],
    index_bytes: int,
    build_seconds: float,
    tables_refused: int,
) -> BuildReport:
    """Summarise a set of chunks into a :class:`BuildReport`.

    Split out from :func:`build_corpus` so that every number in the report can be pinned
    against hand-computed values without a 197-page PDF pass.
    """
    check_serialisation(serialisation)

    by_kind = Counter(chunk.kind for chunk in chunks)
    prose = by_kind[ChunkKind.PROSE]
    definitions = by_kind[ChunkKind.DEFINITION]
    tables = sum(count for kind, count in by_kind.items() if kind.is_table)
    total = len(chunks)

    # The three buckets must partition the chunks. This is not paranoia about addition:
    # adding a fifth ChunkKind (a figure caption, say) would otherwise silently produce a
    # report whose parts no longer sum to its total, and every downstream cost comparison
    # would be quietly understated. Failing here forces the new kind into the report.
    if prose + definitions + tables != total:
        unaccounted = sorted(
            str(kind)
            for kind in by_kind
            if kind not in (ChunkKind.PROSE, ChunkKind.DEFINITION) and not kind.is_table
        )
        raise ValueError(
            f"chunk kinds do not partition the corpus: {prose} prose + {definitions} "
            f"definition + {tables} table != {total} total; unaccounted kinds: {unaccounted}"
        )

    return BuildReport(
        serialisation=serialisation,
        documents=documents,
        prose_chunks=prose,
        definition_chunks=definitions,
        table_chunks=tables,
        total_chunks=total,
        total_chars=sum(chunk.char_len for chunk in chunks),
        index_bytes=index_bytes,
        build_seconds=round(build_seconds, LATENCY_DECIMALS),
        tables_refused=tables_refused,
        pages_uncovered=uncovered_pages(chunks, page_census),
    )


# --------------------------------------------------------------------------------------
# the build
# --------------------------------------------------------------------------------------


def build_corpus(
    manifest_path: Path, raw_dir: Path, *, serialisation: str = "markdown"
) -> CorpusIndex:
    """Chunk every manifest source, index the result, and report the coverage.

    A source whose PDF is absent is skipped with a warning rather than raised on, because
    the coverage report already catches it far more usefully: every page of the missing
    document turns up in ``pages_uncovered``, which is the truth (nothing was indexed for
    it) instead of an exception that says nothing about the other five documents.
    """
    _require(PYMUPDF_MODULE, TABLES_MODULE, CHUNKER_MODULE, LEXICAL_INDEX_MODULE)
    check_serialisation(serialisation)

    started = perf_counter()
    manifest = Manifest.load(manifest_path)
    page_census = read_page_census(manifest_path)

    chunks: dict[str, Chunk] = {}
    documents = 0
    tables_refused = 0

    with span("index.build"):
        for source in manifest.sources:
            pdf_path = raw_dir / f"{source.id}.pdf"
            if not pdf_path.exists():
                log.warning("source_pdf_missing", source_id=source.id, path=str(pdf_path))
                continue

            source_chunks, refused = _document_chunks(source.id, pdf_path, serialisation, source)
            documents += 1
            tables_refused += refused

            for chunk in source_chunks:
                # A collision would silently drop a chunk and take its citable span with
                # it, leaving the index missing a passage no metric could name.
                if chunk.chunk_id in chunks:
                    raise ValueError(
                        f"duplicate chunk_id {chunk.chunk_id!r} from {source.id} "
                        f"(already claimed by {chunks[chunk.chunk_id].source_id})"
                    )
                chunks[chunk.chunk_id] = chunk

    with span("index.lexical"):
        index = _build_lexical_index(chunks)

    report = build_report(
        list(chunks.values()),
        serialisation=serialisation,
        documents=documents,
        page_census=page_census,
        index_bytes=index_size_bytes(index),
        build_seconds=perf_counter() - started,
        tables_refused=tables_refused,
    )
    log.info(
        "corpus_built",
        serialisation=serialisation,
        documents=report.documents,
        total_chunks=report.total_chunks,
        total_chars=report.total_chars,
        index_bytes=report.index_bytes,
        build_seconds=report.build_seconds,
        tables_refused=report.tables_refused,
        uncovered_sources=sorted(report.pages_uncovered),
    )
    return CorpusIndex(chunks=chunks, index=index, serialisation=serialisation, report=report)


# --------------------------------------------------------------------------------------
# T3: the serialisation comparison
# --------------------------------------------------------------------------------------


async def measure_queries(
    retriever: Retriever,
    queries: Sequence[str],
    *,
    top_k: int = 6,
    repeats: int = 1,
) -> QueryMeasurement:
    """Time ``queries`` against ``retriever`` and estimate the context each one costs.

    Queries run one at a time. Concurrency would finish sooner and report latencies that
    include each other's queueing, which is the opposite of the per-query number wanted
    here. No warm-up sample is discarded either: the first query pays the cold-cache cost
    in both T3 arms identically, so dropping it would improve the absolute figures without
    changing the comparison the harness exists to make.

    ``repeats`` exists because p95 over a handful of samples degenerates - nearest-rank
    p95 of 6 samples *is* the maximum - so a caller wanting a distribution rather than a
    worst case can ask for more passes. Token cost is counted on the first pass only,
    since it does not vary between identical queries.
    """
    if not queries:
        raise ValueError("no queries to measure; a latency report over zero queries is empty")
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    samples_ms: list[float] = []
    tokens: list[int] = []

    for pass_index in range(repeats):
        for query in queries:
            start = perf_counter()
            hits = await retriever.retrieve(query, top_k=top_k)
            samples_ms.append((perf_counter() - start) * 1000.0)
            if pass_index == 0:
                # Slice defensively: a retriever is free to return more than top_k, and
                # "tokens at top_k" must mean top_k whatever the retriever hands back.
                tokens.append(estimate_tokens(hit.text for hit in hits[:top_k]))

    return QueryMeasurement(
        queries=len(queries),
        top_k=top_k,
        latency_ms=latency_summary(samples_ms),
        # Mean per-query cost, floored. Integer arithmetic throughout so the reported
        # figure is reproducible rather than a float that shifts with summation order.
        tokens_at_k=sum(tokens) // len(tokens),
    )


def compare_serialisations(
    manifest_path: Path,
    raw_dir: Path,
    queries: list[str],
    *,
    top_k: int = 6,
) -> SerialisationComparison:
    """Build the corpus once per serialisation and measure both, for the T3 decision.

    Reports index size, chunk and character counts, build wall-clock, query latency, and -
    the metric the whole harness is for - the estimated tokens each serialisation sends to
    the generator at the same ``top_k``. row_nl retrieving better is not by itself an
    argument for row_nl; it has to retrieve better *per token*.

    The two arms re-extract the same tables rather than sharing one extraction pass. That
    doubles the wall clock and is the correct choice anyway: ``build_seconds`` is one of
    the reported figures, and a cached second arm would report a build time no operator
    could ever reproduce.

    Synchronous, with :func:`asyncio.run` inside, because it is a harness entry point
    called from a CLI and from tests, never from inside a running event loop.
    """
    _require(
        PYMUPDF_MODULE,
        TABLES_MODULE,
        CHUNKER_MODULE,
        LEXICAL_INDEX_MODULE,
        LEXICAL_RETRIEVER_MODULE,
    )
    if not queries:
        raise ValueError("compare_serialisations needs at least one query to measure")

    reports: dict[str, BuildReport] = {}
    measurements: dict[str, QueryMeasurement] = {}

    for serialisation in SERIALISATIONS:
        with span(f"index.compare.{serialisation}"):
            corpus = build_corpus(manifest_path, raw_dir, serialisation=serialisation)
            reports[serialisation] = corpus.report
            measurements[serialisation] = asyncio.run(
                measure_queries(_lexical_retriever(corpus), queries, top_k=top_k)
            )

    comparison = SerialisationComparison(
        markdown=reports["markdown"],
        row_nl=reports["row_nl"],
        markdown_latency_ms=measurements["markdown"].latency_ms,
        row_nl_latency_ms=measurements["row_nl"].latency_ms,
        markdown_tokens_at_k=measurements["markdown"].tokens_at_k,
        row_nl_tokens_at_k=measurements["row_nl"].tokens_at_k,
        top_k=top_k,
        queries=len(queries),
    )
    log.info(
        "serialisation_compared",
        top_k=top_k,
        queries=len(queries),
        markdown_chunks=comparison.markdown.total_chunks,
        row_nl_chunks=comparison.row_nl.total_chunks,
        chunk_ratio=round(comparison.chunk_ratio, 3),
        markdown_tokens_at_k=comparison.markdown_tokens_at_k,
        row_nl_tokens_at_k=comparison.row_nl_tokens_at_k,
        token_ratio=round(comparison.token_ratio, 3),
    )
    return comparison
