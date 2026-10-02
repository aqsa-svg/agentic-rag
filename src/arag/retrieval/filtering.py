"""Pre-retrieval candidate narrowing, shared by every retriever.

## Why this is its own module rather than a method on each retriever

v2 adds a second retriever and fuses the two. If BM25 and the dense index applied even
slightly different filter semantics, RRF would be fusing two rankings taken over
*different corpora*, and the fused result would be incoherent in a way no metric reports:
recall would simply be lower than either retriever alone, with nothing pointing at the
cause.

The specific field this protects is ``as_of``. It is the supersession control — the
mechanism that keeps a clause replaced in 2025 out of an answer dated 2026 — and the v1
baseline already shows what its absence costs: h-01 retrieves ``star-comprehensive-2021``
at rank 1. A supersession filter that worked on the lexical side and not the dense side
would look like it was working, because half the candidates would be correctly filtered.

So the rule is structural: a retriever does not implement filtering. It calls
``candidate_ids`` and passes the result to its index.

## Pre-filter, not post-filter

``candidate_ids`` narrows the set *before* scoring. Filtering the result list afterwards
is the obvious alternative and it is wrong: a document-scoped query would come back short
because the global top-k was consumed by excluded documents, and "no such clause" is
indistinguishable from "the clause is in a document you filtered out".

``None`` means "no restriction" and is not the same as an empty frozenset, which means
"nothing matched". Conflating them turns a filter that excludes everything into a search
over everything — the failure mode that makes a filter look harmless while it does the
opposite of what was asked.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from arag.obs import get_logger
from arag.retrieval.types import ChunkMeta, RetrievalFilters

if TYPE_CHECKING:
    from collections.abc import Mapping

    from arag.ingest.chunk_types import Chunk

log = get_logger(__name__)


def is_active(filters: RetrievalFilters | None) -> bool:
    """Whether any filter field would narrow the corpus.

    Checked so the common unfiltered query does not pay for building a set of every chunk
    id, and so ``allow=None`` stays distinguishable from an empty allow-set.
    """
    if filters is None:
        return False
    return bool(
        filters.insurer
        or filters.product
        or filters.document_ids
        or filters.as_of is not None
        or filters.tables_only
    )


def in_force(meta: ChunkMeta, as_of: date) -> bool:
    """Whether the chunk's validity window contains ``as_of``.

    Only the dates decide. ``superseded_by`` is intentionally not consulted: it names the
    replacing clause but carries no date, so a chunk marked superseded with no
    ``effective_to`` cannot be placed in time at all. Excluding it would drop a clause that
    may well have been in force on ``as_of``; the ingest invariant is that supersession
    sets ``effective_to``, and a violation of that invariant belongs in an ingest failure
    rather than in a guess made silently at query time.
    """
    if meta.effective_from is not None and meta.effective_from > as_of:
        return False
    return meta.effective_to is None or meta.effective_to >= as_of


def matches(chunk: Chunk, filters: RetrievalFilters) -> bool:
    """Whether one chunk survives the filters."""
    if filters.insurer and chunk.meta.insurer != filters.insurer:
        return False
    if filters.product and chunk.meta.product != filters.product:
        return False
    # Matched against the span's document_id, not ``source_id``: the document id is what
    # appears in a citation, so it is the only id a caller can see well enough to filter
    # on. ``source_id`` is ingest bookkeeping.
    if filters.document_ids and chunk.span.document_id not in filters.document_ids:
        return False
    if filters.as_of is not None and not in_force(chunk.meta, filters.as_of):
        return False
    # OR rather than AND across the two table signals. ``meta.is_table`` is the filterable
    # surface and ``kind.is_table`` is what produced the chunk; they are supposed to agree,
    # and a disagreement is an ingest bug. OR means that bug costs precision on a
    # tables-only query instead of hiding a benefit table from the one query type that
    # asked specifically for tables.
    return not filters.tables_only or chunk.meta.is_table or chunk.kind.is_table


def candidate_ids(
    chunks: Mapping[str, Chunk],
    filters: RetrievalFilters | None,
    *,
    retriever: str,
) -> frozenset[str] | None:
    """The allow-set to hand an index, or ``None`` for "no restriction".

    A linear scan of chunk metadata per query. At this corpus size (~10^3 chunks) that is
    microseconds and needs no supporting structure; secondary indexes on insurer, product
    and date are the optimisation to reach for if the corpus grows by orders of magnitude,
    and building them now would add invalidation logic with nothing to show for it.

    ``retriever`` is only used for logging, so a warning about corpus metadata can be
    attributed to the query that surfaced it.
    """
    if not is_active(filters):
        return None
    assert filters is not None  # is_active(None) is False
    undated_supersession = 0
    allowed: set[str] = set()
    for chunk_id, chunk in chunks.items():
        if chunk.meta.superseded_by and chunk.meta.effective_to is None:
            undated_supersession += 1
        if matches(chunk, filters):
            allowed.add(chunk_id)
    if undated_supersession:
        # An ingest-invariant violation (see ``in_force``). Reported rather than acted on,
        # so the count is visible in the trace of a query whose recall it affects.
        log.warning(
            "superseded_chunk_without_effective_to",
            count=undated_supersession,
            retriever=retriever,
        )
    return frozenset(allowed)


def meta_from_source(source: object, **extra: object) -> ChunkMeta:
    """Build a ``ChunkMeta`` carrying the manifest's filterable facts.

    ## The gap this closes

    Every field this module filters on - ``insurer``, ``product``, ``effective_from``,
    ``effective_to``, ``superseded_by`` - was **never written by anything**. Both
    ``ChunkMeta`` construction sites set only ``is_table``, ``section_path`` and
    ``clause_ids``, so the entire filtering layer read fields that no code populated.

    Measured 2026-10-02, 1,024 chunks: every one of those five fields empty. The
    consequence was not a wrong number but an inert feature - retrieving h-01 with and
    without ``as_of=2026-09-04`` returned the identical top 10, three chunks of it from the
    superseded 2021 wording, because ``in_force()`` cannot exclude a chunk it cannot place
    in time.

    This is the mirror image of the pattern in ``docs/SILENT_WRONGNESS.md``: that one is a
    signal PRODUCED and never consumed; this is a signal CONSUMED and never produced. Both
    look correct from either end, and ``tools/audit_signals.py`` could only see the first -
    these fields are read by this module, so the audit scored them healthy.
    """
    return ChunkMeta(
        insurer=getattr(source, "insurer", None),
        product=getattr(source, "product", None),
        effective_from=getattr(source, "effective_from", None),
        effective_to=getattr(source, "effective_to", None),
        superseded_by=getattr(source, "superseded_by", None),
        **extra,  # type: ignore[arg-type]
    )
