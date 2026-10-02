"""BM25 as a first-class retriever, not merely as one arm of a hybrid.

## Why lexical retrieval is a separate, independently measurable component

Hybrid retrieval is the plan, and a hybrid that cannot be taken apart cannot be debugged.
When an answer misses the clause a human labelled, exactly one of these is true: the
lexical arm never surfaced it, the dense arm never surfaced it, fusion buried it, or the
reranker dropped it. Those are four different fixes. Wiring BM25 straight into a fusion
function would collapse the first two into "retrieval was bad" and leave the eval harness
with nothing to attribute the failure to - which is why this class satisfies the same
``Retriever`` protocol as the dense retriever and can be run against the golden set alone.

There is a second reason, specific to this corpus. Policy wordings answer many questions
by exact string: a clause number ("4.2"), a defined term in title case, a named procedure.
Embeddings blur exactly the tokens those questions turn on, so the lexical arm is not a
legacy baseline here - it is the arm that is expected to win on clause-lookup queries, and
"expected to win" is a claim that needs a number of its own.

## What this module prevents: filtering after ranking

Every ``RetrievalFilters`` field is evaluable from chunk metadata, so all of them are
applied *before* scoring, and the filtered id set is handed to ``Bm25Index.search`` as its
candidate set. The alternative - score the corpus, take the top k, then drop the chunks
that fail the filter - is the bug this module is shaped to avoid: an ``as_of`` or
document-scoped query would silently return fewer results than asked for, or none at all,
because the global top k came from documents the filter excludes. A supersession filter
that quietly returns nothing is worse than no supersession filter, because it looks like
"no such clause" rather than "the filter ate it".

Injection-suspected chunks are deliberately *not* dropped here. ``ChunkMeta`` carries the
flag, but quarantining is a guardrail policy decision that has to be logged and counted
where the answer is composed; a retriever silently hiding rows would make that count
unobtainable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from arag.ingest.chunk_types import Chunk
from arag.obs import get_logger, span
from arag.retrieval.filtering import candidate_ids, in_force, is_active, matches
from arag.retrieval.types import RetrievalFilters, RetrievedChunk

if TYPE_CHECKING:
    # Annotation-only, and deliberately not a runtime import. ``arag.index.__init__``
    # eagerly imports ``arag.index.build``, which eagerly imports this module, so a
    # runtime ``from arag.index.lexical import Bm25Index`` here closes a cycle:
    # retrieval.lexical -> arag.index -> index.build -> retrieval.lexical (still
    # half-initialised). The observed symptom is order-dependent and silent - importing
    # retrieval.lexical first makes index.build record the retriever as missing - which is
    # exactly the kind of failure that shows up as "retrieval returned nothing" in
    # production. The dependency is genuinely type-only: this class is handed a finalised
    # index and never constructs one.
    from arag.index.lexical import Bm25Index

log = get_logger(__name__)


# Filtering moved to arag.retrieval.filtering when the dense retriever arrived. Two
# retrievers with independently-written filter semantics would make RRF fuse two rankings
# taken over different corpora - and the field that would break first is `as_of`, the
# supersession control, whose failure is already the top finding in the v1 baseline.
# Re-exported here because the names are part of this module's tested surface.
_is_active = is_active
_in_force = in_force
_matches = matches


class LexicalRetriever:
    """BM25 retrieval over an in-memory chunk store."""

    name = "bm25"

    def __init__(self, index: Bm25Index, chunks: dict[str, Chunk]) -> None:
        """The chunk store must cover the index.

        Checked at construction because the alternative is a retrieval hole that looks
        like a relevance problem: an index built from one ingest run and a chunk store from
        another would score a chunk, fail to find its text, and either raise mid-request or
        quietly return a shorter list. One linear check at startup converts that into a
        loud failure at the moment the mismatch is created.
        """
        missing = sorted(index.chunk_ids - chunks.keys())
        if missing:
            raise ValueError(
                f"chunk store is missing {len(missing)} of {len(index.chunk_ids)} indexed "
                f"chunks (e.g. {missing[:3]}); the index and the store were built from "
                "different ingest runs, and scoring against one while citing the other "
                "would produce citations for text that was never indexed"
            )
        self._index = index
        self._chunks = chunks
        unindexed = len(chunks.keys() - index.chunk_ids)
        if unindexed:
            log.debug("chunks_present_but_not_indexed", count=unindexed)

    def _candidates(self, filters: RetrievalFilters | None) -> frozenset[str] | None:
        """The allow-set handed to the index, or ``None`` for "no restriction"."""
        return candidate_ids(self._chunks, filters, retriever=self.name)

    async def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
        top_k: int = 30,
    ) -> list[RetrievedChunk]:
        """Return up to ``top_k`` candidates, best-first, each with ``lexical_score`` set.

        ``async`` with nothing awaited: the protocol is async because the dense retriever
        talks to Postgres over the network, and a sync variant of the interface would fork
        the eval harness in two. The cost of an unawaited coroutine is a few microseconds;
        the cost of two retriever interfaces is every caller having to know which it holds.
        """
        with span("retrieval.bm25"):
            if not query.strip():
                # Not an error. An empty query means the abstain path, and the protocol
                # requires retrievers to exercise it rather than raise into a 500.
                log.debug("empty_query", retriever=self.name)
                return []

            allow = self._candidates(filters)
            if allow is not None and not allow:
                log.info("filters_excluded_entire_corpus", n_chunks=len(self._chunks))
                return []

            hits = self._index.search(query, top_k, allow=allow)
            results = [
                RetrievedChunk(
                    chunk_id=chunk_id,
                    span=self._chunks[chunk_id].span,
                    text=self._chunks[chunk_id].text,
                    meta=self._chunks[chunk_id].meta,
                    lexical_score=score,
                )
                for chunk_id, score in hits
            ]
            log.debug(
                "bm25_retrieved",
                n_results=len(results),
                top_k=top_k,
                candidates=len(allow) if allow is not None else len(self._chunks),
                top_score=results[0].lexical_score if results else None,
            )
            return results
