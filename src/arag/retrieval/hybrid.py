"""Reciprocal Rank Fusion over two or more retrievers.

## Why RRF rather than weighted score fusion

BM25 scores and cosine similarities are not on comparable scales and do not become
comparable by normalising them. BM25 is unbounded above and its range depends on corpus
statistics — idf, average document length — so min-max normalisation over a result list
makes the same chunk's normalised score depend on which *other* chunks came back. Cosine
is bounded in [-1, 1] but its useful range for a given model is narrow and nonlinear.

RRF sidesteps the problem by discarding magnitude entirely and fusing **ranks**:

    score(chunk) = sum over retrievers of  weight_r / (k + rank_r(chunk))

Rank is the one thing both retrievers agree on the meaning of. The cost is real and worth
stating: RRF cannot tell a runaway top hit from a marginal one, because rank 1 contributes
``1/(k+1)`` whether the chunk scored 0.99 or 0.31. That information loss is what the
reranker is for.

## k, and why it is not 60 because a paper said so

``k`` controls how quickly a retriever's contribution decays with rank. Small ``k`` makes
the fusion behave like "whoever ranked it first wins"; large ``k`` flattens every retriever
towards an unweighted vote. The literature default is 60, chosen on TREC-scale runs with
thousands of candidates. This corpus returns tens.

So 60 is the *starting* value here, not the answer, and it is exposed as a parameter with
its sensitivity measured by the eval harness rather than assumed. DESIGN §5.4 requires the
same of the dense/lexical weights: tuned against the suite, not guessed. Any claim about
either belongs in EVAL_LOG.md with the numbers that produced it.

## Determinism

Ties break on ``chunk_id``, matching both underlying indexes. A fusion whose output moves
when two chunks tie would make every downstream metric noisy for reasons unrelated to
retrieval quality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from arag.obs import get_logger, span
from arag.retrieval.types import RetrievalFilters, RetrievedChunk

if TYPE_CHECKING:
    from arag.retrieval.protocols import Retriever

log = get_logger(__name__)

DEFAULT_RRF_K = 60


class RRFHybridRetriever:
    """Fuses several retrievers by reciprocal rank."""

    def __init__(
        self,
        retrievers: list[Retriever],
        *,
        weights: list[float] | None = None,
        k: int = DEFAULT_RRF_K,
        candidate_multiplier: int = 3,
        name: str = "hybrid-rrf",
    ) -> None:
        if not retrievers:
            raise ValueError("RRFHybridRetriever needs at least one retriever")
        if weights is not None and len(weights) != len(retrievers):
            raise ValueError(f"{len(weights)} weights for {len(retrievers)} retrievers")
        if k <= 0:
            raise ValueError(f"rrf k must be positive, got {k}")
        self._retrievers = retrievers
        self._weights = weights or [1.0] * len(retrievers)
        self._k = k
        # Each retriever is asked for more than top_k, because a chunk ranked 12th by one
        # and 2nd by the other should be able to reach the fused top 10 - that recovery is
        # the entire point of fusing. Asking each for exactly top_k would make fusion a
        # re-ordering of the intersection and silently discard the complementary recall
        # the dense retriever exists to provide.
        self._candidate_multiplier = max(1, candidate_multiplier)
        self.name = name

    async def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
        top_k: int = 30,
    ) -> list[RetrievedChunk]:
        with span("retrieval.hybrid_rrf"):
            if not query.strip():
                return []

            per_retriever = top_k * self._candidate_multiplier
            fused: dict[str, float] = {}
            best: dict[str, RetrievedChunk] = {}
            contributions: dict[str, int] = {}

            for retriever, weight in zip(self._retrievers, self._weights, strict=True):
                hits = await retriever.retrieve(query, filters=filters, top_k=per_retriever)
                contributions[retriever.name] = len(hits)
                for rank, chunk in enumerate(hits, start=1):
                    fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + weight / (
                        self._k + rank
                    )
                    # Merge the per-stage scores rather than keeping whichever retriever
                    # ran last. RetrievedChunk carries dense_score and lexical_score
                    # separately so a failure can be attributed to the stage that caused
                    # it; collapsing them here would destroy exactly that.
                    existing = best.get(chunk.chunk_id)
                    if existing is None:
                        best[chunk.chunk_id] = chunk
                    else:
                        best[chunk.chunk_id] = existing.model_copy(
                            update={
                                "dense_score": existing.dense_score
                                if existing.dense_score is not None
                                else chunk.dense_score,
                                "lexical_score": existing.lexical_score
                                if existing.lexical_score is not None
                                else chunk.lexical_score,
                            }
                        )

            ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
            results = [
                best[chunk_id].model_copy(update={"fused_score": score})
                for chunk_id, score in ordered
            ]
            log.debug(
                "hybrid_retrieved",
                n_results=len(results),
                top_k=top_k,
                rrf_k=self._k,
                per_retriever=contributions,
                union_size=len(fused),
            )
            return results
