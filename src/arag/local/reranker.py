"""Cross-encoder reranker. The stage that reads the query and the passage *together*.

OFFLINE ONLY — loads `transformers`, so it lives in `arag.local` for the reason recorded in
that package's `__init__`. The online counterpart is the same model as quantised ONNX; the
`Reranker` protocol is what lets them swap without the engine noticing.

## Why a cross-encoder can do what retrieval cannot

Bi-encoder retrieval embeds the query and the passage *separately* and compares vectors, so
the passage's representation is fixed before the query is known. BM25 never sees them
together either — it scores term overlap. A cross-encoder runs the pair through one forward
pass, so attention crosses between them, and it can distinguish "this passage is about
waiting periods" from "this passage answers THIS waiting-period question".

That is precisely the gap the v2 measurement exposed. RRF fusion discards score magnitude
and so cannot tell a confident wrong ranking from a confident right one — measured on h-28,
where fusing BM25 pushed a span dense had found at rank 3 out of the top 10 entirely. A
reranker is the principled answer to that, because it re-scores candidates against the
query directly instead of trusting the rank that produced them.

## What it costs, and why the cost is the point

DESIGN §5.5 does not let this stage in for free: *"nDCG@k is the metric that must justify
its 60ms."* The reranker is worth having only if the measured nDCG gain pays for the
measured latency. Both halves are recorded in `docs/V4_RERANK.md`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from arag.obs import get_logger, span

if TYPE_CHECKING:
    from arag.retrieval.types import RetrievedChunk

log = get_logger(__name__)


class CrossEncoderReranker:
    """Re-scores candidates by running each (query, passage) pair through the model."""

    name = "cross-encoder"

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model = model
        self._torch = torch
        # local_files_only: `from_pretrained` contacts the hub even when every weight is
        # cached, and a DNS failure then burns the whole run on retries. Learned from the
        # generation smoke test, where exactly that happened.
        self._tokeniser = AutoTokenizer.from_pretrained(model, local_files_only=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model, local_files_only=True
        ).to(device)
        self._model.eval()
        self._max_length = max_length
        self._batch_size = batch_size
        log.info("reranker_loaded", model=model, device=device)

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], *, top_n: int = 6
    ) -> list[RetrievedChunk]:
        import asyncio

        if not candidates or not query.strip():
            # Not an error. An empty candidate list means retrieval found nothing, and the
            # abstain path owns that decision - a reranker inventing a result here would
            # take it away from the layer that is supposed to make it.
            return []
        return await asyncio.to_thread(self._rerank_sync, query, candidates, top_n)

    def _rerank_sync(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        scores: list[float] = []
        with span("rerank.cross_encoder"), self._torch.no_grad():
            for start in range(0, len(candidates), self._batch_size):
                batch = candidates[start : start + self._batch_size]
                encoded = self._tokeniser(
                    [query] * len(batch),
                    [c.text for c in batch],
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="pt",
                )
                logits = self._model(**encoded).logits
                scores.extend(float(x) for x in logits.view(-1))

        scored = [
            chunk.model_copy(update={"rerank_score": score})
            for chunk, score in zip(candidates, scores, strict=True)
        ]
        # Ties break on chunk_id, matching both indexes and the fuser. A ranking that moves
        # when two candidates tie would make every eval number noisy for reasons unrelated
        # to quality.
        scored.sort(key=lambda c: (-(c.rerank_score or 0.0), c.chunk_id))
        log.debug(
            "reranked",
            n_in=len(candidates),
            n_out=min(top_n, len(scored)),
            top_score=scored[0].rerank_score if scored else None,
        )
        return scored[:top_n]


class RerankingRetriever:
    """A ``Retriever`` that retrieves wide and returns narrow.

    Composed rather than built into the engine, so the quality-ceiling experiment in
    DESIGN §5.5 — swap the reranker, hold retrieval fixed — stays possible. A combined
    interface would make that experiment impossible to run.
    """

    def __init__(
        self,
        inner: object,
        reranker: CrossEncoderReranker,
        *,
        fetch_k: int = 30,
        name: str = "reranked",
    ) -> None:
        self._inner = inner
        self._reranker = reranker
        # Retrieve MORE than the caller asked for, then cut. The whole value of reranking
        # is promoting something the first stage ranked poorly; fetching exactly top_k
        # would make it a re-ordering of a list already judged good, which is the cheapest
        # way to buy latency and no quality.
        self._fetch_k = fetch_k
        self.name = name

    async def retrieve(self, query: str, *, filters: object = None, top_k: int = 30) -> list:  # type: ignore[type-arg]
        candidates = await self._inner.retrieve(  # type: ignore[attr-defined]
            query, filters=filters, top_k=max(self._fetch_k, top_k)
        )
        return await self._reranker.rerank(query, candidates, top_n=top_k)
