"""Dense retrieval over a vector index. Online-safe: holds a model, never imports one.

The embedder arrives by injection and is used only through the ``Embedder`` protocol, so
this module imports neither torch nor sentence-transformers and stays inside the online
bundle boundary (DESIGN §9, enforced by ``tests/test_import_closure.py``). Offline the
injected embedder is ``SentenceTransformerEmbedder``; online it is the ONNX one. Neither is
named here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from arag.obs import get_logger, span
from arag.retrieval.filtering import candidate_ids
from arag.retrieval.types import RetrievalFilters, RetrievedChunk

if TYPE_CHECKING:
    from collections.abc import Mapping

    from arag.index.dense import DenseIndex
    from arag.ingest.chunk_types import Chunk
    from arag.retrieval.embedding import Embedder

log = get_logger(__name__)


class DenseRetriever:
    """Cosine retrieval over ``DenseIndex``, with the index/query model contract enforced."""

    name = "dense"

    def __init__(self, index: DenseIndex, chunks: Mapping[str, Chunk], embedder: Embedder) -> None:
        """Three checks at construction, all of which prevent a silent failure.

        1. **Model identity.** An index built with one model and queried with another
           returns a complete, confidently-ordered ranking of nonsense: the vectors are the
           right shape, the scores are valid floats, nothing raises. There is no runtime
           symptom at all, and the only way to notice is that recall is worse than it
           should be - which is indistinguishable from the retriever simply being bad.
        2. **Query prefix.** Same class of failure, smaller blast radius. BGE embeds
           queries with an instruction prefix and passages without; supply an embedder
           configured differently from the one that built the index and every query lands
           in a slightly wrong region of the space. Recall drops, nothing errors.
        3. **Chunk coverage.** The index scores ids the store must be able to render. A
           mismatch means citing text that was never indexed, or a short result list that
           reads as "nothing matched".

        All three are loud here rather than silent later, which is the trade this project
        makes everywhere: a crash at construction costs a minute, a silent ranking defect
        costs every number measured afterwards.
        """
        if embedder.model_id != index.model_id:
            raise ValueError(
                f"index was built with {index.model_id!r} but the query embedder is "
                f"{embedder.model_id!r}. Cosine between two different embedding spaces is "
                "a valid float and a meaningless ranking - it would not raise anywhere"
            )
        if embedder.dim != index.dim:
            raise ValueError(f"embedder is {embedder.dim}-d, index is {index.dim}-d")
        index_prefix = index.query_prefix
        embedder_prefix = getattr(embedder, "query_prefix", "")
        if index_prefix != embedder_prefix:
            raise ValueError(
                f"query prefix disagrees: index recorded {index_prefix!r}, embedder uses "
                f"{embedder_prefix!r}. The prefix is part of the model contract and a "
                "mismatch costs recall with no error anywhere"
            )
        missing = sorted(set(index.chunk_ids) - set(chunks))
        if missing:
            raise ValueError(
                f"chunk store is missing {len(missing)} of {index.n_chunks} indexed chunks "
                f"(e.g. {missing[:3]}); the index and the store were built from different "
                "ingest runs"
            )
        self._index = index
        self._chunks = chunks
        self._embedder = embedder

    async def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
        top_k: int = 30,
    ) -> list[RetrievedChunk]:
        with span("retrieval.dense"):
            if not query.strip():
                log.debug("empty_query", retriever=self.name)
                return []

            allow = candidate_ids(self._chunks, filters, retriever=self.name)
            if allow is not None and not allow:
                log.info("filters_excluded_entire_corpus", n_chunks=len(self._chunks))
                return []

            vector = self._embedder.embed_query(query)
            hits = self._index.search(vector, top_k, allow=allow)
            results = [
                RetrievedChunk(
                    chunk_id=chunk_id,
                    span=self._chunks[chunk_id].span,
                    text=self._chunks[chunk_id].text,
                    meta=self._chunks[chunk_id].meta,
                    dense_score=score,
                )
                for chunk_id, score in hits
            ]
            log.debug(
                "dense_retrieved",
                n_results=len(results),
                top_k=top_k,
                candidates=len(allow) if allow is not None else self._index.n_chunks,
                top_score=results[0].dense_score if results else None,
            )
            return results
