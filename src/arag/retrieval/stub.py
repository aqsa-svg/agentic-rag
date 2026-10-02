"""Stub retrievers used to establish the metric floor and to test error paths.

``NullRetriever`` is not a placeholder to be deleted. It is the instrument that proves the
harness can register total failure: run the suite against it and every retrieval metric
must read exactly 0.0. If any metric comes back non-zero against a retriever that returns
nothing, the metric is broken — and finding that out now is far cheaper than discovering
it after a real pipeline has been "improving" the number for three slices.

``ExplodingRetriever`` is the Phase 4 fault injector, defined here so the failure-path
tests in F1/F2 do not need to monkeypatch a real client.
"""

from __future__ import annotations

from arag.obs import get_logger, span
from arag.retrieval.types import RetrievalFilters, RetrievedChunk

log = get_logger(__name__)


class NullRetriever:
    """Always returns nothing. The metric floor."""

    name = "null"

    async def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
        top_k: int = 30,
    ) -> list[RetrievedChunk]:
        with span("retrieval.null"):
            log.debug("null_retriever_called", query_chars=len(query), top_k=top_k)
            return []


class FixedRetriever:
    """Returns a canned list regardless of query. Used by metric unit tests so that
    expected values can be computed by hand rather than asserted against whatever the
    implementation happens to produce.
    """

    name = "fixed"

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    async def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
        top_k: int = 30,
    ) -> list[RetrievedChunk]:
        return self._chunks[:top_k]


class ExplodingRetriever:
    """Raises on every call. Drives failure modes F1 and F2 in Phase 4."""

    name = "exploding"

    def __init__(self, exc: BaseException | None = None) -> None:
        self._exc = exc or ConnectionError("simulated pgvector outage")

    async def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
        top_k: int = 30,
    ) -> list[RetrievedChunk]:
        raise self._exc
