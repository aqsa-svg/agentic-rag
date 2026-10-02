"""Retrieval interfaces.

``Protocol`` rather than an ABC because nothing here needs shared implementation and
structural typing keeps the eval harness free of any import from a concrete retriever.
The harness depends on this module; the concrete dense/lexical/hybrid retrievers depend on
it too. Neither depends on the other, which is what lets v1's naive retriever and v2's
hybrid one be measured by the same runner without the runner changing.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from arag.retrieval.types import RetrievalFilters, RetrievedChunk


@runtime_checkable
class Retriever(Protocol):
    """Returns candidates ordered best-first. Implementations must not raise on an empty
    corpus; they return an empty list so the abstain path is exercised rather than a 500.
    """

    name: str

    async def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
        top_k: int = 30,
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class Reranker(Protocol):
    """Re-scores candidates against the query. Separate from ``Retriever`` because the
    quality ceiling experiment in DESIGN §5.5 swaps the reranker alone while holding
    retrieval fixed; a combined interface would make that experiment impossible.
    """

    name: str

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], *, top_n: int = 6
    ) -> list[RetrievedChunk]: ...
