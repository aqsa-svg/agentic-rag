"""An engine that retrieves and then deliberately abstains.

This is how the v1 baseline gets measured by the eval harness rather than by a bespoke
script, which matters because the project's standard is that every performance claim
traces to a number the eval suite produced.

It satisfies ``QueryEngine`` while doing no generation at all: it runs retrieval, attaches
the candidates to the ``AnswerResult``, and abstains with
``AbstainReason.GENERATION_UNAVAILABLE``. That is not a placeholder — it is the honest
description of v1. There is no generator wired, no judge whose agreement has been
measured, and therefore nothing that could justify emitting an answer.

The consequence is deliberate and worth understanding before reading the numbers: every
item scores as a **behavioural** failure (it abstained where an answer was expected) while
still producing real **retrieval** metrics. Retrieval quality and answer quality are
separate measurements, and v1 only earns the first.
"""

from __future__ import annotations

import time
from datetime import date

from arag.agent.types import AbstainReason, AnswerResult, StageLatency
from arag.obs import current_trace_id, get_logger, span
from arag.retrieval.protocols import Retriever
from arag.retrieval.types import RetrievalFilters

log = get_logger(__name__)


class RetrievalOnlyEngine:
    """Wraps any ``Retriever`` so the eval harness can score retrieval in isolation."""

    def __init__(self, retriever: Retriever, *, top_k: int = 10, name: str | None = None) -> None:
        self._retriever = retriever
        self._top_k = top_k
        # Named after the retriever so EVAL_LOG.md rows say which retriever produced them.
        self.name = name or retriever.name

    async def answer(self, question: str, *, as_of: date | None = None) -> AnswerResult:
        with span(f"engine.retrieval_only.{self._retriever.name}"):
            started = time.perf_counter()
            # as_of is threaded through rather than dropped: the supersession filter is a
            # retrieval concern, and silently ignoring it here would make a superseded-clause
            # hit look like a retrieval failure instead of a missing filter.
            filters = RetrievalFilters(as_of=as_of) if as_of else None
            chunks = await self._retriever.retrieve(question, filters=filters, top_k=self._top_k)
            elapsed = (time.perf_counter() - started) * 1000

            log.debug(
                "retrieval_only",
                retriever=self._retriever.name,
                n=len(chunks),
                retrieve_ms=round(elapsed, 2),
            )
            return AnswerResult(
                question=question,
                answer=None,
                abstained=True,
                abstain_reason=AbstainReason.GENERATION_UNAVAILABLE,
                confidence=chunks[0].confidence if chunks else 0.0,
                retrieved=tuple(chunks),
                engine=self.name,
                as_of=as_of,
                trace_id=current_trace_id(),
                latency=StageLatency(retrieve_ms=elapsed, total_ms=elapsed),
            )
