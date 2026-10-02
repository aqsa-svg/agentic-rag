"""Stub engines. The v0 metric floor and the fixtures for behavioural tests.

``NullEngine`` abstains on everything. Running the suite against it produces a genuinely
instructive floor rather than a row of zeroes:

* every retrieval metric reads 0.0 (nothing was retrieved),
* ``abstain_recall`` reads 1.0 (it correctly refused every unanswerable item),
* ``false_abstention_rate`` reads 1.0 (it also refused every answerable item).

That last pair is the point. A single "abstain accuracy" number would let a system that
refuses everything look calibrated. Splitting over-refusal from under-refusal makes the
degenerate strategy visibly useless, which is exactly the property the F7 threshold needs
from its metric before anyone tunes it.
"""

from __future__ import annotations

from datetime import date

from arag.agent.types import AbstainReason, AnswerResult, StageLatency
from arag.obs import current_trace_id, get_logger, span

log = get_logger(__name__)


class NullEngine:
    """Abstains on every question."""

    name = "null"

    async def answer(self, question: str, *, as_of: date | None = None) -> AnswerResult:
        with span("engine.null"):
            log.debug("null_engine_abstain", question_chars=len(question))
            return AnswerResult(
                question=question,
                abstained=True,
                abstain_reason=AbstainReason.NOT_IMPLEMENTED,
                confidence=0.0,
                engine=self.name,
                as_of=as_of,
                trace_id=current_trace_id(),
                latency=StageLatency(total_ms=0.0),
            )


class EchoEngine:
    """Answers by echoing the question with no citations.

    Deliberately bad in a *different* way from ``NullEngine``: it never abstains and never
    cites. Used to prove the harness penalises confident unsupported answers — faithfulness
    and citation validity must both fail here. Two stubs failing in opposite directions
    bracket the metric space from below.
    """

    name = "echo"

    async def answer(self, question: str, *, as_of: date | None = None) -> AnswerResult:
        with span("engine.echo"):
            return AnswerResult(
                question=question,
                answer=f"You asked: {question}",
                citations=(),
                abstained=False,
                confidence=1.0,
                engine=self.name,
                as_of=as_of,
                trace_id=current_trace_id(),
                latency=StageLatency(total_ms=0.0),
            )
