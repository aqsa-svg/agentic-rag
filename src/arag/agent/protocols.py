"""The engine interface — the seam that keeps LangGraph replaceable.

Everything upstream (the FastAPI route, the eval runner, the CLI) talks to
``QueryEngine``. Only ``arag.agent.graph`` will import LangGraph, and
``tests/test_import_boundaries.py`` fails the build if that stops being true.

This is what makes DESIGN §5.7's claim honest: "framework choice is reversible" is a
testable property here, not a hopeful sentence in a README.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

from arag.agent.types import AnswerResult


@runtime_checkable
class QueryEngine(Protocol):
    """Answers one question.

    Implementations **must not raise** for expected failure modes. A rate limit, a dead
    database or an unparseable model response is expressed as an ``AnswerResult`` with
    ``abstained`` and ``degraded`` set, because the harness needs to score the degraded
    behaviour. Only genuinely unexpected programming errors should propagate.
    """

    name: str

    async def answer(self, question: str, *, as_of: date | None = None) -> AnswerResult: ...
