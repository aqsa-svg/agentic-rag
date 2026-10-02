"""FastAPI service. Route handlers contain no logic (DESIGN §10).

Day-1 purpose is to prove the deploy path while there is time to fix it, so this ships
with ``NullEngine`` behind it. That is deliberate and more useful than a hello-world:
``/ask`` already exercises the real ``QueryEngine`` contract and returns a real
``AnswerResult``, so when the actual engine is swapped in at v1 nothing about the service
layer changes. The engine is selected by env var, and the response says which one served
it, so a deployed build can never be mistaken for a working system.
"""

from __future__ import annotations

import os
import time
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from arag.agent.protocols import QueryEngine
from arag.agent.stub import EchoEngine, NullEngine
from arag.agent.types import AnswerResult
from arag.ingest.normalise import normalise_query
from arag.obs import configure_logging, get_logger, new_trace_id, trace

log = get_logger(__name__)

ENGINES: dict[str, type[QueryEngine]] = {"null": NullEngine, "echo": EchoEngine}
STARTED_AT = time.time()


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class AskResponse(BaseModel):
    """Deliberately mirrors AnswerResult rather than flattening it.

    ``abstained``, ``abstain_reason`` and ``degraded`` are part of the contract, not error
    handling: a caller must be able to tell "no answer because retrieval was weak" from
    "no answer because the generator was rate limited".
    """

    question: str
    answer: str | None
    citations: list[dict[str, Any]]
    abstained: bool
    abstain_reason: str | None
    confidence: float
    degraded: list[str]
    engine: str
    trace_id: str | None
    latency_ms: float
    warning: str | None = None


def _engine() -> QueryEngine:
    key = os.environ.get("ARAG_ENGINE", "null")
    return ENGINES.get(key, NullEngine)()


def create_app() -> FastAPI:
    configure_logging(os.environ.get("ARAG_LOG_LEVEL", "INFO"))
    app = FastAPI(
        title="Agentic RAG — Indian health insurance policy wordings",
        version="0.1.0",
        description=(
            "Day-1 deploy proving the path. The engine is a stub: /ask abstains on every "
            "question by design. See /status for what is and is not wired."
        ),
    )
    engine = _engine()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "uptime_s": round(time.time() - STARTED_AT, 1)}

    @app.get("/status")
    def status() -> dict[str, Any]:
        """What is actually wired. Kept honest on purpose so a live URL cannot imply a
        working system.
        """
        return {
            "engine": engine.name,
            "engine_is_stub": engine.name in ENGINES,
            "wired": ["normalisation", "engine contract", "structured logging"],
            "not_wired": [
                "corpus index",
                "hybrid retrieval",
                "reranker",
                "LangGraph agent",
                "guardrails",
            ],
            "eval_gate": "no labelled data yet",
        }

    @app.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest) -> AskResponse:
        with trace(new_trace_id()) as trace_id:
            started = time.perf_counter()
            # Query-side normalisation (N1): a user pasting a phrase out of the PDF sends
            # ligatures too, and index-side-only normalisation leaves that query broken.
            question = normalise_query(req.question)
            result: AnswerResult = await engine.answer(question)
            elapsed = (time.perf_counter() - started) * 1000

            log.info(
                "ask",
                engine=result.engine,
                abstained=result.abstained,
                latency_ms=round(elapsed, 1),
            )
            return AskResponse(
                question=question,
                answer=result.answer,
                citations=[
                    {
                        "document_id": c.span.document_id,
                        "page": c.span.page,
                        "clause_id": c.span.clause_id,
                    }
                    for c in result.citations
                ],
                abstained=result.abstained,
                abstain_reason=result.abstain_reason.value if result.abstain_reason else None,
                confidence=result.confidence,
                degraded=[d.value for d in result.degraded],
                engine=result.engine,
                trace_id=trace_id,
                latency_ms=round(elapsed, 2),
                warning=(
                    "Stub engine: this endpoint abstains on every question by design. See /status."
                ),
            )

    return app


app = create_app()
