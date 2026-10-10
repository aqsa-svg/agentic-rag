"""FastAPI service. Route handlers are a translation layer over ``AnswerResult`` (DESIGN §10).

## Public scope: IRDAI regulatory documents only

The public instance serves answers over the **two public IRDAI sources** — the 2024 Master
Circular on Health Insurance Business and its Annexure. **The four insurer policy wordings,
the full six-source corpus, and every measured result stay local**, because the wordings are
copyrighted and the project never ships them (see the landing page and DESIGN §10.1). So a
live URL demonstrates the *pipeline* over public regulation, not the measured system.

## Engine selection

``ARAG_ENGINE`` picks the engine. ``null``/``echo`` are stubs (deploy-path proof). ``agent``
builds the real ``GeneratingEngine`` over an IRDAI-only BM25 index and a hosted Gemini
provider: with no ``ARAG_GEMINI_API_KEY`` it abstains ``generation_not_configured`` honestly;
with a key it answers. The response always says which engine served it, so a live URL can
never be mistaken for the full measured system.

## Discipline enforced here (not just documented)

* No endpoint returns raw chunks or full documents — ``/ask`` returns an answer plus citation
  *metadata* (document, page, clause), never passage text, and any model-supplied quote is
  capped at ``MAX_QUOTE_CHARS``.
* A per-client sliding-window rate limit, in-memory (no datastore — see DESIGN §10.1 on why a
  demo audit log in Postgres buys nothing; the trace lives in the structured stdout logs).
"""

from __future__ import annotations

import os
import time
from collections import deque
from datetime import date
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from arag.agent.protocols import QueryEngine
from arag.agent.stub import EchoEngine, NullEngine
from arag.agent.types import AnswerResult
from arag.ingest.normalise import normalise_query
from arag.obs import configure_logging, get_logger, new_trace_id, trace

log = get_logger(__name__)

STUB_ENGINES: dict[str, type[QueryEngine]] = {"null": NullEngine, "echo": EchoEngine}
STARTED_AT = time.time()

MAX_QUOTE_CHARS = 300
RATE_LIMIT_REQUESTS = int(os.environ.get("ARAG_RATE_LIMIT", "20"))
RATE_LIMIT_WINDOW_S = int(os.environ.get("ARAG_RATE_WINDOW_S", "60"))


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    # Supersession control. The agent filters out wordings superseded as of this date; for the
    # IRDAI-only public corpus it rarely changes the result, but the knob is part of the contract.
    as_of: date | None = None


class Citation(BaseModel):
    document_id: str
    page: int | None
    clause_id: str | None
    quote: str | None = None  # capped at MAX_QUOTE_CHARS; usually absent


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    marginal_usd: float
    shadow_usd: float


class Latency(BaseModel):
    retrieve_ms: float | None
    generate_ms: float | None
    total_ms: float


class AskResponse(BaseModel):
    """Mirrors ``AnswerResult`` rather than flattening it: ``abstained``, ``abstain_reason``,
    ``surfaced_conflict`` and ``degraded`` are the contract, not error handling. A caller must
    be able to tell "no answer because retrieval was weak" from "no answer because the
    generator was not configured".
    """

    question: str
    as_of: date | None
    answer: str | None
    citations: list[Citation]
    abstained: bool
    abstain_reason: str | None
    surfaced_conflict: bool
    confidence: float
    degraded: list[str]
    usage: Usage
    latency: Latency
    engine: str
    trace_id: str | None


def _agent_engine() -> QueryEngine:
    """Build the real engine over the IRDAI-only BM25 index. Imports are LOCAL so the module
    scope stays free of offline deps and the import-boundary test stays green."""
    from pathlib import Path

    from arag.agent.generate import GeneratingEngine
    from arag.agent.providers import GeminiProvider
    from arag.index.build import build_corpus
    from arag.retrieval.lexical import LexicalRetriever

    deploy_dir = Path(os.environ.get("ARAG_DEPLOY_DIR", "data/deploy"))
    raw_dir = Path(os.environ.get("ARAG_RAW_DIR", "data/raw"))
    corpus = build_corpus(deploy_dir / "irdai_sources.jsonl", raw_dir, serialisation="row_nl")
    log.info("agent_corpus_built", chunks=len(corpus.chunks), sources="irdai-only")
    retriever = LexicalRetriever(corpus.index, corpus.chunks)
    provider = GeminiProvider()
    log.info("agent_provider", configured=provider.configured, model=provider.model)
    return GeneratingEngine(
        retriever, provider, top_k=int(os.environ.get("ARAG_TOP_K", "5")), name="agent-irdai"
    )


def _engine() -> QueryEngine:
    key = os.environ.get("ARAG_ENGINE", "null")
    if key == "agent":
        return _agent_engine()
    return STUB_ENGINES.get(key, NullEngine)()


def _cap(text: str | None) -> str | None:
    if text is None:
        return None
    return text if len(text) <= MAX_QUOTE_CHARS else text[:MAX_QUOTE_CHARS] + "…"


def create_app() -> FastAPI:
    configure_logging(os.environ.get("ARAG_LOG_LEVEL", "INFO"))
    app = FastAPI(
        title="Agentic RAG — IRDAI health-insurance regulation",
        version="1.0.0",
        description=(
            "Retrieval-augmented QA over the two public IRDAI 2024 health-insurance documents "
            "(Master Circular + Annexure). The four insurer policy wordings and all measured "
            "results are local and not served here. See / for scope."
        ),
    )
    engine = _engine()
    hits: dict[str, deque[float]] = {}

    def _rate_limit(request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        now = time.time()
        q = hits.setdefault(client, deque())
        while q and q[0] < now - RATE_LIMIT_WINDOW_S:
            q.popleft()
        if len(q) >= RATE_LIMIT_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail=f"rate limit: {RATE_LIMIT_REQUESTS} requests / {RATE_LIMIT_WINDOW_S}s",
            )
        q.append(now)

    @app.get("/", response_class=HTMLResponse)
    def landing() -> str:
        configured = os.environ.get("ARAG_GEMINI_API_KEY") is not None
        gen = (
            "configured"
            if configured
            else (
                "<strong>not configured</strong> — the service abstains with reason "
                "<code>generation_not_configured</code> until a key is set"
            )
        )
        css = (
            "body{font:16px/1.6 system-ui,sans-serif;max-width:46rem;margin:3rem auto;"
            "padding:0 1rem;color:#111}"
            "code{background:#f4f4f5;padding:.1rem .3rem;border-radius:4px}"
            ".scope{background:#fff7ed;border:1px solid #fed7aa;padding:1rem;border-radius:8px}"
            ".warn{background:#fef2f2;border:1px solid #fecaca;padding:1rem;border-radius:8px}"
        )
        return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Agentic RAG — IRDAI regulation</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{css}</style></head>
<body>
<h1>Agentic RAG — Indian health-insurance regulation</h1>
<p>Retrieval-augmented QA over the <strong>two public IRDAI 2024 documents</strong>: the Master
Circular on Health Insurance Business and its Annexure.</p>
<div class="scope"><strong>Scope, stated plainly.</strong> This public instance serves answers
over <em>public regulatory documents only</em>. The four insurer policy wordings, the full
six-source corpus, and every measured result (retrieval, abstention, grounding) are
<strong>local and not served here</strong> — the wordings are copyrighted and the project does
not ship them. A live answer here demonstrates the pipeline over public regulation, not the
measured system.</p>
<p>Engine: <code>{engine.name}</code>. Generation {gen}.</p>
<div class="warn"><strong>Measured limitation — read before trusting an answer.</strong> This runs
a small instrument model and <strong>its abstention is unreliable</strong>: across three
independent signals measured on the labelled set, none reliably told answerable from
unanswerable, and on the 30-item eval the system correctly refused only <strong>2 of 5</strong>
questions the documents do not address — it can answer, with apparent confidence, a question it
should decline. Treat every answer as a <em>pipeline demonstration, not advice</em>, and verify
it against the cited pages. The measurement behind this line is in the repo
(<code>docs/AGENT_EVAL.md</code>), which is the point: the failure is documented, not hidden.</div>
<h2>Ask</h2>
<pre>curl -s -X POST "$URL/ask" -H 'content-type: application/json' \\
  -d '{{"question":"What is the grace period for paying premium in instalments?"}}'</pre>
<p><a href="/health">/health</a> · <a href="/docs">/docs</a></p>
</body></html>"""

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "uptime_s": round(time.time() - STARTED_AT, 1),
            "engine": engine.name,
        }

    @app.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest, request: Request) -> AskResponse:
        _rate_limit(request)
        with trace(new_trace_id()) as trace_id:
            started = time.perf_counter()
            question = normalise_query(req.question)
            result: AnswerResult = await engine.answer(question, as_of=req.as_of)
            elapsed = (time.perf_counter() - started) * 1000
            log.info(
                "ask",
                engine=result.engine,
                abstained=result.abstained,
                abstain_reason=result.abstain_reason.value if result.abstain_reason else None,
                as_of=req.as_of.isoformat() if req.as_of else None,
                latency_ms=round(elapsed, 1),
            )
            return AskResponse(
                question=question,
                as_of=req.as_of,
                answer=result.answer,
                citations=[
                    Citation(
                        document_id=c.span.document_id,
                        page=c.span.page,
                        clause_id=c.span.clause_id,
                        quote=_cap(c.quote),
                    )
                    for c in result.citations
                ],
                abstained=result.abstained,
                abstain_reason=result.abstain_reason.value if result.abstain_reason else None,
                surfaced_conflict=result.surfaced_conflict,
                confidence=result.confidence,
                degraded=[d.value for d in result.degraded],
                usage=Usage(
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    marginal_usd=result.marginal_usd,
                    shadow_usd=result.shadow_usd,
                ),
                latency=Latency(
                    retrieve_ms=result.latency.retrieve_ms,
                    generate_ms=result.latency.generate_ms,
                    total_ms=round(elapsed, 2),
                ),
                engine=result.engine,
                trace_id=trace_id,
            )

    return app


app = create_app()
