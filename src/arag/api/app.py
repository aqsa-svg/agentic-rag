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

# The interactive browser UI, appended to the (f-string) landing head. Kept as a plain string so
# the JavaScript braces are literal. No framework, no build step; it POSTs to the existing /ask.
_Q_GRACE = "What is the grace period for paying premium in instalments?"
_Q_FREELOOK = "What is the free-look period?"
_Q_ROOMRENT = "What is the room rent sub-limit under the Star Comprehensive policy?"

_ASK_UI = (
    """<h2>Ask a question</h2>
<p class="muted">Examples &mdash; click to fill the box:</p>
<p>
  <button class="exbtn" data-q=\""""
    + _Q_GRACE
    + """">Grace period (IRDAI answers this)</button>
  <button class="exbtn" data-q=\""""
    + _Q_FREELOOK
    + """">Free-look period (IRDAI answers this)</button>
  <button class="exbtn" data-q=\""""
    + _Q_ROOMRENT
    + """">Room-rent sub-limit (out of scope &rarr; should abstain)</button>
</p>
<textarea id="q" rows="2"
  placeholder="Ask about the IRDAI Master Circular or its Annexure..."></textarea>
<div><button id="ask" class="primary">Ask</button><span id="status"></span></div>
<div id="out"></div>
<p class="foot"><a href="/health">/health</a> &middot; <a href="/docs">/docs</a>
  &middot; answers are generated; verify against the cited pages.</p>
<script>
const qEl = document.getElementById('q');
const out = document.getElementById('out');
const st = document.getElementById('status');
const btn = document.getElementById('ask');
function esc(s){
  return (s == null ? '' : String(s))
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
document.querySelectorAll('.exbtn').forEach(function(b){
  b.addEventListener('click', function(){ qEl.value = b.getAttribute('data-q'); qEl.focus(); });
});
function render(d){
  if (d.abstained){
    const reason = d.abstain_reason || 'unknown';
    let why = 'The service did not produce a grounded answer.';
    if (reason === 'low_confidence' || reason === 'no_candidates'){
      why = 'The IRDAI documents do not support an answer'
        + ' (for an out-of-scope question, that is the correct outcome).';
    } else if (reason === 'generation_not_configured'){
      why = 'Generation is not configured on this instance.';
    } else if (reason === 'service_degraded' || reason === 'generation_unavailable'){
      why = 'The generation service was unavailable for this request.';
    }
    out.innerHTML = '<div class="ans abstain"><strong>Abstained</strong> &mdash; reason <code>'
      + esc(reason) + '</code>. ' + esc(why) + '</div>';
    return;
  }
  let html = '<div class="ans"><p>' + esc(d.answer) + '</p>';
  if (d.citations && d.citations.length){
    html += '<p class="muted">Citations (document, page, clause &mdash; metadata only):</p><ul>';
    d.citations.forEach(function(c){
      html += '<li><code>' + esc(c.document_id) + '</code> p' + esc(c.page)
        + (c.clause_id ? ' cl.' + esc(c.clause_id) : '') + '</li>';
    });
    html += '</ul>';
  } else {
    html += '<p class="muted">No citations returned.</p>';
  }
  html += '</div>';
  out.innerHTML = html;
}
async function ask(){
  const question = qEl.value.trim();
  if (question.length < 3){
    out.innerHTML = '<p class="muted">Enter a question (at least 3 characters).</p>';
    return;
  }
  btn.disabled = true; st.textContent = ' thinking...'; out.innerHTML = '';
  try {
    const r = await fetch('/ask', {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify({question: question}),
    });
    if (r.status === 429){
      out.innerHTML = '<p class="errmsg">Rate limit reached &mdash; wait a moment and retry.</p>';
      return;
    }
    render(await r.json());
  } catch (e){
    out.innerHTML = '<p class="errmsg">Request failed: ' + esc(e) + '</p>';
  } finally {
    btn.disabled = false; st.textContent = '';
  }
}
btn.addEventListener('click', ask);
qEl.addEventListener('keydown', function(e){
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') ask();
});
</script>
</body></html>"""
)


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
            "textarea{width:100%;font:inherit;padding:.5rem;border:1px solid #d4d4d8;"
            "border-radius:6px;box-sizing:border-box}"
            "button{font:inherit;cursor:pointer;border:1px solid #d4d4d8;background:#fff;"
            "padding:.4rem .7rem;border-radius:6px}"
            "button:disabled{opacity:.5;cursor:default}"
            ".primary{background:#111;color:#fff;border-color:#111;margin-top:.5rem}"
            ".exbtn{margin:.15rem .25rem .15rem 0;font-size:14px}"
            ".ans{background:#f0fdf4;border:1px solid #bbf7d0;padding:1rem;border-radius:8px;"
            "margin-top:1rem}"
            ".ans.abstain{background:#f4f4f5;border-color:#d4d4d8}"
            ".muted{color:#71717a;font-size:14px;margin-bottom:.25rem}"
            ".errmsg{color:#b91c1c}"
            ".foot{color:#71717a;font-size:14px;margin-top:2rem}"
            "#status{color:#71717a;font-size:14px;margin-left:.5rem}"
        )
        return (
            f"""<!doctype html><html><head><meta charset="utf-8">
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
<div class="warn"><strong>Measured limitation — read before trusting an answer.</strong>
<strong>Every measured number in this project comes from a local 1.5B instrument model, not from
the model serving this page.</strong> On that labelled 30-item eval, abstention was unreliable —
three independent signals each failed to tell answerable from unanswerable, and the system
correctly refused only <strong>2 of 5</strong> questions the documents do not address. This
instance generates with a hosted production model (Gemini), whose behaviour here is
<strong>unmeasured</strong> beyond two probe questions (one answered, grounded; one correctly
abstained). So it can still answer, with apparent confidence, a question it should decline. Treat
every answer as a <em>pipeline demonstration, not advice</em>, and verify it against the cited
pages. The measurements are in the repo (<code>docs/AGENT_EVAL.md</code>) — documented, not
hidden.</div>
"""
            + _ASK_UI
        )

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
