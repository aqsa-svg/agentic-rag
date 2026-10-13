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

## The root is a real browser UI

``GET /`` is a single HTML page (vanilla JS, no framework, no build step) whose hero is the
question box; answers render in place against ``/ask``. The scope note and the measured-
limitation caveat live in a collapsible footer, verbatim, out of the main flow.

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

_Q_GRACE = "What is the grace period for paying premium in instalments?"
_Q_FREELOOK = "What is the free-look period?"
_Q_ROOMRENT = "What is the room rent sub-limit under the Star Comprehensive policy?"

# Austere look for a regulatory tool: one sans family + monospace for citations, a single
# restrained accent, generous whitespace, visible focus. No framework, no build step. CSS and JS
# are plain strings (braces stay literal); the HTML skeleton lives in landing(), which is where
# the only dynamic bits (engine name, generation status) are interpolated.
_PAGE_CSS = (
    ":root{--bg:#ffffff;--fg:#17191c;--muted:#6b7280;--line:#e5e7eb;--panel:#f6f7f9;"
    "--accent:#1d4ed8;--ok:#15803d;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}"
    "*{box-sizing:border-box}"
    "body{margin:0;background:var(--bg);color:var(--fg);"
    "font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}"
    ".wrap{max-width:44rem;margin:0 auto;padding:2.75rem 1.25rem 4rem}"
    "h1{font-size:1.3rem;font-weight:650;letter-spacing:-.01em;margin:0 0 .35rem}"
    ".sub{color:var(--muted);margin:0 0 1.75rem}"
    "textarea{width:100%;font:inherit;padding:.85rem 1rem;border:1px solid var(--line);"
    "border-radius:10px;resize:vertical;min-height:3.1rem;background:var(--bg);color:inherit}"
    "textarea:focus-visible{outline:2px solid var(--accent);outline-offset:1px;"
    "border-color:var(--accent)}"
    ".row{display:flex;align-items:center;gap:.75rem;margin-top:.75rem}"
    "button{font:inherit;cursor:pointer;border-radius:8px;border:1px solid var(--line);"
    "background:var(--bg);color:inherit;padding:.5rem .9rem}"
    "button:focus-visible{outline:2px solid var(--accent);outline-offset:1px}"
    "#ask{background:var(--accent);color:#fff;border-color:var(--accent);font-weight:550;"
    "padding:.55rem 1.4rem}"
    "#ask:disabled{opacity:.55;cursor:default}"
    "#status{color:var(--muted);font-size:.9rem}"
    ".examples{display:flex;flex-wrap:wrap;gap:.4rem;margin-top:.9rem}"
    ".chip{font-size:.85rem;background:var(--panel);border:1px solid var(--line);color:inherit;"
    "padding:.3rem .7rem;border-radius:999px}"
    ".chip:hover{border-color:var(--accent);color:var(--accent)}"
    "#out{margin-top:1.75rem}"
    ".card{border:1px solid var(--line);border-radius:12px;padding:1.1rem 1.25rem}"
    ".card .label{display:block;font-size:.74rem;font-weight:650;letter-spacing:.05em;"
    "text-transform:uppercase;margin-bottom:.55rem}"
    ".card.answered{border-left:3px solid var(--ok)}"
    ".card.answered .label{color:var(--ok)}"
    ".card.declined{border-left:3px solid var(--muted);background:var(--panel)}"
    ".card.declined .label{color:var(--muted)}"
    ".answer{font-size:1.05rem;margin:0;white-space:pre-wrap}"
    ".cites{list-style:none;margin:.85rem 0 0;padding:.65rem 0 0;border-top:1px solid var(--line)}"
    ".cites li{font:.85rem/1.6 var(--mono);color:var(--muted);padding:.05rem 0}"
    ".cites .doc{color:var(--fg)}"
    ".spin{display:inline-block;width:.85rem;height:.85rem;border:2px solid var(--line);"
    "border-top-color:var(--accent);border-radius:50%;animation:spin .7s linear infinite;"
    "vertical-align:-1px;margin-right:.4rem}"
    "@keyframes spin{to{transform:rotate(360deg)}}"
    "footer{margin-top:3rem;border-top:1px solid var(--line);padding-top:1.25rem;"
    "color:var(--muted);font-size:.9rem}"
    "summary{cursor:pointer;font-weight:550;color:var(--fg)}"
    "details[open] summary{margin-bottom:.5rem}"
    "footer code{font:.85em var(--mono);background:var(--panel);padding:.05rem .3rem;"
    "border-radius:4px}"
    "a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}"
)

# The client logic. Plain string so the JS braces are literal; appended after the HTML skeleton.
_PAGE_JS = """<script>
const qEl = document.getElementById('q');
const out = document.getElementById('out');
const st = document.getElementById('status');
const btn = document.getElementById('ask');
const form = document.getElementById('askform');
function esc(s){
  return (s == null ? '' : String(s))
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
document.querySelectorAll('.chip').forEach(function(b){
  b.addEventListener('click', function(){ qEl.value = b.getAttribute('data-q'); qEl.focus(); });
});
function render(d){
  if (d.abstained){
    out.innerHTML =
      '<div class="card declined"><span class="label">No answer in these documents</span>'
      + '<p class="answer">The IRDAI Master Circular and its Annexure do not cover this '
      + 'question, so the system declined rather than guess.</p>'
      + '<ul class="cites"><li>reason <span class="doc">'
      + esc(d.abstain_reason || 'unknown') + '</span></li></ul></div>';
    return;
  }
  let h = '<div class="card answered"><span class="label">Answered</span>'
    + '<p class="answer">' + esc(d.answer) + '</p>';
  if (d.citations && d.citations.length){
    h += '<ul class="cites">';
    d.citations.forEach(function(c){
      h += '<li><span class="doc">' + esc(c.document_id) + '</span> &middot; p.' + esc(c.page)
        + (c.clause_id ? ' &middot; cl. ' + esc(c.clause_id) : '') + '</li>';
    });
    h += '</ul>';
  }
  out.innerHTML = h + '</div>';
}
async function ask(){
  const question = qEl.value.trim();
  if (question.length < 3){ qEl.focus(); return; }
  btn.disabled = true;
  st.innerHTML = '<span class="spin"></span>searching the documents...';
  out.innerHTML = '';
  try {
    const r = await fetch('/ask', {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify({question: question}),
    });
    if (r.status === 429){
      out.innerHTML = '<div class="card declined"><span class="label">Rate limited</span>'
        + '<p class="answer">Too many requests - wait a moment and try again.</p></div>';
      return;
    }
    render(await r.json());
  } catch (e){
    out.innerHTML = '<div class="card declined"><span class="label">Error</span>'
      + '<p class="answer">Request failed: ' + esc(e) + '</p></div>';
  } finally {
    btn.disabled = false; st.textContent = '';
  }
}
form.addEventListener('submit', function(e){ e.preventDefault(); ask(); });
</script>
</body></html>"""


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
        gen = "configured" if configured else "not configured (set ARAG_GEMINI_API_KEY)"
        return (
            f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Agentic RAG — IRDAI regulation</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{_PAGE_CSS}</style></head>
<body><div class="wrap">
<h1>Agentic RAG — IRDAI health-insurance regulation</h1>
<p class="sub">Ask about the two public IRDAI 2024 documents — the Master Circular on Health
Insurance Business and its Annexure.</p>
<form id="askform" autocomplete="off">
<textarea id="q" rows="2" aria-label="Your question"
  placeholder="e.g. What is the grace period for paying premium in instalments?"></textarea>
<div class="row"><button id="ask" type="submit">Ask</button><span id="status"></span></div>
</form>
<div class="examples" aria-label="Example questions">
<button type="button" class="chip" data-q="{_Q_GRACE}">Grace period</button>
<button type="button" class="chip" data-q="{_Q_FREELOOK}">Free-look period</button>
<button type="button" class="chip" data-q="{_Q_ROOMRENT}">Room-rent sub-limit</button>
</div>
<section id="out" aria-live="polite"></section>
<footer>
<details><summary>Scope &amp; measured limitations</summary>
<p><strong>Scope, stated plainly.</strong> This public instance serves answers
over <em>public regulatory documents only</em>. The four insurer policy wordings, the full
six-source corpus, and every measured result (retrieval, abstention, grounding) are
<strong>local and not served here</strong> — the wordings are copyrighted and the project does
not ship them. A live answer here demonstrates the pipeline over public regulation, not the
measured system.</p>
<p><strong>Measured limitation — read before trusting an answer.</strong>
<strong>Every measured number in this project comes from a local 1.5B instrument model, not from
the model serving this page.</strong> On that labelled 30-item eval, abstention was unreliable —
three independent signals each failed to tell answerable from unanswerable, and the system
correctly refused only <strong>2 of 5</strong> questions the documents do not address. This
instance generates with a hosted production model (Gemini), whose behaviour here is
<strong>unmeasured</strong> beyond two probe questions (one answered, grounded; one correctly
abstained). So it can still answer, with apparent confidence, a question it should decline. Treat
every answer as a <em>pipeline demonstration, not advice</em>, and verify it against the cited
pages. The measurements are in the repo (<code>docs/AGENT_EVAL.md</code>) — documented, not
hidden.</p>
</details>
<p>Engine <code>{engine.name}</code> · generation {gen} ·
<a href="/health">/health</a> · <a href="/docs">/docs</a></p>
</footer>
</div>
"""
            + _PAGE_JS
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
