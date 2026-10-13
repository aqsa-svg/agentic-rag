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

# The visual world is a statutory document, not an AI product. One idea carries it: an answer is
# a short extract with its clause citation set in the margin beside it, the way a gazette prints
# section numbers in the margin. Square corners, hairline rules, a serif for the document voice
# (the answer and its citations), system sans for the interface chrome, a single oxblood accent
# used only on the clause mark, the focus ring and the Ask button. No framework, no build step.
# CSS and JS are plain strings (braces stay literal); the HTML skeleton lives in landing(), where
# the only dynamic bits (engine name, generation status) are interpolated.
_PAGE_CSS = (
    "@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:"
    "ital,wght@0,400;0,600;1,400&display=swap');"
    ":root{--ink:#22201b;--paper:#f7f6f2;--rule:#cfccc2;--margin:#6b6760;"
    "--seal:#7a1f2b;--seal-dark:#651826;--affirm:#3a5a40;"
    "--serif:'Source Serif 4',Georgia,'Times New Roman',serif;"
    "--sans:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;"
    "--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}"
    "*{box-sizing:border-box}"
    "body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 var(--sans);"
    "-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}"
    ".wrap{max-width:46rem;margin:0 auto;padding:3rem 1.25rem 4rem}"
    "header{border-top:2px solid var(--ink);padding-top:.9rem;margin-bottom:2.25rem}"
    ".docref{display:flex;justify-content:space-between;align-items:baseline;gap:1rem;"
    "font-size:.78rem;color:var(--margin);margin:0 0 .7rem}"
    ".docref .sys{font-family:var(--serif);font-style:italic;color:var(--seal)}"
    ".mast{font-family:var(--serif);font-size:1.6rem;font-weight:600;line-height:1.2;"
    "letter-spacing:-.005em;margin:0}"
    ".mast-rule{height:0;border-top:1px solid var(--rule);margin:.85rem 0 .7rem}"
    ".mast-sub{color:var(--margin);font-size:.95rem;line-height:1.55;margin:0;max-width:36rem}"
    ".qlabel{display:block;font-size:.8rem;font-weight:600;color:var(--margin);margin:0 0 .4rem}"
    "textarea{width:100%;font:1.05rem/1.5 var(--serif);padding:.75rem .85rem;"
    "border:1px solid var(--ink);border-radius:0;background:#fff;color:var(--ink);"
    "resize:vertical;min-height:3rem}"
    "textarea::placeholder{color:#9a968d;font-style:italic}"
    "textarea:focus-visible,button:focus-visible,a:focus-visible,summary:focus-visible{"
    "outline:2px solid var(--seal);outline-offset:2px}"
    ".row{display:flex;align-items:center;gap:1rem;margin-top:.85rem}"
    "#ask{font:600 .95rem var(--sans);background:var(--seal);color:#fff;"
    "border:1px solid var(--seal);border-radius:0;padding:.6rem 1.6rem;cursor:pointer}"
    "#ask:hover{background:var(--seal-dark);border-color:var(--seal-dark)}"
    "#ask:disabled{opacity:.5;cursor:default}"
    "#status{color:var(--margin);font-size:.9rem}"
    ".examples{margin-top:1rem;font-size:.9rem;color:var(--margin)}"
    ".chip{font:inherit;background:none;border:0;color:var(--seal);cursor:pointer;padding:0;"
    "border-bottom:1px solid var(--rule)}"
    ".chip:hover{border-bottom-color:var(--seal)}"
    ".examples .chip{margin-left:.9rem}"
    ".demolink{margin-top:.75rem;font-size:.85rem}"
    ".js-demo{font:inherit;background:none;border:0;color:var(--margin);cursor:pointer;padding:0;"
    "text-decoration:underline;text-decoration-color:var(--rule)}"
    ".js-demo:hover{color:var(--ink)}"
    "#out{margin-top:2rem}"
    ".banner{font:.82rem/1.5 var(--sans);color:var(--margin);background:#fff;"
    "border:1px dashed var(--rule);padding:.55rem .7rem;margin:0 0 .85rem}"
    ".banner strong{color:var(--seal);font-weight:600}"
    ".extract{border:1px solid var(--ink);background:#fff}"
    ".extract-body{display:grid;grid-template-columns:12rem 1fr}"
    ".cite-margin{padding:1.1rem .95rem;border-right:1px solid var(--rule);align-self:start}"
    ".cite-head{font:italic .8rem/1.3 var(--serif);color:var(--seal);margin:0 0 .65rem}"
    ".cite{font-family:var(--serif);font-size:.86rem;line-height:1.4;margin:0 0 .75rem}"
    ".cite:last-child{margin-bottom:0}"
    ".cite .doc{display:block;color:var(--ink);font-weight:600;word-break:break-word}"
    ".cite .loc{color:var(--margin)}"
    ".extract-main{padding:1.1rem 1.2rem 1.25rem}"
    ".answer{font-family:var(--serif);font-size:1.14rem;line-height:1.6;margin:0;"
    "white-space:pre-wrap}"
    ".note{border:1px solid var(--rule);background:#fbfbf8;padding:1.1rem 1.2rem}"
    ".note .head{font:600 .85rem var(--sans);color:var(--margin);margin:0 0 .45rem}"
    ".note .answer{font-size:1.05rem}"
    ".reason{font-size:.82rem;color:var(--margin);margin:.7rem 0 0}"
    ".card.error{border:1px solid var(--seal);border-left:4px solid var(--seal);"
    "background:#fcf4f4;padding:1.1rem 1.2rem}"
    ".card.error .head{font:600 .85rem var(--sans);color:var(--seal);margin:0 0 .45rem}"
    ".card.error .answer{font-size:1.05rem}"
    ".card.error .demo-hint{font-size:.85rem;color:var(--margin);margin:.7rem 0 0}"
    "code{font:.85em var(--mono);background:var(--paper);border:1px solid var(--rule);"
    "padding:.02rem .3rem}"
    ".spin{display:inline-block;width:.8rem;height:.8rem;border:2px solid var(--rule);"
    "border-top-color:var(--seal);border-radius:50%;vertical-align:-1px;margin-right:.45rem}"
    "footer{margin-top:3.5rem;border-top:1px solid var(--rule);padding-top:1.25rem;"
    "color:var(--margin);font-size:.9rem}"
    "summary{cursor:pointer;color:var(--ink);font-weight:600}"
    "details[open] summary{margin-bottom:.6rem}"
    "footer p{max-width:42rem;line-height:1.6}"
    "footer code{background:#fff}"
    "footer a{color:var(--seal);text-decoration:none;border-bottom:1px solid var(--rule)}"
    "footer a:hover{border-bottom-color:var(--seal)}"
    "@media (max-width:34rem){"
    ".extract-body{grid-template-columns:1fr}"
    ".extract-main{order:1;padding-bottom:1rem}"
    ".cite-margin{order:2;border-right:0;border-top:1px solid var(--rule);padding:.9rem 1.2rem}"
    ".docref{flex-direction:column;gap:.15rem}.mast{font-size:1.4rem}"
    "}"
    "@media (prefers-reduced-motion:no-preference){"
    ".reveal{animation:reveal .24s ease-out}"
    "@keyframes reveal{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}"
    ".spin{animation:spin .7s linear infinite}"
    "@keyframes spin{to{transform:rotate(360deg)}}"
    "}"
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
document.querySelectorAll('.examples .chip').forEach(function(b){
  b.addEventListener('click', function(){ qEl.value = b.getAttribute('data-q'); qEl.focus(); });
});
const ERROR_REASONS = [
  'generation_unavailable', 'generation_not_configured', 'malformed_generation',
  'service_degraded', 'generation_forbidden', 'generation_blocked'
];
// A real grace-period answer this service produced on 13 Oct 2026 (probe (a), docs/AGENT_EVAL.md),
// kept so the extract card is visible while live generation is on quota. Clearly labelled on show.
const DEMO = {
  abstained: false,
  answer: 'The grace period for paying the premium is fifteen days where the premium is paid in '
    + 'monthly instalments, and thirty days where it is paid quarterly, half-yearly or annually.',
  citations: [{document_id: 'irdai-master-circular-2024', page: 7, clause_id: '8'}]
};
function citeLoc(c){
  const parts = [];
  if (c.clause_id != null && c.clause_id !== '') parts.push('Clause ' + esc(c.clause_id));
  if (c.page != null) parts.push('p.' + esc(c.page));
  return parts.join(' &middot; ');
}
function citesMargin(cites){
  // Answer-level authority, stacked and top-aligned in the margin - NOT mapped to spans. The
  // AnswerResult carries citations for the answer as a whole, so none is pinned to a line.
  let h = '<div class="cite-margin"><p class="cite-head">&sect; Cited authority</p>';
  if (cites && cites.length){
    cites.forEach(function(c){
      h += '<p class="cite"><span class="doc">' + esc(c.document_id) + '</span>'
        + '<span class="loc">' + citeLoc(c) + '</span></p>';
    });
  } else {
    h += '<p class="cite"><span class="loc">none cited</span></p>';
  }
  return h + '</div>';
}
function render(d, opts){
  opts = opts || {};
  const banner = opts.banner || '';
  if (d.abstained){
    const reason = d.abstain_reason || 'unknown';
    if (ERROR_REASONS.indexOf(reason) >= 0){
      // Not a refusal: generation failed. Shown as an error, visibly distinct from abstention.
      out.innerHTML =
        '<article class="card error reveal"><p class="head">Service error &mdash; not an answer</p>'
        + '<p class="answer">The service could not generate an answer. This is a <strong>failure, '
        + 'not a decision</strong>: retrieval ran, but the generation call did not complete '
        + '(<code>' + esc(reason) + '</code>).</p>'
        + '<p class="demo-hint">The live model may be on quota today. '
        + '<button type="button" class="js-demo">See a saved example answer</button></p>'
        + '</article>';
      return;
    }
    let head, body;
    if (reason === 'low_confidence'){
      head = 'Answer declined';
      body = 'The system declined to answer rather than answer without support in the passages.';
    } else {
      head = 'No provision found';
      body = 'The IRDAI Master Circular and its Annexure do not cover this question, '
        + 'so the system declined rather than guess.';
    }
    out.innerHTML =
      '<article class="note reveal"><p class="head">' + head + '</p>'
      + '<p class="answer">' + body + '</p>'
      + '<p class="reason">Reason: <code>' + esc(reason) + '</code></p></article>';
    return;
  }
  out.innerHTML = banner
    + '<article class="extract reveal"><div class="extract-body">' + citesMargin(d.citations)
    + '<div class="extract-main"><p class="answer">' + esc(d.answer) + '</p></div>'
    + '</div></article>';
}
function showDemo(){
  render(DEMO, {banner:
    '<p class="banner"><strong>Saved example.</strong> The live model is on quota today; this is '
    + 'a real grace-period answer this service produced on 13 Oct 2026, cited to the Master '
    + 'Circular &mdash; shown so the answer card is visible.</p>'});
}
async function ask(){
  const question = qEl.value.trim();
  if (question.length < 3){ qEl.focus(); return; }
  btn.disabled = true;
  st.innerHTML = '<span class="spin"></span>searching the documents&hellip;';
  out.innerHTML = '';
  try {
    const r = await fetch('/ask', {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify({question: question}),
    });
    if (r.status === 429){
      out.innerHTML = '<article class="note reveal"><p class="head">Rate limited</p>'
        + '<p class="answer">Too many requests &mdash; wait a moment and try again.</p></article>';
      return;
    }
    render(await r.json());
  } catch (e){
    out.innerHTML = '<article class="note reveal"><p class="head">Request failed</p>'
      + '<p class="answer">' + esc(e) + '</p></article>';
  } finally {
    btn.disabled = false; st.textContent = '';
  }
}
form.addEventListener('submit', function(e){ e.preventDefault(); ask(); });
document.addEventListener('click', function(e){
  const t = e.target.closest ? e.target.closest('.js-demo') : null;
  if (t){ e.preventDefault(); showDemo(); }
});
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
    # top_k=10 (was 5): measured BM25 recall@k on the 20 labelled golden items rises from
    # 0.425 @5 to 0.500 @10 (docs/RETRIEVAL_RECALL.md). BM25 is kept on the public instance
    # (dense needs the embedding model at runtime, too heavy for the free tier); raising k is
    # the free lever. Env-overridable for the eval, which sets its own k from thresholds.
    return GeneratingEngine(
        retriever, provider, top_k=int(os.environ.get("ARAG_TOP_K", "10")), name="agent-irdai"
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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<style>{_PAGE_CSS}</style></head>
<body><div class="wrap">
<header>
<p class="docref"><span>Insurance Regulatory and Development Authority of India
&middot; 2024</span><span class="sys">Agentic RAG</span></p>
<h1 class="mast">Health Insurance — Master Circular &amp; Annexure</h1>
<div class="mast-rule"></div>
<p class="mast-sub">A retrieval-augmented question system over the two public IRDAI 2024
documents. Ask a question; an answer is returned as an extract with the clause it rests on.</p>
</header>
<form id="askform" autocomplete="off">
<label class="qlabel" for="q">Your question</label>
<textarea id="q" rows="2" aria-label="Your question"
  placeholder="e.g. What is the grace period for paying premium in instalments?"></textarea>
<div class="row"><button id="ask" type="submit">Ask</button><span id="status"></span></div>
</form>
<div class="examples" aria-label="Example questions">Try:
<button type="button" class="chip" data-q="{_Q_GRACE}">grace period</button>
<button type="button" class="chip" data-q="{_Q_FREELOOK}">free-look period</button>
<button type="button" class="chip" data-q="{_Q_ROOMRENT}">room-rent sub-limit</button>
</div>
<p class="demolink"><button type="button" class="js-demo">See a saved example answer</button></p>
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
