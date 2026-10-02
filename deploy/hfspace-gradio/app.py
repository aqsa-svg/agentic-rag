"""Hugging Face Space entrypoint, Gradio SDK (no Dockerfile).

The FastAPI app is mounted *under* Gradio rather than replacing it, so the JSON contract
and a human-browsable UI share one process and one port. That matters for a portfolio
piece: a reviewer who opens the URL gets something to click, and a reviewer who reads the
README gets a real API to curl. Both hit the same QueryEngine.

Deliberately honest: the stub abstains on every question and the UI says so, because a
live URL that looks like a working system is worse than no URL.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The Space is the repo: put ./src on the path so `arag` resolves from the pushed
# source. The alternative - publishing the package just to deploy it - would add a
# release step to every deploy for no benefit.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import gradio as gr

from arag.agent.stub import NullEngine
from arag.api.app import create_app
from arag.ingest.normalise import normalise_query

api = create_app()
engine = NullEngine()

STATUS = (
    "**Day-1 deploy.** The engine is a stub: every question abstains by design.\n\n"
    "Wired: normalisation, engine contract, structured logging.\n"
    "Not wired: corpus index, hybrid retrieval, reranker, LangGraph agent, guardrails.\n"
    "Eval gate: no labelled data yet."
)


async def ask(question: str) -> str:
    if not question or len(question.strip()) < 3:
        return "Ask a longer question."
    # Query-side normalisation (N1) - the same function the index path uses.
    result = await engine.answer(normalise_query(question))
    if result.abstained:
        return (
            f"**Abstained** (reason: `{result.abstain_reason}`)\n\n"
            f"That is correct behaviour for the stub engine. Confidence "
            f"{result.confidence:.2f}."
        )
    return result.answer or "(no answer)"


with gr.Blocks(title="Agentic RAG - Indian health insurance") as demo:
    gr.Markdown("# Agentic RAG over Indian health insurance policy wordings")
    gr.Markdown(STATUS)
    box = gr.Textbox(
        label="Question",
        placeholder="Is bariatric surgery covered under Star Comprehensive?",
    )
    out = gr.Markdown()
    gr.Button("Ask").click(ask, inputs=box, outputs=out)
    gr.Markdown("JSON API: `GET /health`, `GET /status`, `POST /ask`")

app = gr.mount_gradio_app(api, demo, path="/")
