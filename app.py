"""Hugging Face Space (Gradio SDK) entrypoint — the free, no-Docker, no-card public instance.

Same IRDAI-only pipeline as the FastAPI service (src/arag/api/app.py), wrapped in a Gradio UI
so it runs on HF's free CPU tier: BM25 over the two PUBLIC IRDAI 2024 documents built at
startup, generation via the hosted Gemini API. The four insurer wordings are not in git, so
they are not in the Space - the corpus boundary is enforced by what was committed.

Scope and the measured abstention caveat are shown in the UI, and citations are metadata only
(document/page/clause) - no endpoint or widget returns raw passage text.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))  # import arag without packaging

import gradio as gr  # noqa: E402

from arag.agent.generate import GeneratingEngine  # noqa: E402
from arag.agent.providers import GeminiProvider  # noqa: E402
from arag.index.build import build_corpus  # noqa: E402
from arag.ingest.normalise import normalise_query  # noqa: E402
from arag.retrieval.lexical import LexicalRetriever  # noqa: E402

# Build the IRDAI-only BM25 corpus once at startup (pymupdf parse of the 2 public PDFs).
_corpus = build_corpus(
    ROOT / "data" / "deploy" / "irdai_sources.jsonl",
    ROOT / "data" / "raw",
    serialisation="row_nl",
)
_engine = GeneratingEngine(
    LexicalRetriever(_corpus.index, _corpus.chunks),
    GeminiProvider(),
    top_k=5,
    name="agent-irdai",
)

INTRO = """# Agentic RAG — Indian health-insurance regulation
Retrieval-augmented QA over the **two public IRDAI 2024 documents** (Master Circular on Health
Insurance Business + Annexure).

**Scope.** This serves answers over *public regulatory documents only*. The four insurer policy
wordings, the full six-source corpus, and every measured result are **local and not served
here** — the wordings are copyrighted. A live answer demonstrates the pipeline over public
regulation, not the measured system.

> **Measured limitation — read before trusting an answer.** This runs a small instrument model
> and **its abstention is unreliable**: across three independent signals measured on the
> labelled set, none reliably told answerable from unanswerable, and on the 30-item eval the
> system correctly refused only **2 of 5** questions the documents do not address. It can
> answer, with apparent confidence, a question it should decline. Treat every answer as a
> *pipeline demonstration, not advice*, and verify it against the cited pages. The measurement
> is in the repo (`docs/AGENT_EVAL.md`) — documented, not hidden.
"""


def ask(question: str) -> tuple[str, list[list]]:
    q = normalise_query(question or "")
    if len(q.strip()) < 3:
        return "_Enter a question (at least 3 characters)._", []
    result = asyncio.run(_engine.answer(q))
    if result.abstained:
        reason = result.abstain_reason.value if result.abstain_reason else "unknown"
        if reason == "generation_not_configured":
            detail = "the generator is not configured — set ARAG_GEMINI_API_KEY as a Space secret"
        elif reason in ("no_candidates", "low_confidence"):
            detail = (
                "the IRDAI documents do not support an answer (for an out-of-scope question, "
                "that is the correct outcome)"
            )
        else:
            detail = "the answer did not pass verification and was discarded"
        answer_md = f"**Abstained** — reason `{reason}`: {detail}."
    else:
        answer_md = result.answer or "_(empty answer)_"
    citations = [[c.span.document_id, c.span.page, c.span.clause_id] for c in result.citations]
    return answer_md, citations


with gr.Blocks(title="Agentic RAG — IRDAI") as demo:
    gr.Markdown(INTRO)
    question = gr.Textbox(
        label="Question",
        placeholder="e.g. What is the grace period when premium is paid in instalments?",
        lines=2,
    )
    ask_btn = gr.Button("Ask", variant="primary")
    answer = gr.Markdown(label="Answer")
    citations = gr.Dataframe(
        headers=["document", "page", "clause"],
        label="Citations (metadata only — no passage text is returned)",
        wrap=True,
    )
    gr.Examples(
        examples=[
            "What is the grace period when premium is paid in instalments?",
            "What is the free-look period?",
            "What is the room rent sub-limit under the Star Comprehensive policy?",
        ],
        inputs=question,
    )
    ask_btn.click(ask, question, [answer, citations])
    question.submit(ask, question, [answer, citations])

if __name__ == "__main__":
    demo.queue(max_size=16).launch(server_name="0.0.0.0")
