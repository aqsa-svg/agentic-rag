"""Run the 30-item golden set end to end on the HOSTED (Gemini) generator.

This is the production-model counterpart to the 1.5B instrument run in docs/AGENT_EVAL.md:
the same pipeline (retrieve -> generate -> parse -> verify -> finalize), scored by the same
committed harness functions (score_retrieval + score_behaviour via eval.runner._run_item),
but with GeminiProvider instead of the local model. It produces the one thing AGENT_EVAL
calls unmeasured: a real grounding + abstention rate on the deployed generator.

Key handling: the key is read from the ``ARAG_GEMINI_API_KEY`` environment variable by
GeminiProvider. This script never reads a dotfile and never takes the key as an argument.
Run it with the key already in your environment:

    ARAG_GEMINI_API_KEY=... python tools/production_eval.py

Resumable: each item's result is appended to the output JSONL as it completes and flushed,
so a crash (or a daily-quota 429 part-way) loses nothing. Re-running skips items already in
the file and continues. Delete the file to start clean.

Costs real Gemini quota/credits. Defaults to --sleep 4 (the ~15 RPM free-tier ceiling) and
--concurrency 1; lower the sleep only if your key has a higher limit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from arag.config import REPO_ROOT, settings
from arag.eval.matching import SpanMatcher
from arag.eval.runner import _run_item
from arag.eval.schema import GoldenSet
from arag.eval.thresholds import Thresholds
from arag.obs import configure_logging, get_logger

log = get_logger("production_eval")

# Abstention reasons that are infrastructure failures, not refusals (mirrors the UI + the
# provider error mapping). An item ending in one of these is a run error, not a decision.
ERROR_REASONS = frozenset(
    {
        "generation_unavailable",
        "generation_not_configured",
        "malformed_generation",
        "service_degraded",
        "generation_forbidden",
        "generation_blocked",
    }
)


def _build_engine(retriever: str, top_k: int):
    from arag.agent.generate import GeneratingEngine
    from arag.agent.providers import GeminiProvider
    from arag.index.build import build_corpus

    man = REPO_ROOT / "data" / "manifest" / "sources.jsonl"
    raw = REPO_ROOT / "data" / "raw"

    if retriever == "dense":
        from arag.index.dense import DenseIndex, SentenceTransformerEmbedder
        from arag.retrieval.dense import DenseRetriever

        corpus = build_corpus(man, raw, serialisation="markdown")
        index = DenseIndex.load(REPO_ROOT / "data" / "dense_v2.npz")
        embedder = SentenceTransformerEmbedder(index.model_id)
        inner = DenseRetriever(index, corpus.chunks, embedder)
    else:
        from arag.retrieval.lexical import LexicalRetriever

        corpus = build_corpus(man, raw, serialisation="row_nl")
        inner = LexicalRetriever(corpus.index, corpus.chunks)

    provider = GeminiProvider()
    log.info(
        "production_engine",
        retriever=retriever,
        top_k=top_k,
        chunks=len(corpus.chunks),
        gemini_configured=provider.configured,
        gemini_model=provider.model,
    )
    if not provider.configured:
        log.warning(
            "gemini_not_configured",
            note="ARAG_GEMINI_API_KEY is unset; every item will abstain "
            "generation_not_configured. This proves the harness runs; it is not a real eval.",
        )
    return GeneratingEngine(inner, provider, top_k=top_k, name=f"gemini-{retriever}")


def _record(ir) -> dict:
    item, r, b, ret = ir.item, ir.result, ir.behaviour, ir.retrieval
    cites = []
    reason = None
    if r is not None:
        cites = [
            {"document_id": c.span.document_id, "page": c.span.page, "clause_id": c.span.clause_id}
            for c in r.citations
        ]
        reason = r.abstain_reason.value if r.abstain_reason else None
    return {
        "id": item.id,
        "strata": item.strata.value,
        "expected_behaviour": item.expected_behaviour.value,
        "has_span": bool(item.ground_truth_spans),
        "abstained": (r.abstained if r is not None else None),
        "abstain_reason": reason,
        "answer": ((r.answer or "")[:500] if r is not None else None),
        "n_citations": len(cites),
        "citations": cites,
        "recall_at_k": (ret.recall_at_k if ret is not None else None),
        "ndcg_at_k": (ret.ndcg_at_k if ret is not None else None),
        "grounded": (b.citations_all_grounded if b is not None else None),
        "abstained_correctly": (b.abstained_correctly if b is not None else None),
        "abstained_wrongly": (b.abstained_wrongly if b is not None else None),
        "injection_resisted": (b.injection_resisted if b is not None else None),
        "answered_without_citations": (b.answered_without_citations if b is not None else None),
        "generate_ms": (r.latency.generate_ms if r is not None else None),
        "wall_ms": round(ir.wall_ms, 1),
        "input_tokens": (r.input_tokens if r is not None else 0),
        "output_tokens": (r.output_tokens if r is not None else 0),
        "marginal_usd": (r.marginal_usd if r is not None else 0.0),
        "error": ir.error,
    }


def _load_done(out: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rec = json.loads(line)
                done[rec["id"]] = rec
    return done


def _summary(records: list[dict]) -> str:
    n = len(records)
    answered = [r for r in records if r["abstained"] is False]
    abstained = [r for r in records if r["abstained"] is True]
    errored = [r for r in records if r["abstain_reason"] in ERROR_REASONS]

    span_answered = [r for r in answered if r["has_span"]]
    grounded = [r for r in span_answered if r["grounded"] is True]

    unans = [r for r in records if r["expected_behaviour"] == "abstain"]
    unans_refused = [r for r in unans if r["abstained"] is True]

    inj = [r for r in records if r["expected_behaviour"] == "ignore_injection"]
    inj_ok = [r for r in inj if r["injection_resisted"] is True]

    span_items = [r for r in records if r["has_span"] and r["recall_at_k"] is not None]
    mean_recall = (
        sum(r["recall_at_k"] for r in span_items) / len(span_items) if span_items else float("nan")
    )
    cost = sum(r["marginal_usd"] for r in records)

    lines = [
        "",
        "=" * 64,
        f"PRODUCTION EVAL SUMMARY  (n={n})",
        "=" * 64,
        f"  answered / abstained         : {len(answered)} / {len(abstained)}",
        f"  generation errors (not refusal): {len(errored)}"
        + (f"  {sorted(r['id'] for r in errored)}" if errored else ""),
        f"  grounded (answered w/ span)  : {len(grounded)} / {len(span_answered)}",
        f"  mean recall@k (span items)   : {mean_recall:.3f}  (n={len(span_items)})",
        f"  abstention on unanswerable   : {len(unans_refused)} / {len(unans)}  "
        "(the headline the 1.5B failed at 2/5)",
        f"  injection resisted           : {len(inj_ok)} / {len(inj)}",
        f"  marginal cost (USD)          : {cost:.4f}",
        "=" * 64,
    ]
    return "\n".join(lines)


async def _run(args: argparse.Namespace) -> int:
    gs = GoldenSet.load(args.golden or settings().golden_path)
    th = Thresholds.load(settings().thresholds_path)
    matcher = SpanMatcher(mode=th.retrieval.match_mode, page_tolerance=th.retrieval.page_tolerance)
    k = th.retrieval.k

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out)

    items = gs.items if args.limit is None else gs.items[: args.limit]
    remaining = [it for it in items if it.id not in done]
    print(f"golden={len(items)} items  already done={len(done)}  to run={len(remaining)}")
    if not remaining:
        print("nothing to run; printing summary from the existing file.")
        print(_summary(list(done.values())))
        return 0

    engine = _build_engine(args.retriever, args.top_k)

    with out.open("a", encoding="utf-8") as fh:
        for i, item in enumerate(remaining, 1):
            started = time.perf_counter()
            ir = await _run_item(item, engine, k=k, matcher=matcher)
            rec = _record(ir)
            done[item.id] = rec
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            took = time.perf_counter() - started
            print(
                f"  [{i}/{len(remaining)}] {item.id:5s} "
                f"abstained={rec['abstained']} reason={rec['abstain_reason']} "
                f"recall={rec['recall_at_k']} grounded={rec['grounded']} {took:.1f}s"
            )
            if i < len(remaining) and args.sleep > 0:
                await asyncio.sleep(args.sleep)

    print(_summary(list(done.values())))
    print(f"\nwrote -> {out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Run the golden set on the hosted Gemini generator.")
    p.add_argument("--golden", type=Path, default=None, help="Golden JSONL (default: config).")
    p.add_argument(
        "--retriever",
        choices=("bm25-row-nl", "dense"),
        default="bm25-row-nl",
        help="bm25-row-nl matches the deployed service; dense is the stronger measured retriever.",
    )
    p.add_argument("--top-k", type=int, default=10, help="Chunks into the generator (deploy=10).")
    p.add_argument(
        "--out",
        type=str,
        default="data/deploy/production_eval_v1.jsonl",
        help="Resumable per-item output JSONL.",
    )
    p.add_argument("--limit", type=int, default=None, help="Cap item count (debugging).")
    p.add_argument(
        "--sleep",
        type=float,
        default=4.0,
        help="Seconds between calls (~15 RPM free-tier ceiling). Lower only on a higher limit.",
    )
    args = p.parse_args()
    configure_logging(settings().log_level)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
