"""Reporting: console output, a machine-readable run log, and a regenerated EVAL_LOG.md.

The metric history is the most valuable artefact this project produces, so it is stored as
**append-only JSONL and rendered to Markdown**, never hand-edited. A hand-maintained
markdown table rots the first time someone forgets to update it, and once it has rotted
the v1 to v2 improvement story stops being evidence. Deriving the document from the data
means the history cannot silently disagree with the runs that produced it.

``data/eval_runs.jsonl`` is the source of truth. ``EVAL_LOG.md`` is a view.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from arag.eval.runner import EvalRun
from arag.obs import unverified_models

_METRIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("recall@10", "recall_at_k"),
    ("nDCG@10", "ndcg_at_k"),
    ("MRR", "mrr"),
    ("ctx prec", "precision_at_k"),
    ("abstain rec", "abstain_recall"),
    ("false abst", "false_abstention_rate"),
    ("inj resist", "injection_resistance"),
    ("ungrounded", "ungrounded_citation_rate"),
    ("p95 flat", "p95_flat_ms"),
    ("p95 agent", "p95_agentic_ms"),
)

# Expectation columns are rendered separately from the metric table: they are counts, not
# means, and burying "1 known-failure started passing" inside a row of averages is exactly
# how that signal gets missed.
_EXPECTATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("pass", "passed"),
    ("fail", "failed"),
    ("exp-fail", "failed_as_expected"),
    ("FIXED", "fixed"),
)


def _fmt(value: Any, *, ms: bool = False) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)) and not ms:
        return str(value)
    if ms:
        return f"{float(value):.0f}ms"
    return f"{float(value):.3f}"


def to_record(run: EvalRun) -> dict[str, Any]:
    """Flatten a run for the JSONL log. Per-item detail is deliberately excluded: it is
    large, it changes on every run, and it belongs in traces rather than in a file that
    must stay diffable across dozens of runs.
    """
    return {
        "finished_at": run.finished_at.isoformat(),
        "engine": run.engine,
        "profile": run.profile,
        "git_sha": run.git_sha,
        "price_table_version": run.price_table_version,
        "golden_source": run.golden_source,
        "n_items": run.n_items,
        "duration_s": round(run.duration_s, 2),
        "gate_passed": bool(run.gate and run.gate.passed),
        "gate_failures": list(run.gate.failures) if run.gate else [],
        "overall": run.overall,
        "by_strata": run.by_strata,
        "by_provenance": run.by_provenance,
        "warnings": run.warnings,
        "cassette": run.cassette_stats,
    }


def persist_run(run: EvalRun, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(to_record(run), default=str, ensure_ascii=False) + "\n")


def load_runs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def render_console(run: EvalRun) -> str:
    lines: list[str] = []
    o = run.overall
    lines.append("")
    lines.append(f"  engine={run.engine}  profile={run.profile}  sha={run.git_sha}")
    lines.append(
        f"  items={run.n_items}  scored={o.get('n_scored')}  "
        f"retrieval-n/a={o.get('n_retrieval_skipped')}  errors={o.get('errors')}  "
        f"{run.duration_s:.1f}s"
    )
    lines.append("")
    width = max(len(label) for label, _ in _METRIC_COLUMNS) + 2
    for label, key in _METRIC_COLUMNS:
        is_ms = key.endswith("_ms")
        lines.append(f"  {label:<{width}}{_fmt(o.get(key), ms=is_ms)}")
    lines.append("")

    if run.calibration and run.calibration.n:
        c = run.calibration
        lines.append(
            f"  calibration   answers ok={c.correct_answers} wrong={c.wrong_answers}  "
            f"abstentions ok={c.correct_abstentions} wrong={c.wrong_abstentions}  "
            f"expected cost={c.expected_cost:.1f} (at {c.wrong_answer_cost:.0f}:1)"
        )
    lines.append(
        f"  cost          marginal ${o.get('marginal_usd', 0):.6f}  "
        f"shadow ${o.get('shadow_usd', 0):.6f}  "
        f"tokens {o.get('input_tokens', 0)}in/{o.get('output_tokens', 0)}out"
    )
    lines.append("")

    if run.by_strata:
        lines.append("  by stratum")
        lines.append(f"    {'stratum':<24}{'n':>4}  {'recall':>7}{'nDCG':>8}{'ndcg n':>8}")
        for name, m in run.by_strata.items():
            lines.append(
                f"    {name:<24}{m.get('n_items', 0):>4}  "
                f"{_fmt(m.get('recall_at_k')):>7}{_fmt(m.get('ndcg_at_k')):>8}"
                f"{m.get('n_scored', 0):>8}"
            )
        lines.append("")

    exp = o.get("expectations") or {}
    if exp:
        lines.append("  expectation")
        lines.append(
            f"    passed {exp.get('passed', 0)}   "
            f"failed {exp.get('failed', 0)}   "
            f"failed-as-expected {exp.get('failed_as_expected', 0)}   "
            f"FIXED {exp.get('fixed', 0)}"
        )
        if o.get("fixed_ids"):
            # The headline when it is non-empty: a known-failing item now passes. Printed
            # first and named, because it is the fix landing and the label needs updating.
            lines.append(
                f"    *** FIXED - these were expected to fail and PASSED: "
                f"{', '.join(o['fixed_ids'])}"
            )
            lines.append(
                "        Update expected_to_fail on those items, and record the run in "
                "EVAL_LOG.md as the fix."
            )
        if o.get("regressed_ids"):
            lines.append(f"    unexpected failures: {', '.join(o['regressed_ids'])}")
        if o.get("still_failing_ids"):
            lines.append(
                f"    known-failing, still failing (not a regression): "
                f"{', '.join(o['still_failing_ids'])}"
            )
        lines.append("")

    if run.warnings:
        lines.append("  warnings")
        for w in run.warnings:
            lines.append(f"    ! {w}")
        lines.append("")

    if run.gate:
        lines.append("  " + run.gate.summary().replace("\n", "\n  "))
    lines.append("")
    return "\n".join(lines)


def render_eval_log(runs: list[dict[str, Any]]) -> str:
    """Regenerate EVAL_LOG.md from the run log.

    Newest first, because the question a reader has is "where is it now, and what moved?".
    """
    out: list[str] = []
    out.append("# Evaluation log")
    out.append("")
    out.append(
        "Generated by `arag-eval report` from `data/eval_runs.jsonl`. **Do not edit by hand** — "
        "regenerate it. Every row is one recorded run; the commit and threshold profile that "
        "produced it are recorded so any number quoted elsewhere can be traced back here."
    )
    out.append("")
    unverified = unverified_models()
    if unverified:
        out.append(
            f"> **Cost figures are estimates.** Price table entries not yet verified against "
            f"provider pricing pages: {', '.join(unverified)}."
        )
        out.append("")

    if not runs:
        out.append("_No runs recorded yet._")
        out.append("")
        return "\n".join(out)

    out.append("## History")
    out.append("")
    header = ["when", "engine", "profile", "sha", "n", "gate", *[c for c, _ in _METRIC_COLUMNS]]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    for rec in reversed(runs):
        o = rec.get("overall", {})
        when = str(rec.get("finished_at", ""))[:16].replace("T", " ")
        row = [
            when,
            str(rec.get("engine", "")),
            str(rec.get("profile", "")),
            f"`{rec.get('git_sha', '')}`",
            str(rec.get("n_items", "")),
            "pass" if rec.get("gate_passed") else "**fail**",
        ]
        row += [_fmt(o.get(key), ms=key.endswith("_ms")) for _, key in _METRIC_COLUMNS]
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    latest = runs[-1]
    out.append(f"## Latest run — {latest.get('engine')} @ `{latest.get('git_sha')}`")
    out.append("")

    exp = (latest.get("overall") or {}).get("expectations") or {}
    if exp:
        out.append("### Expectation")
        out.append("")
        out.append(
            "`expected_to_fail` is a two-sided assertion. A known-failing item that starts "
            "passing is the fix landing, and it is reported here rather than folded into "
            "the pass count where nobody would see it."
        )
        out.append("")
        out.append("| " + " | ".join(c for c, _ in _EXPECTATION_COLUMNS) + " |")
        out.append("|" + "---|" * len(_EXPECTATION_COLUMNS))
        out.append("| " + " | ".join(str(exp.get(k, 0)) for _, k in _EXPECTATION_COLUMNS) + " |")
        out.append("")
        overall = latest.get("overall") or {}
        if overall.get("fixed_ids"):
            out.append(
                f"**FIXED this run:** `{'`, `'.join(overall['fixed_ids'])}` — expected to "
                "fail, passed. Update `expected_to_fail` on those items."
            )
            out.append("")
        if overall.get("regressed_ids"):
            out.append(f"**Unexpected failures:** `{'`, `'.join(overall['regressed_ids'])}`")
            out.append("")
        if overall.get("still_failing_ids"):
            out.append(
                f"Known-failing, still failing: `{'`, `'.join(overall['still_failing_ids'])}` "
                "(recorded defects, not regressions)"
            )
            out.append("")
    strata = latest.get("by_strata", {})
    if strata:
        out.append("### By stratum")
        out.append("")
        out.append("| stratum | items | scored | recall@10 | nDCG@10 | MRR | false abstention |")
        out.append("|---|---|---|---|---|---|---|")
        for name, m in strata.items():
            out.append(
                f"| {name} | {m.get('n_items', 0)} | {m.get('n_scored', 0)} | "
                f"{_fmt(m.get('recall_at_k'))} | {_fmt(m.get('ndcg_at_k'))} | "
                f"{_fmt(m.get('mrr'))} | {_fmt(m.get('false_abstention_rate'))} |"
            )
        out.append("")

    prov = latest.get("by_provenance", {})
    if prov:
        out.append("### By label provenance")
        out.append("")
        out.append(
            "Aggregate metrics over mixed-provenance labels are not evidence. This slice is "
            'what makes a claim like "recall on human-authored items" checkable.'
        )
        out.append("")
        out.append("| authored_by | items | recall@10 | nDCG@10 |")
        out.append("|---|---|---|---|")
        for name, m in prov.items():
            out.append(
                f"| {name} | {m.get('n_items', 0)} | {_fmt(m.get('recall_at_k'))} | "
                f"{_fmt(m.get('ndcg_at_k'))} |"
            )
        out.append("")

    warnings = latest.get("warnings") or []
    if warnings:
        out.append("### Warnings")
        out.append("")
        for w in warnings:
            out.append(f"- {w}")
        out.append("")

    failures = latest.get("gate_failures") or []
    if failures:
        out.append("### Gate failures")
        out.append("")
        for f in failures:
            out.append(f"- {f}")
        out.append("")

    out.append("---")
    out.append("")
    out.append(f"_Rendered {datetime.now().astimezone().isoformat(timespec='seconds')}._")
    out.append("")
    return "\n".join(out)


def write_eval_log(runs_path: Path, log_path: Path) -> Path:
    log_path.write_text(render_eval_log(load_runs(runs_path)), encoding="utf-8")
    return log_path
