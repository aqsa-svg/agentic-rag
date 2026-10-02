"""Threshold config loader and the gate itself.

The gate is a pure function of (aggregate metrics, profile) so it can be unit-tested
without running an eval, and so the CI failure message is generated from the same code
that produces the human-facing report. A gate whose logic lives in a shell script drifts
from the report within a week.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from arag.eval.matching import MatchMode


class Gate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_recall_at_k: float | None = None
    min_ndcg_at_k: float | None = None
    min_mrr: float | None = None
    max_false_abstention_rate: float | None = None
    min_abstain_recall: float | None = None
    min_injection_resistance: float | None = None
    max_supersession_violations: int | None = None
    max_ungrounded_citation_rate: float | None = None
    p95_flat_ms: float | None = None
    p95_agentic_ms: float | None = None


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = ""
    gate: Gate = Field(default_factory=Gate)
    exact: dict[str, float] = Field(default_factory=dict)


class Calibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wrong_answer_cost: float = 50.0
    abstention_cost: float = 1.0


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_kappa: float = 0.60
    agreement_sample_size: int = 25


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k: int = 10
    match_mode: MatchMode = MatchMode.CLAUSE_PREFIX
    page_tolerance: int = 0


class Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    price_table_version: str
    active_profile: str
    calibration: Calibration = Field(default_factory=Calibration)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    profiles: dict[str, Profile]

    @classmethod
    def load(cls, path: Path) -> Thresholds:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def profile(self, name: str | None = None) -> Profile:
        key = name or self.active_profile
        if key not in self.profiles:
            raise KeyError(f"unknown profile {key!r}; available: {sorted(self.profiles)}")
        return self.profiles[key]


@dataclass
class GateResult:
    passed: bool
    failures: list[str]
    checks_run: int
    checks_skipped: int

    def summary(self) -> str:
        if self.passed:
            return f"GATE PASS ({self.checks_run} checks, {self.checks_skipped} not configured)"
        return "GATE FAIL:\n" + "\n".join(f"  - {f}" for f in self.failures)


def _cmp(
    failures: list[str],
    label: str,
    actual: float | None,
    limit: float | None,
    *,
    lower_is_better: bool,
) -> tuple[int, int]:
    """Returns (ran, skipped). A threshold of ``None`` is *not configured* and is skipped
    loudly in the count, so "0 checks run" can never look like "all checks passed".
    """
    if limit is None:
        return (0, 1)
    if actual is None:
        failures.append(f"{label}: no measurement produced (threshold {limit})")
        return (1, 0)
    ok = actual <= limit if lower_is_better else actual >= limit
    if not ok:
        direction = "exceeds max" if lower_is_better else "below min"
        failures.append(f"{label}: {actual:.4f} {direction} {limit}")
    return (1, 0)


def check_gate(metrics: dict[str, Any], profile: Profile) -> GateResult:
    """Evaluate a profile's gate against a flat metrics dict.

    Keys expected in ``metrics`` mirror the report's overall section. Missing keys are
    treated as "no measurement produced", which fails any configured threshold rather than
    passing silently — a metric that vanished from the pipeline is a regression too.
    """
    failures: list[str] = []
    ran = skipped = 0
    g = profile.gate

    for label, key, limit, lower in (
        ("recall@k", "recall_at_k", g.min_recall_at_k, False),
        ("ndcg@k", "ndcg_at_k", g.min_ndcg_at_k, False),
        ("mrr", "mrr", g.min_mrr, False),
        ("abstain_recall", "abstain_recall", g.min_abstain_recall, False),
        ("injection_resistance", "injection_resistance", g.min_injection_resistance, False),
        ("false_abstention_rate", "false_abstention_rate", g.max_false_abstention_rate, True),
        (
            "ungrounded_citation_rate",
            "ungrounded_citation_rate",
            g.max_ungrounded_citation_rate,
            True,
        ),
        ("p95_flat_ms", "p95_flat_ms", g.p95_flat_ms, True),
        ("p95_agentic_ms", "p95_agentic_ms", g.p95_agentic_ms, True),
    ):
        r, s = _cmp(failures, label, metrics.get(key), limit, lower_is_better=lower)
        ran += r
        skipped += s

    if g.max_supersession_violations is not None:
        ran += 1
        actual = int(metrics.get("supersession_violations") or 0)
        if actual > g.max_supersession_violations:
            failures.append(
                f"supersession_violations: {actual} exceeds max {g.max_supersession_violations} "
                "(citing a superseded clause is a hard failure)"
            )
    else:
        skipped += 1

    # `exact` assertions: used by v0 to prove the harness registers total failure.
    for key, want in profile.exact.items():
        ran += 1
        got = metrics.get(key)
        if got is None or abs(float(got) - want) > 1e-9:
            failures.append(f"{key}: expected exactly {want}, got {got}")

    return GateResult(
        passed=not failures, failures=failures, checks_run=ran, checks_skipped=skipped
    )
