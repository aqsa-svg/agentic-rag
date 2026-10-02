"""The eval runner. One command, one recorded run.

Ordering note: this module was written **before** any retrieval or agent code exists, and
it runs against ``NullEngine``. That is the point of Phase 2. A harness built after the
pipeline is a harness whose thresholds get chosen to make the pipeline look good; a
harness built first has to be honest, because there is nothing yet to flatter.

Two behaviours worth defending:

* **A raising engine is a finding, not a crash.** The ``QueryEngine`` contract says
  expected failures come back as an abstaining ``AnswerResult``. If an engine raises
  anyway, the runner records it as an item error and carries on, so one bad item cannot
  destroy a 120-item run — and ``errors`` appears in the report where it will be noticed.
* **Concurrency is bounded and configurable.** The generator sits behind a 15 RPM free
  tier, so an unbounded ``gather`` over 120 items would spend the run in backoff and make
  the latency numbers meaningless. Default is serial; the caller opts into parallelism.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from arag.agent.protocols import QueryEngine
from arag.agent.types import AnswerResult
from arag.eval.matching import SpanMatcher
from arag.eval.metrics.behavioural import BehaviourScores, CalibrationSummary, score_behaviour
from arag.eval.metrics.retrieval import Aggregate, RetrievalScores, score_retrieval
from arag.eval.schema import AuthoredBy, ExpectedBehaviour, GoldenItem, GoldenSet, Strata
from arag.eval.thresholds import GateResult, Profile, Thresholds, check_gate
from arag.obs import PRICE_TABLE_VERSION, get_logger, new_trace_id, span, trace

log = get_logger(__name__)


class Expectation(StrEnum):
    """Outcome of an item against what the golden set said to expect.

    Four categories, not two, because `expected_to_fail` has to be a two-sided
    assertion. A known-failing item that silently starts passing is the single most
    valuable event in the whole log - it is the fix landing - and a three-category
    scheme would file it under 'pass' where nobody would ever see it.
    """

    # expected to pass, and did. The ordinary case.
    PASSED = "passed"
    # expected to pass, did not. The ordinary failure.
    FAILED = "failed"
    # expected to fail, and did. A recorded, still-open defect. NOT a regression.
    FAILED_AS_EXPECTED = "failed_as_expected"
    # expected to fail, but PASSED. Good news that must be surfaced loudly.
    FIXED = "fixed"

    @property
    def is_regression(self) -> bool:
        """Only an unexpected failure counts against the gate."""
        return self is Expectation.FAILED

    @property
    def needs_attention(self) -> bool:
        """Both surprises deserve a human's eye: an unexpected failure, and a
        known-failure that has started passing (its label needs updating).
        """
        return self in (Expectation.FAILED, Expectation.FIXED)


def item_satisfied(
    item: GoldenItem,
    result: AnswerResult,
    retrieval: RetrievalScores,
    behaviour: BehaviourScores,
) -> bool:
    """Did the system do the right thing for this item?

    Deliberately coarse and retrieval-led at v1, because there is no generation to
    judge yet: an item counts as satisfied when the system produced the right KIND of
    response and, where an answer was expected, actually retrieved a labelled span.
    Answer-correctness joins this once the judge is wired and its kappa is published.

    A hard behavioural failure (citing a superseded document, following an injection,
    a hallucinated citation) overrides everything else. Those are not gradations.
    """
    if behaviour.hard_failure:
        return False

    if item.expected_behaviour is ExpectedBehaviour.ABSTAIN:
        return bool(result.abstained)

    if result.abstained:
        # Every other expectation requires an answer.
        return False

    if item.expected_behaviour is ExpectedBehaviour.SURFACE_CONFLICT:
        return bool(behaviour.conflict_surfaced)

    if item.expected_behaviour is ExpectedBehaviour.IGNORE_INJECTION:
        return behaviour.injection_resisted is not False

    # ExpectedBehaviour.ANSWER: a labelled span must actually have been retrieved.
    # hit_rate is None only when the item carries no spans, which the schema forbids
    # for answer items - so None here means something upstream is wrong, not passing.
    return retrieval.hit_rate == 1.0


def classify_expectation(
    item: GoldenItem,
    result: AnswerResult | None,
    retrieval: RetrievalScores | None,
    behaviour: BehaviourScores | None,
) -> Expectation:
    """Fold the verdict together with what the label said to expect."""
    if result is None or retrieval is None or behaviour is None:
        # The engine raised. Treat as a failure, and as a regression unless the item
        # was already known to fail.
        return Expectation.FAILED_AS_EXPECTED if item.expected_to_fail else Expectation.FAILED

    satisfied = item_satisfied(item, result, retrieval, behaviour)
    if item.expected_to_fail:
        return Expectation.FIXED if satisfied else Expectation.FAILED_AS_EXPECTED
    return Expectation.PASSED if satisfied else Expectation.FAILED


@dataclass
class ItemResult:
    item: GoldenItem
    result: AnswerResult | None
    retrieval: RetrievalScores | None
    behaviour: BehaviourScores | None
    wall_ms: float
    error: str | None = None
    expectation: Expectation | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class EvalRun:
    """Everything needed to reproduce and compare a run.

    The metadata block is not bookkeeping for its own sake: without ``git_sha``,
    ``profile`` and ``price_table_version``, a row in EVAL_LOG.md is an unfalsifiable
    claim. With them, any number in the README can be traced back to the commit and the
    configuration that produced it, which is the standard this project is held to.
    """

    engine: str
    profile: str
    started_at: datetime
    finished_at: datetime
    git_sha: str
    price_table_version: str
    golden_source: str
    n_items: int
    items: list[ItemResult] = field(default_factory=list)
    overall: dict[str, Any] = field(default_factory=dict)
    by_strata: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_provenance: dict[str, dict[str, Any]] = field(default_factory=dict)
    calibration: CalibrationSummary | None = None
    gate: GateResult | None = None
    cassette_stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha = out.stdout.strip()
        if not sha:
            return "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


async def _run_item(
    item: GoldenItem,
    engine: QueryEngine,
    *,
    k: int,
    matcher: SpanMatcher,
) -> ItemResult:
    with trace(new_trace_id()), span(f"eval.{item.id}"):
        started = time.perf_counter()
        try:
            result = await engine.answer(item.question, as_of=item.as_of_date)
        except Exception as exc:
            wall = (time.perf_counter() - started) * 1000
            log.error("eval_item_raised", item_id=item.id, error=str(exc), exc_info=True)
            return ItemResult(
                item=item,
                result=None,
                retrieval=None,
                behaviour=None,
                wall_ms=wall,
                error=f"{type(exc).__name__}: {exc}",
                expectation=classify_expectation(item, None, None, None),
            )
        wall = (time.perf_counter() - started) * 1000

    retrieval = score_retrieval(
        list(result.retrieved), item.ground_truth_spans, k=k, matcher=matcher
    )
    behaviour = score_behaviour(item, result)

    log.info(
        "eval_item_done",
        item_id=item.id,
        strata=item.strata.value,
        abstained=result.abstained,
        recall_at_k=retrieval.recall_at_k,
        ndcg_at_k=retrieval.ndcg_at_k,
        wall_ms=round(wall, 1),
    )
    return ItemResult(
        item=item,
        result=result,
        retrieval=retrieval,
        behaviour=behaviour,
        wall_ms=wall,
        expectation=classify_expectation(item, result, retrieval, behaviour),
    )


def _aggregate(
    results: list[ItemResult], calibration_cfg: Any
) -> tuple[dict[str, Any], CalibrationSummary]:
    """Collapse per-item scores into the flat dict the gate and report both read."""
    aggs = {
        name: Aggregate()
        for name in ("recall_at_k", "precision_at_k", "mrr", "ndcg_at_k", "hit_rate")
    }
    abstain_recall = Aggregate()
    false_abstention = Aggregate()
    injection = Aggregate()
    conflict = Aggregate()
    ungrounded = Aggregate()
    unsupported = Aggregate()
    lat_flat = Aggregate()
    lat_agentic = Aggregate()

    supersession = 0
    errors = 0
    input_tokens = output_tokens = 0
    marginal = shadow = 0.0
    degraded_items = 0
    calib = CalibrationSummary(
        wrong_answer_cost=calibration_cfg.wrong_answer_cost,
        abstention_cost=calibration_cfg.abstention_cost,
    )

    for r in results:
        if not r.ok or r.result is None or r.behaviour is None or r.retrieval is None:
            errors += 1
            continue

        for name, value in r.retrieval.as_dict().items():
            aggs[name].add(value)

        b = r.behaviour
        if b.abstained_correctly is not None:
            abstain_recall.add(1.0 if b.abstained_correctly else 0.0)
        if b.abstained_wrongly is not None:
            false_abstention.add(1.0 if b.abstained_wrongly else 0.0)
        if b.injection_resisted is not None:
            injection.add(1.0 if b.injection_resisted else 0.0)
        if b.conflict_surfaced is not None:
            conflict.add(1.0 if b.conflict_surfaced else 0.0)
        if b.citations_all_grounded is not None:
            ungrounded.add(0.0 if b.citations_all_grounded else 1.0)
        if b.answered_without_citations is not None:
            unsupported.add(1.0 if b.answered_without_citations else 0.0)
        supersession += len(b.supersession_violations)

        (lat_agentic if r.item.strata.needs_agent else lat_flat).add(r.wall_ms)

        input_tokens += r.result.input_tokens
        output_tokens += r.result.output_tokens
        marginal += r.result.marginal_usd
        shadow += r.result.shadow_usd
        if r.result.degraded:
            degraded_items += 1

        # Decision-theoretic tally. "Wrong answer" is approximated at v0 by an answer
        # that is either ungrounded or given where an abstention was expected; once the
        # judge is wired in (v1) correctness replaces the approximation and this comment
        # gets deleted rather than left to rot.
        expected_abstain = r.item.expected_behaviour is ExpectedBehaviour.ABSTAIN
        if r.result.abstained:
            if expected_abstain:
                calib.correct_abstentions += 1
            else:
                calib.wrong_abstentions += 1
        else:
            if expected_abstain or b.hard_failure:
                calib.wrong_answers += 1
            else:
                calib.correct_answers += 1

    expectations: dict[str, int] = {e.value: 0 for e in Expectation}
    fixed_ids: list[str] = []
    regressed_ids: list[str] = []
    still_failing_ids: list[str] = []
    for r in results:
        if r.expectation is None:
            continue
        expectations[r.expectation.value] += 1
        if r.expectation is Expectation.FIXED:
            fixed_ids.append(r.item.id)
        elif r.expectation is Expectation.FAILED:
            regressed_ids.append(r.item.id)
        elif r.expectation is Expectation.FAILED_AS_EXPECTED:
            still_failing_ids.append(r.item.id)

    overall: dict[str, Any] = {name: agg.mean for name, agg in aggs.items()}
    overall.update(
        {
            "n_scored": aggs["recall_at_k"].n,
            "n_retrieval_skipped": aggs["recall_at_k"].skipped,
            "abstain_recall": abstain_recall.mean,
            "abstain_recall_n": abstain_recall.n,
            "false_abstention_rate": false_abstention.mean,
            "false_abstention_n": false_abstention.n,
            "injection_resistance": injection.mean,
            "injection_n": injection.n,
            "conflict_surfaced_rate": conflict.mean,
            "conflict_n": conflict.n,
            "ungrounded_citation_rate": ungrounded.mean,
            "unsupported_answer_rate": unsupported.mean,
            "supersession_violations": supersession,
            "errors": errors,
            "p95_flat_ms": lat_flat.percentile(95),
            "p95_agentic_ms": lat_agentic.percentile(95),
            "mean_flat_ms": lat_flat.mean,
            "mean_agentic_ms": lat_agentic.mean,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "marginal_usd": round(marginal, 6),
            "shadow_usd": round(shadow, 6),
            "degraded_items": degraded_items,
            "expected_cost": calib.expected_cost,
            # Two-sided expectation assertion. `fixed` is the headline when non-empty:
            # a known-failing item has started passing, which is the fix landing.
            "expectations": expectations,
            "fixed_ids": sorted(fixed_ids),
            "regressed_ids": sorted(regressed_ids),
            "still_failing_ids": sorted(still_failing_ids),
        }
    )
    return overall, calib


def _slice(results: list[ItemResult], calibration_cfg: Any, key: Any) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[ItemResult]] = {}
    for r in results:
        buckets.setdefault(key(r.item), []).append(r)
    return {
        name: _aggregate(rs, calibration_cfg)[0] | {"n_items": len(rs)}
        for name, rs in sorted(buckets.items())
    }


async def run_eval(
    golden: GoldenSet,
    engine: QueryEngine,
    thresholds: Thresholds,
    *,
    profile_name: str | None = None,
    concurrency: int = 1,
    cassette_stats: dict[str, Any] | None = None,
) -> EvalRun:
    profile: Profile = thresholds.profile(profile_name)
    matcher = SpanMatcher(
        mode=thresholds.retrieval.match_mode,
        page_tolerance=thresholds.retrieval.page_tolerance,
    )
    k = thresholds.retrieval.k
    started = datetime.now(UTC)

    log.info(
        "eval_run_start",
        engine=engine.name,
        profile=profile_name or thresholds.active_profile,
        n_items=len(golden.items),
        k=k,
        concurrency=concurrency,
    )

    sem = asyncio.Semaphore(max(1, concurrency))

    async def guarded(item: GoldenItem) -> ItemResult:
        async with sem:
            return await _run_item(item, engine, k=k, matcher=matcher)

    results = await asyncio.gather(*(guarded(i) for i in golden.items))
    item_results = list(results)
    finished = datetime.now(UTC)

    overall, calib = _aggregate(item_results, thresholds.calibration)
    run = EvalRun(
        engine=engine.name,
        profile=profile_name or thresholds.active_profile,
        started_at=started,
        finished_at=finished,
        git_sha=_git_sha(),
        price_table_version=PRICE_TABLE_VERSION,
        golden_source=str(golden.source or "<memory>"),
        n_items=len(golden.items),
        items=item_results,
        overall=overall,
        by_strata=_slice(item_results, thresholds.calibration, lambda i: i.strata.value),
        by_provenance=_slice(item_results, thresholds.calibration, lambda i: i.authored_by.value),
        calibration=calib,
        cassette_stats=cassette_stats or {},
    )
    run.gate = check_gate(overall, profile)
    run.warnings = _warnings(golden, thresholds, run)

    log.info(
        "eval_run_done",
        engine=run.engine,
        gate_passed=run.gate.passed,
        duration_s=round(run.duration_s, 2),
        recall_at_k=overall.get("recall_at_k"),
        ndcg_at_k=overall.get("ndcg_at_k"),
    )
    return run


def _warnings(golden: GoldenSet, thresholds: Thresholds, run: EvalRun) -> list[str]:
    """Things that make the numbers less trustworthy than they look.

    Surfaced on every run rather than only on request, because the whole risk with an
    unverified golden set or a stale price table is that nobody remembers to check.
    """
    out: list[str] = []
    unverified = golden.unverified()
    if unverified:
        out.append(
            f"{len(unverified)}/{len(golden.items)} golden items are authored_by=seed_unverified. "
            "Metrics computed against unverified labels are development signal, not evidence, "
            "and must not be quoted in the README."
        )
    gaps = golden.coverage_gaps()
    if gaps:
        out.append(f"golden set has no items for strata: {', '.join(gaps)}")
    if thresholds.price_table_version != run.price_table_version:
        out.append(
            f"price table version mismatch: thresholds pin {thresholds.price_table_version}, "
            f"code has {run.price_table_version}"
        )
    if run.overall.get("errors"):
        out.append(
            f"{run.overall['errors']} item(s) raised instead of returning an abstaining "
            "AnswerResult, which violates the QueryEngine contract"
        )
    human = sum(1 for i in golden.items if i.authored_by is AuthoredBy.HUMAN)
    if human == 0:
        out.append("no human-authored items: every metric is model-derived")
    if not any(i.strata is Strata.INJECTION for i in golden.items):
        out.append("no injection items: prompt-injection resistance is unmeasured")
    return out
