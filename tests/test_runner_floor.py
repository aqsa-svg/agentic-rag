"""The v0 exit criterion, as a test.

DESIGN §11 says slice v0 is done when the suite runs against a stub that returns nothing
and every retrieval metric reads exactly 0.0. That is not a formality. If a metric reads
non-zero against a retriever that retrieved nothing, the metric is broken — and the only
cheap moment to discover that is now, before a real pipeline has spent three slices
"improving" a number that was never measuring anything.

The second thing these tests pin down is the abstain-metric pair. ``NullEngine`` scores
1.0 on abstain recall by refusing everything. Read alone, that looks like perfect
calibration. Read next to a false-abstention rate of 1.0, it is obviously a system that
does nothing. Two stub engines failing in opposite directions bracket the metric space
from below and prove no single number can be gamed into looking good.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arag.agent.stub import EchoEngine, NullEngine
from arag.eval.runner import run_eval
from arag.eval.schema import GoldenSet
from arag.eval.thresholds import Thresholds


@pytest.fixture
def golden(golden_path: Path) -> GoldenSet:
    return GoldenSet.load(golden_path)


@pytest.fixture
def thresholds(thresholds_path: Path) -> Thresholds:
    return Thresholds.load(thresholds_path)


class TestNullEngineFloor:
    async def test_every_retrieval_metric_is_exactly_zero(
        self, golden: GoldenSet, thresholds: Thresholds
    ) -> None:
        run = await run_eval(golden, NullEngine(), thresholds, profile_name="v0_floor")
        for key in ("recall_at_k", "precision_at_k", "mrr", "ndcg_at_k", "hit_rate"):
            assert run.overall[key] == 0.0, f"{key} should be exactly 0.0 against NullEngine"

    async def test_items_without_ground_truth_are_skipped_not_zeroed(
        self, golden: GoldenSet, thresholds: Thresholds
    ) -> None:
        """The unanswerable and injection items carry no spans, so retrieval is undefined
        for them. They must appear in ``n_retrieval_skipped``, not drag the mean down.
        """
        run = await run_eval(golden, NullEngine(), thresholds, profile_name="v0_floor")
        assert run.overall["n_retrieval_skipped"] > 0
        assert run.overall["n_scored"] + run.overall["n_retrieval_skipped"] == len(golden.items)

    async def test_abstain_pair_exposes_the_degenerate_strategy(
        self, golden: GoldenSet, thresholds: Thresholds
    ) -> None:
        run = await run_eval(golden, NullEngine(), thresholds, profile_name="v0_floor")
        assert run.overall["abstain_recall"] == 1.0
        assert run.overall["false_abstention_rate"] == 1.0

    async def test_no_hard_failures_from_a_silent_engine(
        self, golden: GoldenSet, thresholds: Thresholds
    ) -> None:
        """Refusing to answer is never a supersession or injection failure. This keeps the
        hard-failure counters meaningful: they must only fire on things actually emitted.
        """
        run = await run_eval(golden, NullEngine(), thresholds, profile_name="v0_floor")
        assert run.overall["supersession_violations"] == 0
        assert run.overall["injection_resistance"] == 1.0

    async def test_no_item_raised(self, golden: GoldenSet, thresholds: Thresholds) -> None:
        run = await run_eval(golden, NullEngine(), thresholds, profile_name="v0_floor")
        assert run.overall["errors"] == 0
        assert all(r.ok for r in run.items)

    async def test_gate_passes_on_the_floor_profile(
        self, golden: GoldenSet, thresholds: Thresholds
    ) -> None:
        """The v0 gate must go green, and it must have actually run checks to do so."""
        run = await run_eval(golden, NullEngine(), thresholds, profile_name="v0_floor")
        assert run.gate is not None
        assert run.gate.passed, run.gate.summary()
        assert run.gate.checks_run >= 8, "a gate that ran no checks is not a passing gate"

    async def test_gate_would_fail_a_real_profile(
        self, golden: GoldenSet, thresholds: Thresholds
    ) -> None:
        """Sanity check in the other direction: the floor profile passes because its
        thresholds are zero, not because the gate never fails anything.

        NullEngine retrieves nothing, so it must fail the v1_baseline profile on the
        retrieval floors measured from the real BM25 run. Note it does NOT fail on
        false abstention any more - v1 tolerates total abstention by design, because v1
        has no generator - so the assertion names the retrieval metrics instead.
        """
        run = await run_eval(golden, NullEngine(), thresholds, profile_name="v1_baseline")
        assert run.gate is not None
        assert not run.gate.passed
        assert any("recall@k" in f for f in run.gate.failures)


class TestEchoEngineOppositeFailure:
    async def test_answers_everything_without_support(
        self, golden: GoldenSet, thresholds: Thresholds
    ) -> None:
        run = await run_eval(golden, EchoEngine(), thresholds, profile_name="v0_floor")
        assert run.overall["false_abstention_rate"] == 0.0
        assert run.overall["abstain_recall"] == 0.0
        assert run.overall["unsupported_answer_rate"] == 1.0

    async def test_confident_nonsense_costs_more_than_silence(
        self, golden: GoldenSet, thresholds: Thresholds
    ) -> None:
        """The 50:1 cost ratio in action.

        EchoEngine answers everything and NullEngine answers nothing. Under DESIGN §13
        assumption 4 the confident-nonsense strategy must carry the higher expected cost,
        otherwise the calibration objective is pointing the wrong way and every threshold
        tuned against it in v4 would be wrong.
        """
        null_run = await run_eval(golden, NullEngine(), thresholds, profile_name="v0_floor")
        echo_run = await run_eval(golden, EchoEngine(), thresholds, profile_name="v0_floor")
        assert echo_run.overall["expected_cost"] > null_run.overall["expected_cost"]


class TestRunMetadata:
    async def test_records_what_is_needed_to_reproduce(
        self, golden: GoldenSet, thresholds: Thresholds
    ) -> None:
        """Without these fields a row in EVAL_LOG.md is an unfalsifiable claim."""
        run = await run_eval(golden, NullEngine(), thresholds, profile_name="v0_floor")
        assert run.engine == "null"
        assert run.profile == "v0_floor"
        assert run.git_sha
        assert run.price_table_version
        assert run.golden_source.endswith("golden_seed.jsonl")
        assert run.n_items == len(golden.items)

    async def test_warns_about_unverified_labels(
        self, golden: GoldenSet, thresholds: Thresholds
    ) -> None:
        """The shipped golden set is all seed items. If that ever stops producing a
        warning, unverified guesses have started masquerading as evidence.
        """
        run = await run_eval(golden, NullEngine(), thresholds, profile_name="v0_floor")
        assert any("seed_unverified" in w for w in run.warnings)
        assert any("must not be quoted" in w for w in run.warnings)

    async def test_slices_by_stratum_and_provenance(
        self, golden: GoldenSet, thresholds: Thresholds
    ) -> None:
        run = await run_eval(golden, NullEngine(), thresholds, profile_name="v0_floor")
        assert "multihop_temporal" in run.by_strata
        assert "injection" in run.by_strata
        assert "seed_unverified" in run.by_provenance
        assert sum(m["n_items"] for m in run.by_strata.values()) == len(golden.items)


class TestConcurrency:
    async def test_parallel_run_produces_identical_metrics(
        self, golden: GoldenSet, thresholds: Thresholds
    ) -> None:
        """Concurrency is a throughput knob, not a semantic one. If raising it changed the
        numbers, every recorded run would be incomparable to every other.
        """
        serial = await run_eval(
            golden, NullEngine(), thresholds, profile_name="v0_floor", concurrency=1
        )
        parallel = await run_eval(
            golden, NullEngine(), thresholds, profile_name="v0_floor", concurrency=8
        )
        for key in ("recall_at_k", "ndcg_at_k", "abstain_recall", "false_abstention_rate"):
            assert serial.overall[key] == parallel.overall[key]
