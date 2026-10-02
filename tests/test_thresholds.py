"""Gate tests.

The gate is a pure function so it can be tested without running an eval. The cases below
are the ways a gate can lie: passing when nothing was measured, passing when a metric
disappeared, or failing to notice a hard failure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arag.eval.thresholds import Gate, Profile, Thresholds, check_gate


class TestDirectionality:
    def test_min_threshold_fails_below(self) -> None:
        profile = Profile(gate=Gate(min_recall_at_k=0.5))
        assert not check_gate({"recall_at_k": 0.4}, profile).passed
        assert check_gate({"recall_at_k": 0.5}, profile).passed

    def test_max_threshold_fails_above(self) -> None:
        profile = Profile(gate=Gate(max_false_abstention_rate=0.3))
        assert not check_gate({"false_abstention_rate": 0.4}, profile).passed
        assert check_gate({"false_abstention_rate": 0.3}, profile).passed

    def test_failure_message_names_metric_and_limit(self) -> None:
        """A CI failure nobody can read is a CI failure nobody acts on."""
        result = check_gate({"recall_at_k": 0.41}, Profile(gate=Gate(min_recall_at_k=0.7)))
        assert "recall@k" in result.failures[0]
        assert "0.41" in result.failures[0]
        assert "0.7" in result.failures[0]


class TestMissingMeasurements:
    def test_missing_metric_fails_a_configured_threshold(self) -> None:
        """A metric that vanished from the pipeline is a regression too.

        Treating absent as passing is the classic way a gate rots: someone renames a
        metric key, the threshold stops matching anything, and the build stays green.
        """
        result = check_gate({}, Profile(gate=Gate(min_recall_at_k=0.5)))
        assert not result.passed
        assert "no measurement produced" in result.failures[0]

    def test_unconfigured_threshold_is_counted_as_skipped(self) -> None:
        """ "0 checks run" must never be indistinguishable from "all checks passed"."""
        result = check_gate({"recall_at_k": 0.9}, Profile(gate=Gate()))
        assert result.passed
        assert result.checks_run == 0
        assert result.checks_skipped > 0
        assert "not configured" in result.summary()


class TestHardFailures:
    def test_supersession_violation_fails_the_gate(self) -> None:
        profile = Profile(gate=Gate(max_supersession_violations=0))
        result = check_gate({"supersession_violations": 1}, profile)
        assert not result.passed
        assert "hard failure" in result.failures[0]

    def test_zero_violations_passes(self) -> None:
        profile = Profile(gate=Gate(max_supersession_violations=0))
        assert check_gate({"supersession_violations": 0}, profile).passed


class TestExactAssertions:
    def test_exact_value_must_match(self) -> None:
        """Used by v0 to assert the harness registers total failure, which a floor
        threshold of ``>= 0.0`` cannot express: every value satisfies it.
        """
        profile = Profile(exact={"recall_at_k": 0.0})
        assert check_gate({"recall_at_k": 0.0}, profile).passed
        assert not check_gate({"recall_at_k": 0.01}, profile).passed

    def test_exact_missing_value_fails(self) -> None:
        assert not check_gate({}, Profile(exact={"recall_at_k": 0.0})).passed


class TestShippedConfig:
    def test_loads(self, thresholds_path: Path) -> None:
        th = Thresholds.load(thresholds_path)
        # Every superseded profile is KEPT, never deleted: v0_floor proved the harness
        # registers total failure, and v1_baseline is the one derived from a broken
        # matcher. Bumping active_profile is the documented, reviewable act.
        assert th.active_profile == "v1r_baseline"
        assert "v0_floor" in th.profiles
        assert "v1_baseline" in th.profiles

    def test_the_superseded_v1_profile_is_kept_and_marked(self, thresholds_path: Path) -> None:
        """v1_baseline stays in the file, and says why it must not be used.

        Deleting it would erase the record of a gate that passed while unable to catch the
        regression it existed for. The numbers are wrong and that is the point of keeping
        them.
        """
        profile = Thresholds.load(thresholds_path).profile("v1_baseline")
        assert profile.gate.min_recall_at_k == 0.30, "the superseded values must not be edited"
        assert "SUPERSEDED" in profile.description
        assert "instance 9" in profile.description

    def test_v1r_gate_is_derived_from_the_corrected_run(self, thresholds_path: Path) -> None:
        """DESIGN section 11: the gate comes from the control-group run, not a guess.

        Re-measured BM25-only baseline after the instance 9 matcher fix, markdown
        serialisation: recall@10 0.400, nDCG@10 0.2349, MRR 0.2067. The gate sits at those
        values with no tolerance - deliberate and brittle, because at n=5 one item flipping
        moves recall by 0.2 and a loose gate hides a real regression.
        """
        gate = Thresholds.load(thresholds_path).profile("v1r_baseline").gate
        assert gate.min_recall_at_k == 0.40
        assert gate.min_ndcg_at_k == 0.23
        assert gate.min_mrr == 0.20

    def test_the_re_derived_floor_catches_the_regression_the_old_one_allowed(
        self, thresholds_path: Path
    ) -> None:
        """The reason the re-freeze was worth doing, asserted rather than described.

        Corrected per-item BM25 recall is h-01 1.0, h-02 0.0, h-03 0.0, h-04 0.5, h-28 0.5
        -> mean 0.400. A regression that destroys h-28's retrieval entirely takes the mean
        to 0.300, which CLEARED the old 0.30 floor. The gate derived from a mismeasurement
        could not catch a regression in the numbers it was built to guard - instance 9's
        sharpest consequence.
        """
        per_item = [1.0, 0.0, 0.0, 0.5, 0.5]
        assert sum(per_item) / len(per_item) == pytest.approx(0.400)

        regressed = [1.0, 0.0, 0.0, 0.5, 0.0]
        regressed_recall = sum(regressed) / len(regressed)
        assert regressed_recall == pytest.approx(0.300)

        th = Thresholds.load(thresholds_path)
        old_floor = th.profile("v1_baseline").gate.min_recall_at_k
        new_floor = th.profile("v1r_baseline").gate.min_recall_at_k
        assert old_floor is not None and new_floor is not None
        assert regressed_recall >= old_floor, "the old gate would have passed this regression"
        assert regressed_recall < new_floor, "the re-derived gate must fail it"

    def test_v1_leaves_unmeasurable_thresholds_unset(self, thresholds_path: Path) -> None:
        """A threshold on a metric that cannot be measured fails the gate for the wrong
        reason. v1 has no generator, so no citations; and the golden set has no
        unanswerable or injection items yet.
        """
        gate = Thresholds.load(thresholds_path).profile("v1_baseline").gate
        assert gate.min_abstain_recall is None
        assert gate.min_injection_resistance is None
        assert gate.max_ungrounded_citation_rate is None

    def test_v1_tolerates_total_abstention(self, thresholds_path: Path) -> None:
        """RetrievalOnlyEngine abstains on every item by design - there is no generator
        and no judge whose agreement has been measured. 1.0 is correct for v1 and tightens
        at v2, which is why it is a per-profile value rather than a global one.
        """
        gate = Thresholds.load(thresholds_path).profile("v1_baseline").gate
        assert gate.max_false_abstention_rate == 1.00

    def test_unknown_profile_raises_and_lists_options(self, thresholds_path: Path) -> None:
        th = Thresholds.load(thresholds_path)
        with pytest.raises(KeyError, match="v0_floor"):
            th.profile("does-not-exist")

    def test_cost_ratio_matches_the_design_assumption(self, thresholds_path: Path) -> None:
        """DESIGN §13 assumption 4. If this changes, the v4 threshold sweep changes with
        it, so the number lives in one place and is asserted here.
        """
        th = Thresholds.load(thresholds_path)
        ratio = th.calibration.wrong_answer_cost / th.calibration.abstention_cost
        assert ratio == 50.0

    def test_page_tolerance_is_zero(self, thresholds_path: Path) -> None:
        """Widening this inflates recall by crediting neighbouring-page chunks, which is
        the chunking defect the metric exists to expose.
        """
        assert Thresholds.load(thresholds_path).retrieval.page_tolerance == 0

    def test_judge_kappa_floor_is_configured(self, thresholds_path: Path) -> None:
        assert Thresholds.load(thresholds_path).judge.min_kappa == 0.60

    def test_v0_profile_asserts_exact_zeros(self, thresholds_path: Path) -> None:
        exact = Thresholds.load(thresholds_path).profile("v0_floor").exact
        assert exact == {"recall_at_k": 0.0, "ndcg_at_k": 0.0, "mrr": 0.0, "hit_rate": 0.0}

    def test_unknown_key_in_config_is_rejected(self, tmp_path: Path) -> None:
        """extra='forbid' means a typo'd threshold name fails loudly instead of being
        silently ignored while the author believes the gate is enforcing it.
        """
        bad = tmp_path / "t.yaml"
        bad.write_text(
            "price_table_version: x\nactive_profile: p\n"
            "profiles:\n  p:\n    gate:\n      min_recal_at_k: 0.5\n",
            encoding="utf-8",
        )
        with pytest.raises(Exception, match="min_recal_at_k"):
            Thresholds.load(bad)
