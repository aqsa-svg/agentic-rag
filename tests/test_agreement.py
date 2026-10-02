"""Cohen's kappa tests, with the arithmetic written out.

This number decides whether the free local judge's scores count as evidence, so it gets
the same hand-computed treatment as the retrieval metrics.
"""

from __future__ import annotations

import pytest

from arag.eval.metrics.agreement import KAPPA_TRUSTWORTHY, cohens_kappa


class TestKappa:
    def test_hand_computed_value(self) -> None:
        """human = [1,1,0,0,1], judge = [1,0,0,0,1]

        agreement at positions 1,3,4,5 -> po = 4/5 = 0.8
        human marginals: p(1)=3/5=0.6, p(0)=2/5=0.4
        judge marginals: p(1)=2/5=0.4, p(0)=3/5=0.6
        pe = 0.6*0.4 + 0.4*0.6 = 0.24 + 0.24 = 0.48
        kappa = (0.8 - 0.48) / (1 - 0.48) = 0.32 / 0.52 = 0.615384...
        """
        report = cohens_kappa(["1", "1", "0", "0", "1"], ["1", "0", "0", "0", "1"])
        assert report.observed_agreement == pytest.approx(0.8)
        assert report.expected_agreement == pytest.approx(0.48)
        assert report.kappa == pytest.approx(0.32 / 0.52)
        assert report.kappa == pytest.approx(0.615385, abs=1e-6)

    def test_perfect_agreement_with_both_labels_used(self) -> None:
        report = cohens_kappa(["1", "0", "1", "0"], ["1", "0", "1", "0"])
        assert report.kappa == pytest.approx(1.0)
        assert report.band == "strong"

    def test_chance_agreement_is_penalised(self) -> None:
        """Why kappa and not raw agreement.

        A judge that says "correct" every time agrees with a human 80% of the time when
        80% of answers are correct — and carries zero information. Raw agreement rewards
        that; kappa reports 0.0 and the gate rejects it.
        """
        human = ["1"] * 8 + ["0"] * 2
        lazy_judge = ["1"] * 10
        report = cohens_kappa(human, lazy_judge)
        assert report.observed_agreement == pytest.approx(0.8)
        assert report.kappa == pytest.approx(0.0)
        assert not report.trustworthy

    def test_degenerate_single_label_sample_returns_zero_not_one(self) -> None:
        """Both raters used one identical label: pe == 1 and kappa is 0/0.

        Returning 1.0 would claim perfect reliability from a sample carrying no
        information. Returning 0.0 plus visible per-label counts makes the sampling bug
        obvious instead.
        """
        report = cohens_kappa(["1"] * 5, ["1"] * 5)
        assert report.expected_agreement == pytest.approx(1.0)
        assert report.kappa == 0.0
        assert report.per_label_counts == {"1": 5}

    def test_systematic_disagreement_is_negative(self) -> None:
        report = cohens_kappa(["1", "1", "0", "0"], ["0", "0", "1", "1"])
        assert report.kappa < 0

    def test_length_mismatch_raises_rather_than_truncating(self) -> None:
        """A silent zip() truncation would misalign every pair after the missing label."""
        with pytest.raises(ValueError, match="lengths differ"):
            cohens_kappa(["1", "0", "1"], ["1", "0"])

    def test_empty_sample_raises(self) -> None:
        with pytest.raises(ValueError, match="empty sample"):
            cohens_kappa([], [])


class TestBands:
    @pytest.mark.parametrize(
        ("kappa_pair", "expected_band"),
        [
            ((["1", "0", "1", "0"], ["1", "0", "1", "0"]), "strong"),
            ((["1", "1", "0", "0", "1"], ["1", "0", "0", "0", "1"]), "moderate"),
            ((["1"] * 8 + ["0"] * 2, ["1"] * 10), "insufficient"),
        ],
    )
    def test_band_thresholds(self, kappa_pair, expected_band: str) -> None:  # type: ignore[no-untyped-def]
        human, judge = kappa_pair
        assert cohens_kappa(human, judge).band == expected_band

    def test_trustworthy_line_matches_the_documented_threshold(self) -> None:
        assert KAPPA_TRUSTWORTHY == 0.60


class TestDisagreementReport:
    def test_names_the_items_that_disagreed(self) -> None:
        """The disagreement list is the actionable output: it tells you which answers to
        re-read when kappa comes back low.
        """
        report = cohens_kappa(
            ["1", "1", "0"], ["1", "0", "0"], item_ids=["g-001", "g-002", "g-003"]
        )
        assert report.disagreements == (("g-002", "1", "0"),)
