"""Fast-subset sampling tests.

The fast subset is what the PR gate actually runs, so its composition decides what the
gate can see. ``head -n`` would give 20 flat lookups and a gate blind to exactly the
multi-hop and adversarial regressions it exists to catch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arag.eval.sampling import stratified_subset
from arag.eval.schema import GoldenSet, Strata


@pytest.fixture
def golden(golden_path: Path) -> GoldenSet:
    return GoldenSet.load(golden_path)


class TestComposition:
    def test_adversarial_strata_are_always_present(self, golden: GoldenSet) -> None:
        """A gate that cannot see injection or unanswerable regressions is not a safety
        gate. These three get first claim on the item budget.
        """
        subset = stratified_subset(golden, 6)
        present = {i.strata for i in subset.items}
        assert Strata.UNANSWERABLE in present
        assert Strata.CONTRADICTORY in present
        assert Strata.INJECTION in present

    def test_spreads_across_strata_rather_than_taking_the_top(self, golden: GoldenSet) -> None:
        subset = stratified_subset(golden, 10)
        assert len({i.strata for i in subset.items}) >= 6

    def test_respects_the_requested_size(self, golden: GoldenSet) -> None:
        assert len(stratified_subset(golden, 8).items) == 8

    def test_returns_everything_when_n_exceeds_the_set(self, golden: GoldenSet) -> None:
        """Must terminate rather than loop forever looking for items that do not exist."""
        subset = stratified_subset(golden, 10_000)
        assert len(subset.items) == len(golden.items)

    def test_rejects_a_non_positive_size(self, golden: GoldenSet) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            stratified_subset(golden, 0)


class TestDeterminism:
    def test_same_seed_same_items(self, golden: GoldenSet) -> None:
        """Metric noise from a resampled subset is indistinguishable from a regression,
        which would make the gate untrustworthy in exactly the moment it matters.
        """
        a = stratified_subset(golden, 10)
        b = stratified_subset(golden, 10)
        assert [i.id for i in a.items] == [i.id for i in b.items]

    def test_different_seed_can_differ(self, golden: GoldenSet) -> None:
        a = stratified_subset(golden, 10, seed=1)
        b = stratified_subset(golden, 10, seed=2)
        assert {i.id for i in a.items} <= {i.id for i in golden.items}
        assert {i.id for i in b.items} <= {i.id for i in golden.items}

    def test_result_is_a_valid_golden_set(self, golden: GoldenSet) -> None:
        subset = stratified_subset(golden, 10)
        assert len({i.id for i in subset.items}) == len(subset.items)
        assert subset.source == golden.source
