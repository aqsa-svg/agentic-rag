"""Two-sided expectation assertion tests.

`expected_to_fail` is only useful if it is checked in **both** directions. A known-failing
item that silently starts passing is the fix landing — the single most valuable event in
the log — and a scheme with only pass/fail would file it under "pass" where nobody would
ever see it. These tests pin all four categories, and specifically that FIXED is not a
regression and FAILED_AS_EXPECTED is not either.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from arag.agent.stub import EchoEngine, NullEngine
from arag.eval.report import render_console
from arag.eval.runner import Expectation, item_satisfied, run_eval
from arag.eval.schema import GoldenItem, GoldenSet
from arag.eval.thresholds import Thresholds

KNOWN_REASON = "Known defect: caption-only table header makes the row unretrievable."


def item(iid: str, behaviour: str, *, known_fail: bool = False) -> GoldenItem:
    base: dict[str, object] = {
        "id": iid,
        "question": "Is this benefit covered under the policy?",
        "strata": "unanswerable" if behaviour == "abstain" else "flat_lookup",
        "authored_by": "human",
        "expected_behaviour": behaviour,
        "reference_answer": "Some reference answer.",
    }
    if behaviour == "answer":
        base["ground_truth_spans"] = [{"document_id": "d1", "page": 1, "clause_id": "3.1"}]
    if known_fail:
        base["expected_to_fail"] = True
        base["expected_failure_reason"] = KNOWN_REASON
    return GoldenItem.model_validate(base)


@pytest.fixture
def thresholds(thresholds_path: Path) -> Thresholds:
    return Thresholds.load(thresholds_path)


def run(golden: GoldenSet, engine: object, thresholds: Thresholds):  # type: ignore[no-untyped-def]
    return asyncio.run(run_eval(golden, engine, thresholds, profile_name="v0_floor"))  # type: ignore[arg-type]


class TestFourCategories:
    """NullEngine abstains on everything, so abstain items pass and answer items fail.
    That gives all four categories from one engine.
    """

    @pytest.fixture
    def result(self, thresholds: Thresholds):  # type: ignore[no-untyped-def]
        golden = GoldenSet(
            items=(
                item("ok-abstain", "abstain"),
                item("ordinary-answer", "answer"),
                item("known-fail-answer", "answer", known_fail=True),
                item("known-fail-abstain", "abstain", known_fail=True),
            )
        )
        return run(golden, NullEngine(), thresholds)

    def test_all_four_are_counted_separately(self, result) -> None:  # type: ignore[no-untyped-def]
        assert result.overall["expectations"] == {
            "passed": 1,
            "failed": 1,
            "failed_as_expected": 1,
            "fixed": 1,
        }

    def test_a_known_failure_that_passes_is_reported_as_fixed(self, result) -> None:  # type: ignore[no-untyped-def]
        """The signal the whole feature exists for."""
        assert result.overall["fixed_ids"] == ["known-fail-abstain"]

    def test_an_unexpected_failure_is_reported_separately(self, result) -> None:  # type: ignore[no-untyped-def]
        assert result.overall["regressed_ids"] == ["ordinary-answer"]

    def test_a_still_failing_known_defect_is_not_a_regression(self, result) -> None:  # type: ignore[no-untyped-def]
        """It is a recorded, open defect. Filing it as a regression would make every run
        look like it broke something and train people to ignore the count.
        """
        assert result.overall["still_failing_ids"] == ["known-fail-answer"]
        assert "known-fail-answer" not in result.overall["regressed_ids"]

    def test_per_item_expectation_is_attached(self, result) -> None:  # type: ignore[no-untyped-def]
        got = {r.item.id: r.expectation for r in result.items}
        assert got["ok-abstain"] is Expectation.PASSED
        assert got["ordinary-answer"] is Expectation.FAILED
        assert got["known-fail-answer"] is Expectation.FAILED_AS_EXPECTED
        assert got["known-fail-abstain"] is Expectation.FIXED


class TestSemantics:
    def test_only_an_unexpected_failure_is_a_regression(self) -> None:
        assert Expectation.FAILED.is_regression
        assert not Expectation.FAILED_AS_EXPECTED.is_regression
        assert not Expectation.FIXED.is_regression
        assert not Expectation.PASSED.is_regression

    def test_both_surprises_need_attention(self) -> None:
        """An unexpected failure AND a fixed known-failure both need a human: one is a
        regression, the other means a label is now stale.
        """
        assert Expectation.FAILED.needs_attention
        assert Expectation.FIXED.needs_attention
        assert not Expectation.PASSED.needs_attention
        assert not Expectation.FAILED_AS_EXPECTED.needs_attention


class TestItemSatisfied:
    def test_hard_behavioural_failure_overrides_everything(self, thresholds: Thresholds) -> None:
        """Citing a superseded document, following an injection, or hallucinating a
        citation are not gradations - they fail the item regardless of retrieval.
        """
        from arag.agent.types import AnswerResult
        from arag.eval.metrics.behavioural import BehaviourScores
        from arag.eval.metrics.retrieval import RetrievalScores

        result = AnswerResult(question="q", answer="a", abstained=False)
        perfect = RetrievalScores(hit_rate=1.0)
        bad = BehaviourScores(supersession_violations=("star-comprehensive-2021",))
        assert bad.hard_failure
        assert not item_satisfied(item("x", "answer"), result, perfect, bad)

    def test_abstaining_fails_an_answer_item(self, thresholds: Thresholds) -> None:
        from arag.agent.types import AnswerResult
        from arag.eval.metrics.behavioural import BehaviourScores
        from arag.eval.metrics.retrieval import RetrievalScores

        abstained = AnswerResult(question="q", abstained=True)
        assert not item_satisfied(
            item("x", "answer"), abstained, RetrievalScores(hit_rate=1.0), BehaviourScores()
        )

    def test_answer_item_needs_a_retrieved_span(self, thresholds: Thresholds) -> None:
        """An answer with no labelled span retrieved is not satisfied, however fluent."""
        from arag.agent.types import AnswerResult
        from arag.eval.metrics.behavioural import BehaviourScores
        from arag.eval.metrics.retrieval import RetrievalScores

        answered = AnswerResult(question="q", answer="a", abstained=False)
        assert not item_satisfied(
            item("x", "answer"), answered, RetrievalScores(hit_rate=0.0), BehaviourScores()
        )
        assert item_satisfied(
            item("x", "answer"), answered, RetrievalScores(hit_rate=1.0), BehaviourScores()
        )


class TestReporting:
    def test_fixed_items_are_named_and_flagged_in_the_console(self, thresholds: Thresholds) -> None:
        """Buried good news is lost good news. The FIXED line must name the ids and say
        what to do about them.
        """
        golden = GoldenSet(items=(item("known-fail-abstain", "abstain", known_fail=True),))
        out = render_console(run(golden, NullEngine(), thresholds))
        assert "FIXED" in out
        assert "known-fail-abstain" in out
        assert "Update expected_to_fail" in out

    def test_expectation_block_is_present_even_when_unremarkable(
        self, thresholds: Thresholds
    ) -> None:
        golden = GoldenSet(items=(item("ok-abstain", "abstain"),))
        out = render_console(run(golden, NullEngine(), thresholds))
        assert "expectation" in out
        assert "passed 1" in out

    def test_echo_engine_flips_which_items_are_satisfied(self, thresholds: Thresholds) -> None:
        """Sanity check that the verdict tracks the engine rather than the label.

        EchoEngine never abstains, so the abstain item now fails - and since it was a
        known-failure, it stays FAILED_AS_EXPECTED rather than flipping to FIXED.
        """
        golden = GoldenSet(items=(item("known-fail-abstain", "abstain", known_fail=True),))
        out = run(golden, EchoEngine(), thresholds)
        assert out.overall["fixed_ids"] == []
        assert out.overall["still_failing_ids"] == ["known-fail-abstain"]
