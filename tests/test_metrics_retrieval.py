"""Retrieval metric tests with hand-computed expected values.

Every expected number here was worked out on paper and the arithmetic is written into the
test. This is the point of the file: if these metrics are subtly wrong, every number this
project ever publishes is wrong, and asserting against "whatever the implementation
returns today" would lock the bug in rather than catch it.
"""

from __future__ import annotations

from math import log2

import pytest

from arag.eval.matching import SpanMatcher
from arag.eval.metrics.retrieval import (
    Aggregate,
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    score_retrieval,
)
from tests.conftest import chunk, truth

M = SpanMatcher()


class TestNdcg:
    def test_hand_computed_value(self) -> None:
        """Ranking [X, A, Y, B, Z] against truths {A, B}, k=5.

        relevance   = [0, 1, 0, 1, 0]
        DCG         = 1/log2(3) + 1/log2(5)
                    = 0.6309297535714574 + 0.43067655807339306
                    = 1.0616063116448505
        ideal (2 relevant items first)
        IDCG        = 1/log2(2) + 1/log2(3)
                    = 1.0 + 0.6309297535714574
                    = 1.6309297535714574
        nDCG        = 1.0616063116448505 / 1.6309297535714574
                    = 0.650921...
        """
        chunks = [
            chunk("d1", "9.9"),
            chunk("d1", "4.2"),
            chunk("d1", "8.8"),
            chunk("d1", "2.1"),
            chunk("d1", "7.7"),
        ]
        truths = (truth("d1", "4.2"), truth("d1", "2.1"))

        expected_dcg = 1 / log2(3) + 1 / log2(5)
        expected_idcg = 1 / log2(2) + 1 / log2(3)
        expected = expected_dcg / expected_idcg

        assert expected == pytest.approx(0.650921, abs=1e-6)
        assert ndcg_at_k(chunks, truths, 5, M) == pytest.approx(expected)

    def test_perfect_ranking_is_one(self) -> None:
        chunks = [chunk("d1", "4.2"), chunk("d1", "2.1"), chunk("d1", "9.9")]
        truths = (truth("d1", "4.2"), truth("d1", "2.1"))
        assert ndcg_at_k(chunks, truths, 10, M) == pytest.approx(1.0)

    def test_ideal_is_capped_at_k(self) -> None:
        """With 3 truths but k=1, finding one at rank 1 is a perfect result *for k=1*.

        IDCG must be computed over min(len(truths), k) ideal positions. Computing it over
        all 3 truths would cap nDCG@1 at ~0.55 and make the metric impossible to reach,
        silently penalising a retriever for a k the caller chose.
        """
        chunks = [chunk("d1", "4.2")]
        truths = (truth("d1", "4.2"), truth("d1", "2.1"), truth("d1", "3.3"))
        assert ndcg_at_k(chunks, truths, 1, M) == pytest.approx(1.0)

    def test_reranking_moves_the_number(self) -> None:
        """The whole justification for a 60ms reranker is that it moves this metric."""
        truths = (truth("d1", "4.2"),)
        bad = [chunk("d1", "9.9"), chunk("d1", "8.8"), chunk("d1", "4.2")]
        good = [chunk("d1", "4.2"), chunk("d1", "9.9"), chunk("d1", "8.8")]
        assert ndcg_at_k(good, truths, 10, M) > ndcg_at_k(bad, truths, 10, M)  # type: ignore[operator]
        # ...while recall is blind to the reordering, which is why nDCG is the headline.
        assert recall_at_k(good, truths, 10, M) == recall_at_k(bad, truths, 10, M)


class TestRecall:
    def test_counts_distinct_truths_not_relevant_chunks(self) -> None:
        """Three chunks all covering clause 4.2 is ONE of two truths found, not three.

        Without this, a retriever that returns near-duplicate chunks would outscore a
        precise one, and duplicate-heavy chunking would look like a quality improvement.
        """
        chunks = [
            chunk("d1", "4.2", cid="a"),
            chunk("d1", "4.2", cid="b"),
            chunk("d1", "4.2", cid="c"),
        ]
        truths = (truth("d1", "4.2"), truth("d1", "2.1"))
        assert recall_at_k(chunks, truths, 10, M) == pytest.approx(0.5)

    def test_respects_k(self) -> None:
        chunks = [chunk("d1", "9.9"), chunk("d1", "8.8"), chunk("d1", "4.2")]
        truths = (truth("d1", "4.2"),)
        assert recall_at_k(chunks, truths, 2, M) == 0.0
        assert recall_at_k(chunks, truths, 3, M) == 1.0

    def test_multihop_partial_credit(self) -> None:
        """The multi-hop failure signature: inclusion clause found, waiting period missed."""
        chunks = [chunk("star", "3.14")]
        truths = (truth("star", "3.14"), truth("star", "4.2.b"))
        assert recall_at_k(chunks, truths, 10, M) == pytest.approx(0.5)


class TestPrecision:
    def test_denominator_is_what_was_actually_returned(self) -> None:
        """2 relevant out of 3 returned = 0.667, even though k=10.

        Using k as the denominator would give 0.2 and conflate under-retrieval with
        context pollution. Recall already measures under-retrieval.
        """
        chunks = [chunk("d1", "4.2"), chunk("d1", "2.1"), chunk("d1", "9.9")]
        truths = (truth("d1", "4.2"), truth("d1", "2.1"))
        assert precision_at_k(chunks, truths, 10, M) == pytest.approx(2 / 3)

    def test_empty_retrieval_is_zero_not_undefined(self) -> None:
        assert precision_at_k([], (truth("d1", "4.2"),), 10, M) == 0.0


class TestMrr:
    def test_reciprocal_of_first_hit(self) -> None:
        chunks = [chunk("d1", "9.9"), chunk("d1", "4.2"), chunk("d1", "2.1")]
        truths = (truth("d1", "4.2"), truth("d1", "2.1"))
        assert mrr(chunks, truths, M) == pytest.approx(0.5)

    def test_no_hit_is_zero(self) -> None:
        assert mrr([chunk("d1", "9.9")], (truth("d1", "4.2"),), M) == 0.0


class TestUndefinedVersusZero:
    """The single most consequential convention in the harness.

    An unanswerable item has no ground truth, so retrieval quality is not merely zero on
    it — it is undefined. Returning 0.0 would mean that every adversarial item added to
    the golden set drags the reported recall down, i.e. a *more* rigorous eval set would
    produce *worse* numbers. That inverts the incentive the whole project rests on.
    """

    @pytest.mark.parametrize(
        "fn",
        [
            lambda c, t: recall_at_k(c, t, 10, M),
            lambda c, t: precision_at_k(c, t, 10, M),
            lambda c, t: ndcg_at_k(c, t, 10, M),
            lambda c, t: hit_rate(c, t, 10, M),
            lambda c, t: mrr(c, t, M),
        ],
    )
    def test_no_truths_returns_none(self, fn) -> None:  # type: ignore[no-untyped-def]
        assert fn([chunk("d1", "4.2")], ()) is None

    def test_aggregate_skips_none_and_reports_its_denominator(self) -> None:
        agg = Aggregate()
        for value in (1.0, 0.0, None, 1.0, None):
            agg.add(value)
        assert agg.n == 3
        assert agg.skipped == 2
        assert agg.mean == pytest.approx(2 / 3)

    def test_empty_aggregate_mean_is_none_not_zero(self) -> None:
        assert Aggregate().mean is None


class TestScoreRetrieval:
    def test_bundles_consistent_values(self) -> None:
        chunks = [chunk("d1", "9.9"), chunk("d1", "4.2")]
        truths = (truth("d1", "4.2"),)
        s = score_retrieval(chunks, truths, k=10)
        assert s.recall_at_k == 1.0
        assert s.mrr == pytest.approx(0.5)
        assert s.hit_rate == 1.0
        assert s.n_retrieved == 2
        assert s.n_truths == 1

    def test_empty_retrieval_floors_every_metric(self) -> None:
        """The v0 exit criterion, at the level of a single item."""
        s = score_retrieval([], (truth("d1", "4.2"),), k=10)
        assert s.as_dict() == {
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
            "hit_rate": 0.0,
        }


class TestPercentile:
    def test_p95_of_twenty_values_is_the_top_one(self) -> None:
        agg = Aggregate()
        for i in range(1, 21):
            agg.add(float(i))
        assert agg.percentile(95) == 20.0

    def test_p50_is_a_middle_value(self) -> None:
        agg = Aggregate()
        for value in (10.0, 20.0, 30.0):
            agg.add(value)
        assert agg.percentile(50) == 20.0
