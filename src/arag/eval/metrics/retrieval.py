"""Deterministic retrieval metrics — no LLM, no network, no cost.

These are the metrics that carry the CI gate, and that is a deliberate architectural
choice. RAGAS and the LLM judge are valuable but they are slow, stochastic, and priced;
a gate built on them flakes, and a gate that flakes is a gate people learn to ignore.
Recall, MRR and nDCG computed against human-labelled spans are exact, free, and
reproducible, so a regression in retrieval quality can block a PR with no argument about
whether the judge was having an off day.

**Undefined vs zero.** Every function returns ``None`` when the item has no ground-truth
spans (an unanswerable item). It does not return ``0.0``. Averaging a zero into the mean
for an item that has nothing to retrieve would understate retrieval quality in proportion
to how many adversarial items the golden set contains — that is, the more rigorous the
eval set, the worse the numbers would look, which is exactly backwards. The aggregator
skips ``None``s and reports the denominator it actually used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log2

from arag.eval.matching import SpanMatcher
from arag.retrieval.types import DocSpan, RetrievedChunk


def recall_at_k(
    chunks: list[RetrievedChunk],
    truths: tuple[DocSpan, ...],
    k: int,
    matcher: SpanMatcher,
) -> float | None:
    """Fraction of *distinct ground-truth spans* covered by the top-k.

    Distinctness matters: a retriever returning five chunks that all cover the same clause
    has found one of two required clauses, not five. This is the metric that catches
    multi-hop failure, where the system reliably finds the inclusion clause and never the
    waiting-period table.
    """
    if not truths:
        return None
    hits = matcher.hit_truths(chunks[:k], truths)
    return len(hits) / len(truths)


def precision_at_k(
    chunks: list[RetrievedChunk],
    truths: tuple[DocSpan, ...],
    k: int,
    matcher: SpanMatcher,
) -> float | None:
    """Fraction of the passages actually handed to the generator that are relevant.

    Denominator is ``min(k, len(chunks))``, not ``k``. The question this metric answers is
    "how polluted is the context window?", and an unfilled slot pollutes nothing. Using a
    fixed ``k`` would conflate context pollution with under-retrieval, which recall
    already measures.
    """
    if not truths:
        return None
    window = chunks[:k]
    if not window:
        return 0.0
    rel = matcher.relevance_vector(window, truths)
    return sum(rel) / len(window)


def mrr(
    chunks: list[RetrievedChunk],
    truths: tuple[DocSpan, ...],
    matcher: SpanMatcher,
) -> float | None:
    """Reciprocal rank of the first relevant chunk. Sensitive to the reranker's top slot."""
    if not truths:
        return None
    rel = matcher.relevance_vector(chunks, truths)
    for i, r in enumerate(rel):
        if r:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    chunks: list[RetrievedChunk],
    truths: tuple[DocSpan, ...],
    k: int,
    matcher: SpanMatcher,
) -> float | None:
    """Binary-gain nDCG@k with the standard ``1/log2(rank+1)`` discount.

    The ideal ranking places ``min(len(truths), k)`` relevant items first, so nDCG reaches
    1.0 only when every labelled span appears above every distractor. Chosen over plain
    recall as the headline retrieval number because it is the one metric that moves when
    the reranker reorders a fixed candidate set — which is the whole point of having a
    reranker, and therefore the number that must justify its 60ms.
    """
    if not truths:
        return None
    window = chunks[:k]
    rel = matcher.relevance_vector(window, truths)
    dcg = sum(r / log2(i + 2) for i, r in enumerate(rel))
    ideal_n = min(len(truths), k)
    idcg = sum(1.0 / log2(i + 2) for i in range(ideal_n))
    if idcg == 0:
        return None
    return dcg / idcg


def hit_rate(
    chunks: list[RetrievedChunk],
    truths: tuple[DocSpan, ...],
    k: int,
    matcher: SpanMatcher,
) -> float | None:
    """1.0 if any labelled span appears in the top-k. The retrieval-hit-rate logged in
    production (DESIGN §8) uses this same function, so the offline and online definitions
    cannot drift apart.
    """
    if not truths:
        return None
    return 1.0 if matcher.hit_truths(chunks[:k], truths) else 0.0


@dataclass
class RetrievalScores:
    """Per-item retrieval scores. ``None`` means "not applicable to this item"."""

    recall_at_k: float | None = None
    precision_at_k: float | None = None
    mrr: float | None = None
    ndcg_at_k: float | None = None
    hit_rate: float | None = None
    k: int = 10
    n_retrieved: int = 0
    n_truths: int = 0

    def as_dict(self) -> dict[str, float | None]:
        return {
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
            "hit_rate": self.hit_rate,
        }


def score_retrieval(
    chunks: list[RetrievedChunk],
    truths: tuple[DocSpan, ...],
    *,
    k: int = 10,
    matcher: SpanMatcher | None = None,
) -> RetrievalScores:
    m = matcher or SpanMatcher()
    return RetrievalScores(
        recall_at_k=recall_at_k(chunks, truths, k, m),
        precision_at_k=precision_at_k(chunks, truths, k, m),
        mrr=mrr(chunks, truths, m),
        ndcg_at_k=ndcg_at_k(chunks, truths, k, m),
        hit_rate=hit_rate(chunks, truths, k, m),
        k=k,
        n_retrieved=len(chunks),
        n_truths=len(truths),
    )


@dataclass
class Aggregate:
    """Mean over defined values, carrying its own denominator.

    Reporting ``n`` alongside the mean is not decoration. "recall 0.71 (n=54)" and
    "recall 0.71 (n=4)" are different claims, and a metric table that omits ``n`` lets a
    stratum with three items look as authoritative as one with fifty.
    """

    values: list[float] = field(default_factory=list)
    skipped: int = 0

    def add(self, value: float | None) -> None:
        if value is None:
            self.skipped += 1
        else:
            self.values.append(value)

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float | None:
        return sum(self.values) / len(self.values) if self.values else None

    def percentile(self, p: float) -> float | None:
        """Nearest-rank percentile. Used for latency, where p95 is the contract."""
        if not self.values:
            return None
        ordered = sorted(self.values)
        idx = max(0, min(len(ordered) - 1, round(p / 100 * len(ordered) + 0.5) - 1))
        return ordered[idx]
