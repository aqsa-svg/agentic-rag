"""Judge validity — Cohen's kappa between the local judge and human labels.

This module exists because of a specific risk accepted in DESIGN §5.8. Running the judge
on local hardware is free and unlimited, which is why it was chosen; the cost is that a
~30B model at 4-bit is a weaker judge than a frontier model, and **a weak judge fails in
the flattering direction** — it tends to accept answers a careful human would reject. A
free judge that is never validated produces metrics that look like evidence and are not.

So the harness refuses to let that stay unmeasured. ``arag-eval judge-agreement`` scores a
stratified hand-labelled sample and reports kappa. Raw agreement is deliberately *not*
the headline number: if 85% of answers are correct, a judge that says "correct" every
time achieves 85% agreement while carrying zero information. Kappa corrects for exactly
that chance agreement, which is why it is the gate.

Interpretation used in this project (Landis & Koch bands):

* kappa >= 0.80  strong — publish the metrics as-is
* 0.60-0.79      moderate — publish with the kappa stated alongside
* < 0.60         insufficient — the metrics are not trustworthy; switch to a paid judge

The 0.60 line is enforced in ``thresholds.yaml``, not left to judgement.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

KAPPA_TRUSTWORTHY = 0.60
KAPPA_STRONG = 0.80


@dataclass(frozen=True)
class AgreementReport:
    n: int
    observed_agreement: float
    expected_agreement: float
    kappa: float
    per_label_counts: dict[str, int]
    disagreements: tuple[tuple[str, str, str], ...] = ()  # (item_id, human, judge)

    @property
    def band(self) -> str:
        if self.kappa >= KAPPA_STRONG:
            return "strong"
        if self.kappa >= KAPPA_TRUSTWORTHY:
            return "moderate"
        return "insufficient"

    @property
    def trustworthy(self) -> bool:
        return self.kappa >= KAPPA_TRUSTWORTHY

    def summary(self) -> str:
        return (
            f"kappa={self.kappa:.3f} ({self.band}, n={self.n}), "
            f"raw agreement={self.observed_agreement:.3f}, "
            f"chance agreement={self.expected_agreement:.3f}"
        )


def cohens_kappa(
    human: Sequence[str],
    judge: Sequence[str],
    *,
    item_ids: Sequence[str] | None = None,
) -> AgreementReport:
    """Cohen's kappa for two raters over the same items.

    ``kappa = (po - pe) / (1 - pe)`` where ``po`` is observed agreement and ``pe`` is the
    agreement expected if both raters assigned labels independently at their own observed
    marginal rates.

    Edge case: when ``pe == 1`` (both raters used exactly one label, and the same one)
    kappa is mathematically undefined — 0/0. Returning 1.0 would claim perfect reliability
    from a sample carrying no information. This returns 0.0 and the caller sees ``n`` and
    ``per_label_counts``, which make the degenerate sample obvious. A stratified sample
    that hits only one label is a sampling bug to fix, not a result to report.
    """
    if len(human) != len(judge):
        raise ValueError(f"rater lengths differ: {len(human)} vs {len(judge)}")
        # A silent zip() truncation here would misalign every pair after the first
        # missing label and quietly destroy the number.
    if not human:
        raise ValueError("cannot compute kappa on an empty sample")

    n = len(human)
    agree = sum(1 for h, j in zip(human, judge, strict=True) if h == j)
    po = agree / n

    h_counts, j_counts = Counter(human), Counter(judge)
    labels = set(h_counts) | set(j_counts)
    pe = sum((h_counts[label] / n) * (j_counts[label] / n) for label in labels)

    kappa = 0.0 if pe >= 1.0 else (po - pe) / (1.0 - pe)

    ids = list(item_ids) if item_ids is not None else [str(i) for i in range(n)]
    disagreements = tuple((ids[i], human[i], judge[i]) for i in range(n) if human[i] != judge[i])

    return AgreementReport(
        n=n,
        observed_agreement=po,
        expected_agreement=pe,
        kappa=kappa,
        per_label_counts=dict(h_counts),
        disagreements=disagreements,
    )
