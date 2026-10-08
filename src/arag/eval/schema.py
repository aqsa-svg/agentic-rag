"""Golden set schema.

The schema is where eval integrity is won or lost, so the rules are enforced by the model
rather than by discipline. Four decisions carried over from DESIGN §7, plus one addition:

1. **Ground truth is a span a human chose, not prose a model wrote.** ``reference_answer``
   exists only for answer-correctness judging; ``ground_truth_spans`` is what retrieval is
   scored against. Retrieval metrics therefore never depend on generated text.
2. **``authored_by`` on every row.** Aggregate metrics over a mixed-provenance set are
   close to meaningless, so every report slices by it. ``SEED_UNVERIFIED`` is a third
   value the design doc did not have: items written to develop the harness before the
   corpus exists. The runner warns on them and the report labels them, because silently
   scoring against unverified labels is the exact failure this field prevents.
3. **``strata`` on every row.** A single mean hides that multi-hop collapsed while flat
   lookups improved.
4. **``must_not_cite`` makes supersession a failing condition**, not a soft preference.
5. **``injection_canary``** (addition): injection items carry a unique string that only a
   compromised system would emit. Without a canary, "did it resist the injection?" is a
   judgement call; with one it is a substring test that cannot be argued with.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from arag.eval.concepts import Concept, validate_clause_id
from arag.retrieval.types import DocSpan


class Strata(StrEnum):
    """The 15 query shapes from DESIGN §3.

    Four are adversarial: UNANSWERABLE, CONTRADICTORY, SUPERSESSION, INJECTION. Those are
    hand-authored only, never LLM-generated.

    The count has been wrong in this docstring before. It said 13 after stratum 14 was
    added, and the same stale 13 sat in docs/EVAL_EXPLAINED.md and in DESIGN §3's
    "Strata 11-13 are hand-authored" line - so adding a stratum silently renumbered the
    documentation three times over. If a member is added here, grep for the count.
    """

    FLAT_LOOKUP = "flat_lookup"
    MULTIHOP_TEMPORAL = "multihop_temporal"
    TABLE_FORMULA = "table_formula"
    CROSS_DOC_COMPARISON = "cross_doc_comparison"
    VOCAB_MISMATCH = "vocab_mismatch"
    ELIGIBILITY = "eligibility"
    PROCEDURAL = "procedural"
    DEFINITIONAL_CARVEOUT = "definitional_carveout"
    EXHAUSTIVE_LIST = "exhaustive_list"
    CLAUSE_TENSION = "clause_tension"
    CONDITIONAL_OVERRIDE = "conditional_override"
    UNANSWERABLE = "unanswerable"
    CONTRADICTORY = "contradictory"
    SUPERSESSION = "supersession"
    INJECTION = "injection"

    @property
    def is_adversarial(self) -> bool:
        return self in {
            Strata.UNANSWERABLE,
            Strata.CONTRADICTORY,
            Strata.SUPERSESSION,
            Strata.INJECTION,
        }

    @property
    def needs_agent(self) -> bool:
        """Strata that require conditional retrieval (DESIGN §1). Used to check that the
        agent earns its latency: these must improve in v3 while the others must not regress.
        """
        return self in {
            Strata.MULTIHOP_TEMPORAL,
            Strata.TABLE_FORMULA,
            Strata.CROSS_DOC_COMPARISON,
            Strata.ELIGIBILITY,
            Strata.CLAUSE_TENSION,
            Strata.CONDITIONAL_OVERRIDE,
        }


# SUPERSESSION (stratum 15), and why it is not CONTRADICTORY.
#
# Split out of CONTRADICTORY before any supersession item was labelled. The old stratum was
# carrying two requirements that are opposites, and the name did not say which applied:
#
#                       | CONTRADICTORY              | SUPERSESSION
#   status of positions | both CURRENT               | one is STALE
#   correct behaviour   | surface both, attributed   | suppress the stale one
#   expected_behaviour  | surface_conflict           | answer
#   hard failure        | one insurer's limit given  | citing star-comprehensive-2021
#                       | as though universal        | at all
#   resolvable by code? | no - a fact about the      | yes - version metadata and
#                       | market                    | as_of_date decide it
#   metric              | behaviour.conflict_surfaced| max_supersession_violations
#
# Per-stratum recall over a mixed population of those two measures nothing in particular.
# The split was made at 7 labelled items because the code cost is flat and the relabelling
# cost grows linearly with items already filed under the ambiguous name - at 120 items the
# taxonomy would have been frozen by its own data.
#
# needs_agent is False, and that is the substantive claim rather than an omission. Deciding
# which of two versions governs is a RETRIEVAL-side filter over document metadata: given an
# as_of_date, the stale wording should never enter the candidate set. It needs no
# conditional retrieval, no second hop, and no tool call, so it must not be used to justify
# the agent's latency. Contrast CONDITIONAL_OVERRIDE, which is needs_agent=True because
# answering genuinely requires retrieving a figure, noticing it is gated, and testing the
# gate.
#
# The stratum implies non-empty must_not_cite: without it there is nothing to violate, and
# max_supersession_violations reports a vacuous 0. That is not hypothetical - h-01 in the
# v1 baseline retrieves the superseded 2021 definition of "Hospital" at RANK 1, scores
# recall 1.000, and passes the gate, because the label carries no must_not_cite. See
# docs/BASELINE_V1.md.
#
# It also implies expected_behaviour=answer. The correct response is a single answer from
# the current wording, not a both-sides presentation - surfacing the stale clause IS the
# failure, so surface_conflict would invert the requirement.
#
# STRATUM 15 ARRIVED THE WAY STRATUM 14 DID: found while labelling, not during corpus
# analysis. That is now twice, and it is the argument for hand-labelling as a discovery
# method rather than data entry. Stratum 14 (conditional_override) was found by reading two
# clauses on adjacent pages; stratum 15 was found because a labelled span pointed at a code
# the index had silently dropped, which led to reading three deleted exclusions, one of
# which turned out to be a polarity reversal rather than a deletion. Neither was visible to
# any scan written beforehand.


# CLAUSE_TENSION - what the reference answer for one of these must and must not do.
#
# THE RULE: resolve the tension only if the DOCUMENT resolves it. Otherwise state the
# tension and what the reader should do about it.
#
# Where the wording does resolve it - a specificity rule, an explicit "notwithstanding
# clause X", an ordering of precedence - the reference answer says so and cites the
# resolving clause. Where it does not, the reference answer names every limb, says they
# point in different directions, and stops.
#
# Three reasons this is a hard rule rather than a preference:
#
# 1. A resolving answer the document does not support is UNGROUNDED BY CONSTRUCTION.
#    Every downstream metric - RAGAS faithfulness, the judge, max_ungrounded_citation_rate
#    - scores an answer against retrieved text. A reference answer asserting a resolution
#    with no span behind it rewards a system for a claim nothing supports, and penalises
#    the system that correctly surfaced both limbs. That inverts the gate.
# 2. It is what DESIGN R1 already requires: for a gated figure, surface the gate, do not
#    pick a branch. A reference answer that picked a branch would contradict the
#    requirement the system is being built to satisfy.
# 3. It is the honest answer. A policyholder asking "will you pay for my implants?" is
#    served by "the wording says two conflicting things, get it in writing before the
#    procedure" - not by a confident figure derived from whichever clause the retriever
#    happened to reach first.
#
# Worked example, h-21 (star-comprehensive-2025, entirely intra-document):
#   p35 excl.32  "(Dental implants are not payable)" - flat denial
#   p35 excl.32  "(except to the extent covered under Section II.17)" - and an exception
#   p16 cl.17    limb 1: "acute treatment to a natural tooth" - an implant is not one
#   p16 cl.17    limb 2: "services and supplies provided by a licensed dentist" - it is one
# Read the exception as empty and the drafter wrote a pointless cross-reference; read it as
# live and the parenthetical is false. The document does not choose. Neither does the label.


# CONDITIONAL_OVERRIDE, and why it is not CLAUSE_TENSION or CONTRADICTORY.
#
# Found during labelling: star-comprehensive-2025 states the pre-existing-disease waiting
# period twice, and NEITHER figure is wrong.
#
#   p31, excl.01  36 months of continuous coverage - the policy-wide rule
#   p30, clause 25 "Optional Cover (Buy Back of PED Waiting Period)"
#                 12 months - but only on payment of additional premium, only at first
#                 purchase, not on renewal, not on ported policies, and subject to
#                 pre-acceptance medical screening
#
# A retriever that reaches p30 and stops reports 12 months to someone who has 36. The two
# clauses are NOT in conflict, so CONTRADICTORY is wrong: one is a gated exception to the
# other. Nor is it CLAUSE_TENSION, whose mechanism is two rules pointing opposite ways
# with specificity resolving them - here the mechanism is a PRECONDITION that is closed by
# default and that the question almost never mentions.
#
# It also cannot be caught by version logic, because both clauses are in the same current
# document. That makes it a more realistic failure than the cross-version supersession
# trap, and it earns its own stratum so the two are never scored together.
#
# needs_agent is True: answering requires retrieving the figure, noticing it is gated,
# testing whether the gate is satisfied, and falling back to the policy-wide rule when it
# is not. That is conditional retrieval, which is the honest agent justification.


class ExpectedBehaviour(StrEnum):
    ANSWER = "answer"
    ABSTAIN = "abstain"
    SURFACE_CONFLICT = "surface_conflict"
    IGNORE_INJECTION = "ignore_injection"


class AuthoredBy(StrEnum):
    HUMAN = "human"
    LLM_VERIFIED = "llm_verified"
    SEED_UNVERIFIED = "seed_unverified"

    @property
    def is_trustworthy(self) -> bool:
        return self in {AuthoredBy.HUMAN, AuthoredBy.LLM_VERIFIED}


class GoldenItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    question: str
    strata: Strata
    authored_by: AuthoredBy
    expected_behaviour: ExpectedBehaviour

    ground_truth_spans: tuple[DocSpan, ...] = ()
    reference_answer: str | None = None
    as_of_date: date | None = None

    # Cross-document identity. clause_id is within-document provenance and CANNOT join
    # two versions of the same product: spike S5 measured that bariatric surgery is
    # 'Section 6' in star-2021 and item '15' under 'Section II' in star-2025, with no
    # arithmetic relationship. concept_id is the controlled-vocabulary join key, and it is
    # REQUIRED for any item whose whole point is comparing across documents.
    concept_id: Concept | None = None

    # Citing any of these is a hard failure. Format: "doc_id" or "doc_id#clause_id".
    must_not_cite: tuple[str, ...] = ()

    # For IGNORE_INJECTION only: a string that appears in the answer only if the
    # injected instruction was followed.
    injection_canary: str | None = None

    notes: str | None = None
    tags: tuple[str, ...] = ()

    # A label the system is KNOWN to get wrong today, recorded deliberately.
    #
    # An eval containing only cases expected to pass measures nothing. A known-failing
    # item is not a broken label - it is a diagnosis with a date on it, and the run where
    # it flips to passing is the most valuable row in EVAL_LOG.md. The reason is mandatory
    # so a stale known-failure cannot masquerade as an accepted defect.
    expected_to_fail: bool = False
    expected_failure_reason: str | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        """Collect EVERY problem with this item and raise once.

        Pydantic stops at the first raise, which turns hand-labelling 30 items into a
        fix-one-rerun loop: you correct the missing reference_answer, re-run, and only
        then discover the clause_id was also a placeholder. Accumulating means one pass
        per item.
        """
        problems: list[str] = []
        beh, strata = self.expected_behaviour, self.strata

        if not self.question.strip():
            problems.append("question is empty")
        elif len(self.question.strip()) < 10:
            problems.append(
                f"question {self.question.strip()!r} is too short to be a real user question"
            )

        if beh is ExpectedBehaviour.ANSWER:
            if not self.ground_truth_spans:
                problems.append(
                    "expected_behaviour=answer requires at least one ground_truth_span, "
                    "otherwise context recall is undefined for this item"
                )
            if not self.reference_answer or not self.reference_answer.strip():
                problems.append(
                    "expected_behaviour=answer requires a reference_answer for "
                    "answer-correctness judging"
                )

        if beh is ExpectedBehaviour.ABSTAIN and self.ground_truth_spans:
            problems.append(
                "expected_behaviour=abstain must have no ground_truth_spans; if the corpus "
                "can support an answer, the item is not unanswerable"
            )

        if strata is Strata.CONDITIONAL_OVERRIDE and len(self.ground_truth_spans) < 2:
            problems.append(
                "strata=conditional_override requires at least two ground_truth_spans: "
                "the gated figure AND the policy-wide rule it overrides. A single span "
                "cannot express 'this number applies only if...', which is the entire "
                "failure mode. Note the spans MAY be from the same document - unlike "
                "surface_conflict, this trap lives inside one current wording."
            )

        # A conflict needs at least two things in conflict. That is the requirement.
        #
        # This check originally read "spans from at least two DOCUMENTS", which was a proxy
        # for the requirement rather than the requirement itself - true of every example
        # that existed when it was written, all of them cross-version or cross-insurer.
        # Two items falsified the proxy while leaving the intent untouched: h-21, whose
        # three-way dental-implant tension sits entirely inside star-comprehensive-2025,
        # and the grace-period case, likewise intra-document. Neither could declare the
        # behaviour the item exists to test, so `behaviour.conflict_surfaced` was never
        # computed for them and a system that resolved the tension by picking a branch
        # would have scored identically to one that surfaced both limbs.
        #
        # So the general rule is restated at the level of the actual requirement, and the
        # strong form is kept only where the stratum makes it meaningful.
        #
        # Be honest about the direction, because the wrong version of this was briefly the
        # agreed one. The restatement was proposed and accepted as a "net tightening" on
        # the claim that a two-document item with the same clause id on both sides used to
        # pass and would no longer. That is false: two distinct documents are necessarily
        # two distinct (document_id, clause_id) pairs, so that case passes under both rules
        # - correctly, since it is the supersession shape. Neither party checked the
        # direction before agreeing on it. The accept/reject matrix is pinned in
        # tests/test_schema.py::TestConflictRequiresTwoThingsInConflict rather than
        # described, for that reason.
        #
        # For surface_conflict this is a pure LOOSENING.
        # Every span set the document rule accepted, the pair rule also accepts - two
        # distinct documents are necessarily two distinct pairs - and it additionally
        # accepts the same-document case. Nothing that used to pass now fails. What
        # remains is the guard the check was actually for: an item claiming a conflict
        # must name two things in conflict, so one span, or two spans differing only by
        # page, is still refused.
        #
        # The tightening is the separate CONTRADICTORY rule below, which applies to the
        # stratum regardless of expected_behaviour - previously a contradictory item with
        # expected_behaviour=answer carried no cross-source requirement at all.
        if beh is ExpectedBehaviour.SURFACE_CONFLICT:
            sides = {(s.document_id, s.clause_id) for s in self.ground_truth_spans}
            if len(sides) < 2:
                problems.append(
                    f"expected_behaviour=surface_conflict requires at least two distinct "
                    f"(document_id, clause_id) spans - there must be two things in conflict "
                    f"- got {len(sides)}. The spans MAY be from one document: an "
                    f"intra-document tension is still a conflict, and is the harder kind, "
                    f"because no version logic or authority rank can resolve it."
                )

        # CONTRADICTORY now means exactly one thing again: cross-source disagreement where
        # BOTH positions are current. Supersession moved to its own stratum.
        if strata is Strata.CONTRADICTORY:
            docs = {s.document_id for s in self.ground_truth_spans}
            if len(docs) < 2:
                problems.append(
                    f"strata=contradictory requires spans from at least two documents, got "
                    f"{len(docs)}. This stratum is cross-source disagreement between "
                    f"positions that are BOTH CURRENT - two insurers, or two products - and "
                    f"the cross-source comparison is the point. An item whose conflict is "
                    f"internal to one wording belongs in clause_tension; one where a "
                    f"position is stale and must not be surfaced belongs in supersession."
                )

        if strata is Strata.SUPERSESSION:
            if not self.must_not_cite:
                problems.append(
                    "strata=supersession requires a non-empty must_not_cite naming the "
                    "superseded document. Without it there is nothing to violate and "
                    "max_supersession_violations reports a vacuous 0 - which is exactly "
                    "how h-01's rank-1 retrieval of the 2021 wording passes the v1 gate "
                    "while scoring recall 1.000"
                )
            if beh is not ExpectedBehaviour.ANSWER:
                problems.append(
                    f"strata=supersession implies expected_behaviour=answer, got "
                    f"{beh.value!r}. The correct response is one answer from the current "
                    f"wording; surfacing the stale clause IS the failure, so "
                    f"surface_conflict would invert the requirement"
                )

        if beh is ExpectedBehaviour.IGNORE_INJECTION and not self.injection_canary:
            problems.append(
                "expected_behaviour=ignore_injection requires an injection_canary; without "
                "one, resistance is a judgement call rather than a test"
            )

        if strata is Strata.INJECTION and beh is not ExpectedBehaviour.IGNORE_INJECTION:
            problems.append("strata=injection implies expected_behaviour=ignore_injection")

        if strata is Strata.UNANSWERABLE and beh is not ExpectedBehaviour.ABSTAIN:
            problems.append("strata=unanswerable implies expected_behaviour=abstain")

        needs_concept = strata in {
            Strata.CONTRADICTORY,
            Strata.CROSS_DOC_COMPARISON,
            Strata.CONDITIONAL_OVERRIDE,
        } or bool(self.must_not_cite)
        if needs_concept and self.concept_id is None:
            problems.append(
                f"strata={strata.value} compares across documents, so it needs a concept_id. "
                "clause_id cannot join two versions of the same product (S5: 'Section 6 m.' "
                "in star-2021 == item '15' in star-2025), so without a concept_id this item "
                "cannot express 'same clause, different version'"
            )

        for span in self.ground_truth_spans:
            if span.clause_id is None:
                continue
            try:
                # A clause_id the SpanMatcher cannot parse produces an item that is present
                # in the file and absent from every metric, with no error anywhere.
                validate_clause_id(span.clause_id)
            except ValueError as exc:
                problems.append(f"{span.document_id}: {exc}")

        if self.expected_to_fail and not (self.expected_failure_reason or "").strip():
            problems.append(
                "expected_to_fail requires expected_failure_reason. Without a stated "
                "diagnosis a known-failure is indistinguishable from an abandoned one, "
                "and nobody can tell when it has been fixed"
            )
        if self.expected_failure_reason and not self.expected_to_fail:
            problems.append(
                "expected_failure_reason is set but expected_to_fail is false - if the "
                "item is expected to pass, the reason is stale and misleading"
            )

        for entry in self.must_not_cite:
            if not entry or entry.startswith("#"):
                problems.append(
                    f"must_not_cite entry {entry!r} must be 'doc_id' or 'doc_id#clause_id'"
                )

        if problems:
            joined = "; ".join(problems)
            raise ValueError(f"{self.id}: {len(problems)} problem(s): {joined}")
        return self


class GoldenSet(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: tuple[GoldenItem, ...]
    source: Path | None = None

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        dupes = [i for i, n in Counter(item.id for item in self.items).items() if n > 1]
        if dupes:
            raise ValueError(f"duplicate golden ids: {sorted(dupes)}")
        return self

    @classmethod
    def load(cls, path: Path) -> GoldenSet:
        """Read JSONL. Blank lines and ``#`` comment lines are skipped so the file can
        carry section headers for a human maintaining it by hand.
        """
        items: list[GoldenItem] = []
        errors: list[str] = []
        for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                items.append(GoldenItem.model_validate(json.loads(line)))
            except Exception as exc:
                errors.append(f"  {path.name}:{lineno}: {exc}")
        if errors:
            raise ValueError(f"{len(errors)} invalid golden item(s):\n" + "\n".join(errors))
        return cls(items=tuple(items), source=path)

    # --- slicing helpers used by the report generator ---

    def by_strata(self, strata: Strata) -> tuple[GoldenItem, ...]:
        return tuple(i for i in self.items if i.strata is strata)

    def trustworthy(self) -> tuple[GoldenItem, ...]:
        return tuple(i for i in self.items if i.authored_by.is_trustworthy)

    def unverified(self) -> tuple[GoldenItem, ...]:
        return tuple(i for i in self.items if not i.authored_by.is_trustworthy)

    def known_failing(self) -> tuple[GoldenItem, ...]:
        """Items deliberately labelled as currently-wrong. Reported separately so they
        never read as broken labels, and so a fix is visible the moment it lands.
        """
        return tuple(i for i in self.items if i.expected_to_fail)

    def strata_counts(self) -> dict[str, int]:
        return dict(Counter(i.strata.value for i in self.items))

    def provenance_counts(self) -> dict[str, int]:
        return dict(Counter(i.authored_by.value for i in self.items))

    def coverage_gaps(self) -> list[str]:
        """Strata with no items. Reported by ``arag-eval validate`` so an incomplete
        golden set is visible rather than silently producing confident partial metrics.
        """
        present = {i.strata for i in self.items}
        return [s.value for s in Strata if s not in present]

    def subset(self, ids: set[str]) -> GoldenSet:
        return GoldenSet(items=tuple(i for i in self.items if i.id in ids), source=self.source)


class SetTargets(BaseModel):
    """Composition the golden set must reach before v1 numbers are publishable.

    Encoded here rather than in prose so ``arag-eval validate`` can report the shortfall.
    Values follow DESIGN §13 assumption 6.
    """

    # CUT FROM 120 ON 2026-10-02, and the cost is stated rather than hidden.
    #
    # 120 was chosen for statistical comfort. Measured against how labelling actually goes -
    # 7 items in four weeks, each needing a human to read the PDFs, locate the clause and
    # write a reference answer - the remaining 113 are roughly 56 hours. A target nobody
    # reaches is not a target; it is a permanent NOT YET PUBLISHABLE that stops carrying
    # information.
    #
    # What the cut COSTS, said plainly: at 40 items with 5 per adversarial stratum, no rate
    # here is publishable. One item moves an adversarial rate by 20 percentage points. The
    # floors below are COVERAGE floors - enough labels that each failure mode is represented
    # at all - and they are not enough for a quotable abstain-recall or injection-resistance
    # figure. Any number computed from this set remains a development signal, which is what
    # the warning in `arag-eval validate` has always said.
    #
    # Raising it back is one edit. Doing so should follow evidence that the labelling rate
    # changed, not optimism that it will.
    #
    # LOWERED 40 -> 30 on 2026-10-08, and this is the evidence the paragraph above asks for,
    # applied DOWNWARD rather than up. 30 was the planned human-authored deliverable the
    # whole way through; 40 was set at the 2026-10-02 cut before any labelling rate had been
    # measured, as headroom over the 30. Measured rate since: 30 items over several days,
    # with a defect caught in most batches (a resolving-but-wrong clause, a conflict that
    # lived only in a summary, a reference answer that picked one branch of a gate). The
    # last ten items would cost about a day and would be taken from the v1->v2 cycle, which
    # is the higher-value use of that day. So total meets min_human_authored: the deliverable
    # is 30 human-authored items, every stratum floor met, and raising total again is a v2
    # item, not a v1 one.
    total: int = 30
    # Equal to total now: the set is entirely human-authored, which is the project's central
    # claim (human reading found what scanning missed). Kept as a separate field, not folded
    # into total, because a future non-human portion would reopen the gap and this is the
    # floor that would still have to hold.
    min_human_authored: int = 30
    min_per_adversarial_stratum: int = 5
    max_seed_unverified: int = 0

    # Per-stratum floors, because the 10 is not one requirement.
    #
    # SUPERSEDED REASONING, kept because the conclusion changed and the record should show
    # it. This block argued that rate-gated strata need n=10 while supersession, gated by a
    # violation COUNT, needs only coverage. That distinction was real and is now moot: the
    # 2026-10-02 cut put every adversarial floor at 5, which is below the threshold the
    # rate argument itself set. The floors are therefore all coverage floors now, and no
    # rate computed from this set is quotable - stated in `total` above rather than left for
    # a reader to infer from a comment that no longer matches the numbers.
    #
    # SUPERSESSION is gated by a COUNT, not a rate: `max_supersession_violations: 0`, where
    # a single violation fails. So its floor is not a statistical-resolution question but a
    # COVERAGE one - how many distinct ways can a superseded wording mislead? Measured
    # against the one superseded/current pair this corpus has (star-2021 -> star-2025),
    # five mechanisms are attested:
    #
    #   1. outright deletion            excl.23, excl.30 - subject gone from 2025 entirely
    #   2. polarity reversal            excl.33 - sleep apnea moves from excluded to a
    #                                   co-morbidity that QUALIFIES for bariatric cover
    #   3. renumbering, same rule       bariatric surgery: "II - Section 6 m." -> item "15"
    #   4. changed definition           def.hospital differs between the wordings; this is
    #                                   the mechanism h-01 already exercises
    #   5. pagination drift             18 pages vs 48 for near-identical content, so a
    #                                   page-keyed answer from the wrong version misdirects
    #
    # Hence 5: one item per attested mechanism. Raising it to 10 would demand five items
    # whose mechanism nobody has yet shown exists in this corpus, and unfindable label
    # targets are how a composition gate teaches people to bypass it. Revisit if a sixth
    # mechanism turns up - the constraint is the corpus, not the number.
    # CONTRADICTORY is lowered to 3, and the reason is the corpus rather than the labelling.
    #
    # The floor was set at 5 before anyone surveyed what the corpus actually contains. A
    # systematic sweep on 2026-10-07 across every CURRENT document - star-comprehensive-2025,
    # nivabupa-rise, nivabupa-reassure2, irdai-master-circular-2024, irdai-annexure-2024 -
    # found THREE genuine cross-source contradictions and no more.
    #
    # What it found instead: IRDAI's 2024 standardisation mandates the WORDING, not merely
    # the substance, for grace period, free look, moratorium, cancellation, portability and
    # claim settlement. Those clauses are now verbatim-identical across all four insurer
    # documents - 30 days, 60 months, 7 days' notice with proportionate refund, 15/30 days
    # by payment mode. The cross-insurer disagreements this stratum exists to catch have
    # largely been REGULATED OUT OF EXISTENCE. What survives is where an insurer wording
    # retains a pre-standardisation term.
    #
    # The survivors, kept as the record of what the floor is based on:
    #   1. cover during the grace period   star p41 cl.9 says NOT available;
    #                                      nivabupa-rise p18 cl.8.1.3 says it IS
    #   2. who submits claim documents     irdai circular p9 cl.17 says the policyholder
    #                                      "shall not be required to"; star p37-38 makes
    #                                      filing within 15 days a condition PRECEDENT
    #
    # LOWERED ONCE MORE 2 -> 1 on 2026-10-08, and this is the floor the corpus can actually
    # meet. Only ONE genuine cross-insurer contradiction survives here: h-18, grace-period
    # cover (Star says not available on renewal, Niva-rise says available). The grace-period
    # case was authored as two items testing two different failures - h-18 under
    # contradictory (insurer unstated; surface both, attributed) and h-25 under
    # cross_doc_comparison (Star named; must_not_cite the Niva clause). h-25 is correctly
    # filed under cross_doc_comparison - its test is provenance, not conflict - so it does
    # NOT count toward this floor, and contradictory holds exactly one item.
    #
    # A floor of 2 would be a gate the corpus cannot satisfy no matter how much labelling is
    # done, because IRDAI's 2024 standardisation left exactly one surviving cross-insurer
    # contradiction (C2 and C3 were withdrawn - C2 is authority-override, which no stratum
    # expresses, and C3 was a form heading misread as an obligation). An unmeetable gate is
    # a permanent red light that people learn to ignore, which is worse than a lower one that
    # means something. The constraint is the corpus, not the labelling effort.
    # LOWERED AGAIN 3 -> 2 on 2026-10-07. The third candidate was withdrawn on reading its
    # text rather than its summary: nivabupa-reassure2 p25 cl.6.2.4 / rise p23 cl.8.2.4 say
    # "Documents required with claim form:" followed by a list. That is a LIST HEADING on a
    # form, not an obligation clause - it does not say the claimant must supply them, and a
    # form convention does not disagree with a rule about who collects documents. It was
    # offered as a candidate on the strength of a paraphrase; the paraphrase asserted a duty
    # the document never states.
    #
    # Recorded rather than quietly dropped, because the error is instructive and is the one
    # this project keeps finding: a summary that reads as a position when the source is
    # silent. Two candidates survive, and #2's own premise is contested - see DESIGN on
    # authority-override, which this taxonomy has no stratum for.
    #
    # Deliberately NOT done: widening to nivabupa-rise against nivabupa-reassure2. Two
    # products from one insurer disagreeing is not the failure this stratum exists to
    # catch, and padding a floor to hit a number makes the composition gate decorative -
    # the gate would then be measuring the labeller's willingness to pad, not the corpus.
    #
    # Revisit if a fifth insurer enters the corpus. The constraint is the corpus, not the
    # number, and not the labelling effort. Recorded in DESIGN and LIMITATIONS.
    min_per_stratum: dict[Strata, int] = {Strata.SUPERSESSION: 5, Strata.CONTRADICTORY: 1}

    def floor_for(self, stratum: Strata) -> int:
        return self.min_per_stratum.get(stratum, self.min_per_adversarial_stratum)

    def shortfall(self, gs: GoldenSet) -> list[str]:
        out: list[str] = []
        if len(gs.items) < self.total:
            out.append(f"total {len(gs.items)}/{self.total}")
        human = sum(1 for i in gs.items if i.authored_by is AuthoredBy.HUMAN)
        if human < self.min_human_authored:
            out.append(f"human-authored {human}/{self.min_human_authored}")
        for s in (s for s in Strata if s.is_adversarial):
            n = len(gs.by_strata(s))
            floor = self.floor_for(s)
            if n < floor:
                out.append(f"{s.value} {n}/{floor}")
        seed = len(gs.unverified())
        if seed > self.max_seed_unverified:
            out.append(f"seed_unverified {seed} (must be {self.max_seed_unverified})")
        return out
