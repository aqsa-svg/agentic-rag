"""Golden-set schema tests.

The schema enforces eval integrity so that discipline doesn't have to. Each test below
corresponds to a way a golden set can silently become unmeasurable.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import ValidationError

from arag.eval.schema import (
    AuthoredBy,
    ExpectedBehaviour,
    GoldenItem,
    GoldenSet,
    SetTargets,
    Strata,
)


def item(**overrides: object) -> GoldenItem:
    base: dict[str, object] = {
        "id": "t-1",
        "question": "Is X covered?",
        "strata": Strata.FLAT_LOOKUP,
        "authored_by": AuthoredBy.HUMAN,
        "expected_behaviour": ExpectedBehaviour.ANSWER,
        "ground_truth_spans": [{"document_id": "d1", "page": 1, "clause_id": "3.1"}],
        "reference_answer": "Yes, subject to conditions.",
    }
    return GoldenItem.model_validate(base | overrides)


class TestAnswerItems:
    def test_valid(self) -> None:
        assert item().expected_behaviour is ExpectedBehaviour.ANSWER

    def test_requires_a_ground_truth_span(self) -> None:
        """Without a span, context recall is undefined for the item and it would be
        silently skipped by the aggregator — present in the file, absent from the metrics.
        """
        with pytest.raises(ValidationError, match="requires at least one"):
            item(ground_truth_spans=[])

    def test_requires_a_reference_answer(self) -> None:
        with pytest.raises(ValidationError, match="requires a reference_answer"):
            item(reference_answer=None)


class TestAbstainItems:
    def test_must_not_carry_spans(self) -> None:
        """If the corpus can support an answer, the item is not unanswerable.

        A span on an abstain item usually means the author mislabelled a hard question as
        impossible, which would train the threshold to refuse answerable queries.
        """
        with pytest.raises(ValidationError, match="must have no ground_truth_spans"):
            item(
                strata=Strata.UNANSWERABLE,
                expected_behaviour=ExpectedBehaviour.ABSTAIN,
            )

    def test_valid_without_spans(self) -> None:
        got = item(
            strata=Strata.UNANSWERABLE,
            expected_behaviour=ExpectedBehaviour.ABSTAIN,
            ground_truth_spans=[],
            reference_answer="Not stated in the corpus.",
        )
        assert got.ground_truth_spans == ()

    def test_unanswerable_stratum_implies_abstain(self) -> None:
        """Stratum and expected behaviour cannot disagree. An 'unanswerable' item that
        expects an answer would be scored as a retrieval failure forever.
        """
        with pytest.raises(ValidationError, match="implies expected_behaviour=abstain"):
            item(
                strata=Strata.UNANSWERABLE,
                expected_behaviour=ExpectedBehaviour.SURFACE_CONFLICT,
                concept_id="limit.room_rent",
                ground_truth_spans=[
                    {"document_id": "d1", "clause_id": "3.1"},
                    {"document_id": "d2", "clause_id": "3.2"},
                ],
            )


class TestConflictItems:
    def test_requires_two_documents(self) -> None:
        """One document cannot contradict itself across insurers, and a single-document
        'conflict' item would pass just by citing that document once.
        """
        with pytest.raises(ValidationError, match="at least two documents"):
            item(
                strata=Strata.CONTRADICTORY,
                expected_behaviour=ExpectedBehaviour.SURFACE_CONFLICT,
                concept_id="limit.room_rent",
                ground_truth_spans=[
                    {"document_id": "d1", "clause_id": "3.1"},
                    {"document_id": "d1", "clause_id": "3.2"},
                ],
            )

    def test_valid_with_two_documents(self) -> None:
        got = item(
            strata=Strata.CONTRADICTORY,
            expected_behaviour=ExpectedBehaviour.SURFACE_CONFLICT,
            concept_id="limit.room_rent",
            ground_truth_spans=[
                {"document_id": "star", "clause_id": "3.1"},
                {"document_id": "nivabupa", "clause_id": "3.2"},
            ],
        )
        assert len({s.document_id for s in got.ground_truth_spans}) == 2


class TestInjectionItems:
    def test_requires_a_canary(self) -> None:
        """Without a canary, "did it resist?" is a judgement call rather than a test."""
        with pytest.raises(ValidationError, match="requires an injection_canary"):
            item(
                strata=Strata.INJECTION,
                expected_behaviour=ExpectedBehaviour.IGNORE_INJECTION,
                ground_truth_spans=[],
                reference_answer=None,
            )

    def test_injection_stratum_implies_ignore_injection(self) -> None:
        with pytest.raises(ValidationError, match="implies expected_behaviour=ignore_injection"):
            item(strata=Strata.INJECTION)


class TestCrossDocumentJoinKey:
    """concept_id is required wherever clause_id provably cannot join.

    Spike S5 measured that the two Star wordings use incompatible identifier schemes for
    identical content: bariatric surgery is "Section 6 m." in the 2021 document and item
    "15" under "Section II" in the 2025 one, with no arithmetic relationship. So an item
    whose purpose is comparing across documents cannot express itself with clause_id alone.
    """

    def test_contradictory_requires_a_concept_id(self) -> None:
        with pytest.raises(ValidationError, match="needs a concept_id"):
            item(
                strata=Strata.CONTRADICTORY,
                expected_behaviour=ExpectedBehaviour.SURFACE_CONFLICT,
                ground_truth_spans=[
                    {"document_id": "star", "clause_id": "3.1"},
                    {"document_id": "nivabupa", "clause_id": "3.2"},
                ],
            )

    def test_cross_doc_comparison_requires_a_concept_id(self) -> None:
        with pytest.raises(ValidationError, match="needs a concept_id"):
            item(strata=Strata.CROSS_DOC_COMPARISON)

    def test_must_not_cite_requires_a_concept_id(self) -> None:
        """A supersession trap without a concept_id cannot say WHICH clause superseded."""
        with pytest.raises(ValidationError, match="needs a concept_id"):
            item(must_not_cite=["star-comprehensive-2021"])

    def test_unknown_concept_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            item(strata=Strata.CROSS_DOC_COMPARISON, concept_id="benefit.teleportation")


class TestClauseIdCanonicalisation:
    def test_non_canonical_clause_id_is_rejected(self) -> None:
        """A clause_id the SpanMatcher cannot parse yields an item that is present in the
        file and absent from every metric, with no error anywhere.
        """
        with pytest.raises(ValidationError, match="not in canonical form"):
            item(ground_truth_spans=[{"document_id": "d1", "clause_id": "Room Rent"}])

    def test_template_placeholder_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not in canonical form"):
            item(ground_truth_spans=[{"document_id": "d1", "clause_id": "FIXME"}])

    def test_page_zero_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            item(ground_truth_spans=[{"document_id": "d1", "page": 0, "clause_id": "3.1"}])

    def test_empty_question_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="question is empty"):
            item(question="")

    def test_all_problems_reported_at_once(self) -> None:
        """Hand-labelling 30 items must not be a fix-one-rerun loop."""
        with pytest.raises(ValidationError, match="3 problem"):
            item(
                question="x",
                reference_answer="",
                ground_truth_spans=[{"document_id": "d1", "clause_id": "FIXME"}],
            )


class TestMustNotCite:
    def test_rejects_a_pattern_with_no_document(self) -> None:
        with pytest.raises(ValidationError, match="must be 'doc_id'"):
            item(must_not_cite=["#4.2"])


class TestExtraFields:
    def test_unknown_key_is_rejected(self) -> None:
        """extra='forbid' catches the typo that would otherwise make a field silently
        inert — e.g. ``must_not_site`` sitting in the file doing nothing.
        """
        with pytest.raises(ValidationError):
            item(mustnotcite=["d1"])


class TestGoldenSet:
    def test_duplicate_ids_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate golden ids"):
            GoldenSet(items=(item(id="a"), item(id="a")))

    def test_load_reports_every_bad_line_at_once(self, tmp_path: Path) -> None:
        """Reporting one error per run would make fixing a 120-item file a 120-run loop."""
        path = tmp_path / "bad.jsonl"
        path.write_text(
            "\n".join(
                [
                    "# a comment line is skipped",
                    "",
                    '{"id": "x1", "question": "q", "strata": "flat_lookup", '
                    '"authored_by": "human", "expected_behaviour": "answer"}',
                    '{"id": "x2", "question": "q", "strata": "injection", '
                    '"authored_by": "human", "expected_behaviour": "ignore_injection"}',
                ]
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError) as exc:
            GoldenSet.load(path)
        message = str(exc.value)
        assert "2 invalid golden item(s)" in message
        assert "bad.jsonl:3" in message
        assert "bad.jsonl:4" in message

    def test_slices(self) -> None:
        gs = GoldenSet(
            items=(
                item(id="a"),
                item(id="b", authored_by=AuthoredBy.SEED_UNVERIFIED),
            )
        )
        assert len(gs.trustworthy()) == 1
        assert len(gs.unverified()) == 1
        assert gs.provenance_counts() == {"human": 1, "seed_unverified": 1}

    def test_coverage_gaps_lists_missing_strata(self) -> None:
        gs = GoldenSet(items=(item(),))
        gaps = gs.coverage_gaps()
        assert "injection" in gaps
        assert "flat_lookup" not in gaps


class TestStrataProperties:
    def test_adversarial_set(self) -> None:
        assert Strata.UNANSWERABLE.is_adversarial
        assert Strata.CONTRADICTORY.is_adversarial
        assert Strata.INJECTION.is_adversarial
        assert not Strata.FLAT_LOOKUP.is_adversarial

    def test_agent_requiring_strata_match_the_design_doc(self) -> None:
        """DESIGN §3 marks exactly five strata as needing conditional retrieval. If this
        drifts, the v3 claim "the agent improves the strata that need it" stops meaning
        anything.
        """
        needs = {s.value for s in Strata if s.needs_agent}
        assert needs == {
            "multihop_temporal",
            "table_formula",
            "cross_doc_comparison",
            "eligibility",
            "clause_tension",
            # Added after labelling surfaced the gated-figure trap. Answering requires
            # testing whether a precondition holds, which the query never states.
            "conditional_override",
        }


class TestShippedGoldenSet:
    def test_loads_and_validates(self, golden_path: Path) -> None:
        gs = GoldenSet.load(golden_path)
        assert len(gs.items) >= 16

    def test_every_stratum_is_represented(self, golden_path: Path) -> None:
        assert GoldenSet.load(golden_path).coverage_gaps() == []

    def test_all_seed_items_are_marked_unverified(self, golden_path: Path) -> None:
        """Guards against the worst failure available here: a seed item quietly relabelled
        as human-authored, making unverified guesses look like evidence.
        """
        gs = GoldenSet.load(golden_path)
        assert all(i.authored_by is AuthoredBy.SEED_UNVERIFIED for i in gs.items)

    def test_is_not_yet_publishable(self, golden_path: Path) -> None:
        """This test is expected to FAIL-to-empty (i.e. shortfall becomes []) only once the
        real 120-item set exists. Until then it documents that we know it isn't ready.
        """
        shortfall = SetTargets().shortfall(GoldenSet.load(golden_path))
        assert shortfall, "golden set now meets targets — update this test and the README"
        assert any("seed_unverified" in line for line in shortfall)


class TestConditionalOverride:
    """A figure that is real but gated on a precondition the question never mentions.

    Found during labelling, not during corpus analysis: star-comprehensive-2025 states
    the PED waiting period as 36 months (p31, excl.01) and as 12 months (p30, clause 25
    "Optional Cover - Buy Back of PED Waiting Period"). Neither is wrong. A retriever
    that reaches p30 and stops reports 12 months to someone who has 36.

    It gets its own stratum because it is neither of the two nearby ones:
      * not CONTRADICTORY - the clauses do not conflict, one gates the other
      * not CLAUSE_TENSION - that mechanism is specificity resolving opposed rules
    and because both clauses live in the SAME current document, so no version logic
    can catch it.
    """

    def test_requires_at_least_two_spans(self) -> None:
        """One span cannot express 'this number applies only if...'."""
        with pytest.raises(ValidationError, match="at least two ground_truth_spans"):
            item(
                strata=Strata.CONDITIONAL_OVERRIDE,
                concept_id="waiting_period.pre_existing_disease",
                ground_truth_spans=[{"document_id": "d1", "page": 31, "clause_id": "excl.01"}],
            )

    def test_both_spans_may_come_from_the_same_document(self) -> None:
        """The distinguishing property. surface_conflict demands two documents;
        this trap lives inside one, which is exactly why it is more dangerous.
        """
        got = item(
            strata=Strata.CONDITIONAL_OVERRIDE,
            concept_id="waiting_period.pre_existing_disease",
            ground_truth_spans=[
                {"document_id": "star", "page": 31, "clause_id": "excl.01"},
                {"document_id": "star", "page": 30, "clause_id": "25"},
            ],
        )
        assert len({s.document_id for s in got.ground_truth_spans}) == 1

    def test_requires_a_concept_id(self) -> None:
        with pytest.raises(ValidationError, match="needs a concept_id"):
            item(
                strata=Strata.CONDITIONAL_OVERRIDE,
                ground_truth_spans=[
                    {"document_id": "star", "page": 31, "clause_id": "excl.01"},
                    {"document_id": "star", "page": 30, "clause_id": "25"},
                ],
            )

    def test_needs_the_agent(self) -> None:
        """Answering requires retrieving the figure, noticing it is gated, testing
        the gate, and falling back to the policy-wide rule. That is conditional
        retrieval - the honest agent justification from DESIGN section 1.
        """
        assert Strata.CONDITIONAL_OVERRIDE.needs_agent

    def test_is_not_adversarial(self) -> None:
        """Both clauses are legitimate policy text, so this is not a planted trap
        like the injection or unanswerable strata - it is the document as written.
        """
        assert not Strata.CONDITIONAL_OVERRIDE.is_adversarial


class TestConflictRequiresTwoThingsInConflict:
    """The restated surface_conflict rule, and the contradictory rule that replaces it.

    The original check required spans from >=2 DOCUMENTS. That was a proxy for "a conflict
    needs two things in conflict", true of every example that existed when it was written.
    h-21 (dental implants, three-way, entirely inside star-comprehensive-2025) and the
    grace-period case falsified the proxy, not the intent: neither could declare the
    behaviour it exists to test, so `conflict_surfaced` was never computed for them and a
    system that picked a branch scored the same as one that surfaced both limbs.

    These tests pin both halves, including the case that shows the rule is not vacuous.
    """

    def _item(self, **kw: object) -> dict[str, object]:
        base: dict[str, object] = {
            "id": "t-01",
            "question": "Will the policy pay for treatment of my sleep apnea condition?",
            "authored_by": "human",
            "reference_answer": "The wording points in two directions; see both clauses.",
            "concept_id": "exclusion.permanent",
        }
        base.update(kw)
        return base

    def _span(self, doc: str, clause: str | None, page: int = 1) -> dict[str, object]:
        return {"document_id": doc, "page": page, "clause_id": clause}

    def test_intra_document_conflict_is_accepted(self) -> None:
        """h-21's shape. This is the case the old rule wrongly refused."""
        GoldenItem(
            **self._item(
                strata="clause_tension",
                expected_behaviour="surface_conflict",
                ground_truth_spans=[
                    self._span("star-comprehensive-2025", "excl.32", 35),
                    self._span("star-comprehensive-2025", "17", 16),
                ],
            )
        )

    def test_cross_document_conflict_is_still_accepted(self) -> None:
        """Same clause id in two versions - the supersession shape - must keep working."""
        GoldenItem(
            **self._item(
                strata="clause_tension",
                expected_behaviour="surface_conflict",
                ground_truth_spans=[
                    self._span("star-comprehensive-2025", "excl.33"),
                    self._span("star-comprehensive-2021", "excl.33"),
                ],
            )
        )

    def test_one_span_is_not_a_conflict(self) -> None:
        with pytest.raises(ValidationError, match="two distinct"):
            GoldenItem(
                **self._item(
                    strata="clause_tension",
                    expected_behaviour="surface_conflict",
                    ground_truth_spans=[self._span("star-comprehensive-2025", "excl.32")],
                )
            )

    def test_two_spans_differing_only_by_page_is_not_a_conflict(self) -> None:
        """LOAD-BEARING. Do not delete this as redundant with the one-span test.

        The >=2-document rule was replaced by ">=2 distinct (document_id, clause_id)
        spans", and that replacement is only a *restatement* of the original guard while
        something still refuses a set of spans that names one clause twice. This test is
        that something.

        Delete it and the remaining coverage is "one span is refused" - which every
        two-span item passes, including an item citing `excl.32` on p35 and p44 and
        claiming a conflict between a clause and itself. The rule would then be satisfied
        by span *count* rather than by two things actually being in conflict, the guard
        would be gone rather than restated, and every other test in this class would still
        be green.

        It also refuses the shape a labeller most plausibly writes by accident: the same
        clause found on two pages, which the clause index reports for most bare numeric
        ids in this corpus.
        """
        with pytest.raises(ValidationError, match="two distinct"):
            GoldenItem(
                **self._item(
                    strata="clause_tension",
                    expected_behaviour="surface_conflict",
                    ground_truth_spans=[
                        self._span("star-comprehensive-2025", "excl.32", 35),
                        self._span("star-comprehensive-2025", "excl.32", 44),
                    ],
                )
            )

    def test_contradictory_still_requires_two_documents(self) -> None:
        """The cross-source requirement moves to the stratum, where it means something.

        It now applies regardless of expected_behaviour; before, a contradictory item with
        expected_behaviour=answer carried no cross-source requirement at all.
        """
        with pytest.raises(ValidationError, match="at least two documents"):
            GoldenItem(
                **self._item(
                    strata="contradictory",
                    expected_behaviour="answer",
                    ground_truth_spans=[
                        self._span("star-comprehensive-2025", "excl.32", 35),
                        self._span("star-comprehensive-2025", "17", 16),
                    ],
                )
            )

    def test_contradictory_across_two_insurers_is_accepted(self) -> None:
        GoldenItem(
            **self._item(
                strata="contradictory",
                expected_behaviour="answer",
                ground_truth_spans=[
                    self._span("star-comprehensive-2025", "excl.06", 33),
                    self._span("nivabupa-rise", "4.1"),
                ],
            )
        )


class TestSupersessionStratum:
    """Stratum 15, split out of CONTRADICTORY before any supersession item was labelled.

    The two strata had opposite requirements under one name: contradictory means both
    positions are current and both must be surfaced; supersession means one is stale and
    must NOT be surfaced. Per-stratum metrics over a mixed population of the two measure
    nothing in particular.
    """

    def _item(self, **kw: object) -> dict[str, object]:
        base: dict[str, object] = {
            "id": "s-01",
            "question": "Is treatment for my sleep apnea covered under the current policy?",
            "authored_by": "human",
            "reference_answer": "The current wording carries no exclusion for it.",
            "strata": "supersession",
            "expected_behaviour": "answer",
            "concept_id": "exclusion.permanent",
            "ground_truth_spans": [
                {"document_id": "star-comprehensive-2025", "page": 33, "clause_id": "excl.06"}
            ],
            "must_not_cite": ["star-comprehensive-2021"],
        }
        base.update(kw)
        return base

    def test_a_well_formed_supersession_item_is_accepted(self) -> None:
        item = GoldenItem(**self._item())
        assert item.strata is Strata.SUPERSESSION

    def test_supersession_is_adversarial(self) -> None:
        assert Strata.SUPERSESSION.is_adversarial

    def test_supersession_does_not_need_the_agent(self) -> None:
        """Substantive, not an omission.

        Choosing between two versions is a retrieval-side filter over document metadata:
        given an as_of_date the stale wording should never enter the candidate set. No
        second hop, no tool call. So this stratum must not be used to justify the agent's
        latency in v3, unlike conditional_override.
        """
        assert not Strata.SUPERSESSION.needs_agent
        assert Strata.CONDITIONAL_OVERRIDE.needs_agent

    def test_must_not_cite_is_required(self) -> None:
        """Without it, max_supersession_violations reports a vacuous 0.

        Not hypothetical: h-01 retrieves the superseded 2021 definition of Hospital at
        rank 1, scores recall 1.000 and passes the v1 gate, because it carries no
        must_not_cite. See docs/BASELINE_V1.md.
        """
        with pytest.raises(ValidationError, match="non-empty must_not_cite"):
            GoldenItem(**self._item(must_not_cite=[]))

    def test_surface_conflict_is_refused(self) -> None:
        """Surfacing the stale clause IS the failure, so both-sides inverts the point."""
        with pytest.raises(ValidationError, match="implies expected_behaviour=answer"):
            GoldenItem(
                **self._item(
                    expected_behaviour="surface_conflict",
                    ground_truth_spans=[
                        {
                            "document_id": "star-comprehensive-2025",
                            "page": 33,
                            "clause_id": "excl.06",
                        },
                        {"document_id": "nivabupa-rise", "page": 1, "clause_id": "4.1"},
                    ],
                )
            )

    def test_abstain_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="implies expected_behaviour=answer"):
            GoldenItem(**self._item(expected_behaviour="abstain", ground_truth_spans=[]))


class TestCompositionFloors:
    """The floors, and the reasoning that produced them - which has changed once.

    ORIGINALLY: rate-gated strata (unanswerable, contradictory, injection) took n=10,
    because one item moves a rate by 10 percentage points and below that the number
    describes the sample rather than the system. Supersession took 5, because it is gated
    by a violation COUNT rather than a rate, so its floor was a coverage question - five
    attested mechanisms in this corpus, one item each.

    CHANGED 2026-10-02. The total was cut 120 -> 40 after measuring the labelling rate: 7
    items in four weeks, each needing a human to read the PDFs and write a reference answer,
    leaving ~56 hours of work against a target that had stopped carrying information.

    The cut puts every adversarial floor at 5, which is BELOW the threshold the rate
    argument itself set. So that argument no longer applies and the test no longer asserts
    it: all floors are now coverage floors, and no rate from this set is quotable. Recorded
    rather than quietly dropped, because the earlier reasoning was sound and it was the
    premise that moved, not the logic.

    CHANGED AGAIN 2026-10-07, for contradictory only, and in the other direction. A
    systematic survey of every CURRENT document found THREE genuine cross-source
    contradictions and no more: IRDAI's 2024 standardisation mandates the wording for
    grace period, free look, moratorium, cancellation, portability and claim settlement,
    so those clauses are verbatim-identical across all four insurer wordings. The
    disagreements this stratum exists to catch have largely been regulated out of
    existence, and what survives is where an insurer retained a pre-standardisation term.

    So the floor was lowered to 3 rather than filled - not by relaxing what counts as a
    disagreement, and not by pairing nivabupa-rise against nivabupa-reassure2, which is two
    products from one insurer and not the failure this stratum is for. A composition gate
    satisfiable by padding measures the labeller's willingness to pad. See DESIGN and
    LIMITATIONS.
    """

    # Pinned individually. "They are all five" stopped being expressible the moment one of
    # them was not, and uniformity was never the property worth guarding - this is: NO floor
    # moves without someone editing this table and recording why.
    EXPECTED_FLOORS: ClassVar[dict[Strata, int]] = {
        Strata.SUPERSESSION: 5,  # five attested mechanisms in this corpus, one item each
        Strata.CONTRADICTORY: 1,  # corpus-limited: one survives (h-18); see schema.py
        Strata.UNANSWERABLE: 5,
        Strata.INJECTION: 5,
    }

    def test_each_adversarial_floor_matches_its_recorded_reason(self) -> None:
        targets = SetTargets()
        actual = {s: targets.floor_for(s) for s in Strata if s.is_adversarial}
        assert actual == self.EXPECTED_FLOORS, (
            "an adversarial floor changed. That is allowed, and not allowed to happen "
            "quietly: edit EXPECTED_FLOORS and record WHY in docs/DESIGN.md, as the "
            "2026-10-02 cut and the 2026-10-07 contradictory survey both are."
        )

    def test_a_lowered_floor_still_gates(self) -> None:
        """Lowering a floor must not turn the gate off for that stratum.

        contradictory went 5 -> 3 because the corpus holds three. At zero items it must
        still appear in the shortfall - otherwise "we lowered it" and "we removed it" look
        identical from the outside, which is how a gate quietly stops gating.
        """
        shortfall = " ".join(SetTargets().shortfall(GoldenSet(items=[], source=Path("x"))))
        assert "contradictory 0/1" in shortfall, shortfall

    def test_the_total_and_human_floor_match_the_cut(self) -> None:
        """Pinned so the target cannot drift without someone editing this line.

        human-authored is 30 of 40 - 75%, up from the 50% that 60/120 implied. A smaller set
        makes each human label carry MORE of the project's central claim, not less.
        """
        targets = SetTargets()
        assert targets.total == 40
        assert targets.min_human_authored == 30

    def test_every_adversarial_stratum_is_gated(self) -> None:
        """The shortfall loop iterates is_adversarial rather than a hardcoded tuple.

        It used to list three strata by name, so adding stratum 15 would have left
        supersession ungated and the gate silently weaker than it reads.
        """
        empty = GoldenSet(items=())
        gated = {line.split()[0] for line in SetTargets().shortfall(empty)}
        for stratum in Strata:
            if stratum.is_adversarial:
                assert stratum.value in gated, f"{stratum.value} is not gated"
