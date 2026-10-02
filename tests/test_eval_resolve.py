"""Corpus-resolution tests.

The property under test: **a hand-labelled row that cannot be scored must fail while the
labeller is looking at it**, not silently at SpanMatcher time. Every test below corresponds
to a way a well-formed span can still be false.

The most important one is `test_wrong_page_is_caught_by_the_concept_check`. It documents a
real limitation found by testing rather than assumed: the obvious check ("is this clause on
this page?") is **insufficient** for this corpus, because bare numeric clause ids also
match ordinary numbered list items on unrelated pages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from arag.eval.resolve import (
    AMBIGUOUS_CLAUSE_PAGES,
    CorpusIndex,
    Severity,
    resolve_golden_set,
    resolve_item,
)
from arag.eval.schema import GoldenItem, GoldenSet


def make_index(**docs: Any) -> CorpusIndex:
    return CorpusIndex(generated_at="2026-09-04T00:00:00+00:00", documents=dict(docs))


@pytest.fixture
def index() -> CorpusIndex:
    """A miniature corpus mirroring the real one's awkward shape.

    `15` deliberately appears on four pages, because that is what the real
    star-comprehensive-2025 does and it is the reason the naive check fails.
    """
    return make_index(
        **{
            "star-2025": {
                "pages": 48,
                "clauses": {"15": [14, 34, 36, 42], "9": [11], "II": [14]},
                "page_clauses": {"14": ["15", "II"], "11": ["9"]},
                "concepts": {
                    "benefit.bariatric_surgery": {"pages": [14, 15], "clauses": ["15"]},
                    "benefit.organ_donor": {"pages": [11], "clauses": ["9"]},
                },
            },
            "star-2021": {
                "pages": 18,
                "clauses": {"6": [1, 6], "2": [5]},
                "page_clauses": {"6": ["6"], "5": ["2"]},
                "concepts": {"benefit.bariatric_surgery": {"pages": [1, 6], "clauses": ["6"]}},
            },
        }
    )


def item(**overrides: Any) -> GoldenItem:
    base: dict[str, Any] = {
        "id": "t-1",
        "question": "Is weight loss surgery covered and is there a cap?",
        "strata": "table_formula",
        "authored_by": "human",
        "expected_behaviour": "answer",
        "concept_id": "benefit.bariatric_surgery",
        "ground_truth_spans": [{"document_id": "star-2025", "page": 14, "clause_id": "15"}],
        "reference_answer": "Payable subject to the benefit table.",
    }
    return GoldenItem.model_validate(base | overrides)


def errors(findings: list[Any]) -> list[str]:
    return [f.message for f in findings if f.severity is Severity.ERROR]


def warnings(findings: list[Any]) -> list[str]:
    return [f.message for f in findings if f.severity is Severity.WARN]


def unverifiable(findings: list[Any]) -> list[str]:
    return [f.message for f in findings if f.severity is Severity.UNVERIFIABLE]


class TestCorrectSpan:
    def test_a_true_span_produces_no_errors(self, index: CorpusIndex) -> None:
        assert errors(resolve_item(item(), index)) == []


class TestUnknownTargets:
    def test_unknown_document_errors_and_lists_the_known_ones(self, index: CorpusIndex) -> None:
        found = resolve_item(
            item(
                ground_truth_spans=[
                    {"document_id": "carehealth-supreme", "page": 3, "clause_id": "2.1"}
                ]
            ),
            index,
        )
        assert len(errors(found)) == 1
        assert "unknown document" in errors(found)[0]
        assert "star-2025" in errors(found)[0], "the message must name valid options"

    def test_page_beyond_the_document_errors(self, index: CorpusIndex) -> None:
        found = resolve_item(
            item(ground_truth_spans=[{"document_id": "star-2025", "page": 99, "clause_id": "15"}]),
            index,
        )
        assert any("does not exist" in e for e in errors(found))

    def test_clause_absent_from_the_document_errors(self, index: CorpusIndex) -> None:
        found = resolve_item(
            item(
                ground_truth_spans=[{"document_id": "star-2025", "page": 7, "clause_id": "99.99"}]
            ),
            index,
        )
        assert any("not found anywhere" in e for e in errors(found))
        assert any("score zero forever" in e for e in errors(found)), (
            "the message must say WHY this matters, or a labeller will not act on it"
        )

    def test_must_not_cite_pointing_at_nothing_errors(self, index: CorpusIndex) -> None:
        """A supersession trap aimed at a non-existent document tests nothing at all."""
        found = resolve_item(item(must_not_cite=["star-comprehensive-2019"]), index)
        assert any("can never fire" in e for e in errors(found))

    def test_must_not_cite_with_a_clause_suffix_checks_the_document(
        self, index: CorpusIndex
    ) -> None:
        assert errors(resolve_item(item(must_not_cite=["star-2021#6"]), index)) == []


class TestWrongPage:
    def test_unambiguous_clause_on_the_wrong_page_errors(self, index: CorpusIndex) -> None:
        """Clause 9 exists only on page 11, so claiming page 30 is caught directly."""
        found = resolve_item(
            item(
                concept_id="benefit.organ_donor",
                ground_truth_spans=[{"document_id": "star-2025", "page": 30, "clause_id": "9"}],
            ),
            index,
        )
        assert any("NOT on page 30" in e for e in errors(found))

    def test_wrong_page_is_caught_by_the_concept_check(self, index: CorpusIndex) -> None:
        """The limitation this module had to work around.

        Clause "15" legitimately appears on pages [14, 34, 36, 42] of the real document,
        because a line beginning "15." is indistinguishable from a numbered list item. So
        "clause 15 on page 34" passes the clause-on-page check even though the clause is on
        page 14. The concept anchor cross-check is what actually catches it, which is why
        that check is an ERROR and not a warning.
        """
        found = resolve_item(
            item(ground_truth_spans=[{"document_id": "star-2025", "page": 34, "clause_id": "15"}]),
            index,
        )
        clause_errors = [e for e in errors(found) if "NOT on page" in e]
        assert clause_errors == [], "the naive check cannot catch this - that is the point"

        concept_errors = [e for e in errors(found) if "does not appear on page 34" in e]
        assert len(concept_errors) == 1
        assert "[14, 15]" in concept_errors[0], "must show where the concept actually is"

    def test_ambiguous_id_verified_by_concept_is_only_a_warning(self, index: CorpusIndex) -> None:
        """Ambiguity alone must NOT block.

        It fires on plenty of correct labels - including this one, where the concept
        anchors independently confirm page 14. Erroring here would block good work and
        teach the labeller to bypass the tool.
        """
        found = resolve_item(item(), index)
        assert errors(found) == []
        assert unverifiable(found) == []
        assert any("was confirmed independently" in w for w in warnings(found))
        assert len(index.clause_pages("star-2025", "15")) > AMBIGUOUS_CLAUSE_PAGES


class TestUnverifiable:
    """The hole that ERROR and WARN together did not cover.

    A bare numeric clause id resolves to several pages, so the clause check cannot
    locate it. If nothing else can confirm the page either, the label is not shown
    false - it is simply uncheckable, which is indistinguishable from being wrong. That
    must block, and must be reported separately so it cannot be skimmed past.
    """

    def test_ambiguous_id_with_no_concept_blocks(self, index: CorpusIndex) -> None:
        found = resolve_item(item(concept_id=None), index)
        # Blocking includes UNVERIFIABLE, which is the whole point of the third state -
        # `errors()` here filters Severity.ERROR only, so check the property directly.
        assert any(f.severity.blocking for f in found), "an uncheckable label must block"
        assert errors(found) == [], "it is not shown false, only uncheckable"
        assert len(unverifiable(found)) == 1
        assert "no concept_id" in unverifiable(found)[0]
        assert "excl.NN" in unverifiable(found)[0], "the message must name the fix"

    def test_ambiguous_id_with_unanchored_concept_is_unverifiable(self, index: CorpusIndex) -> None:
        """The concept exists but has no anchors in this document, so it cannot
        confirm the page. Ambiguous id + no working cross-check = uncheckable.
        """
        found = resolve_item(item(concept_id="procedure.portability"), index)
        assert len(unverifiable(found)) == 1
        assert "no anchors in this document" in unverifiable(found)[0]

    def test_unverifiable_counts_as_blocking(self, index: CorpusIndex) -> None:
        report = resolve_golden_set(GoldenSet(items=(item(id="u", concept_id=None),)), index)
        assert not report.ok
        assert report.unusable_item_ids == {"u"}
        assert report.hard_errors == []
        assert len(report.unverifiable) == 1

    def test_wrong_page_is_not_also_reported_as_unverifiable(self, index: CorpusIndex) -> None:
        """Noise control. A span already errored for being on the wrong page gains
        nothing from "and also uncheckable", and the UNVERIFIABLE section is only
        useful while it stays short enough to read.
        """
        found = resolve_item(
            item(ground_truth_spans=[{"document_id": "star-2025", "page": 34, "clause_id": "15"}]),
            index,
        )
        assert len(errors(found)) == 1
        assert unverifiable(found) == []

    def test_unambiguous_id_needs_no_cross_check(self, index: CorpusIndex) -> None:
        """Clause 9 sits on exactly one page, so it locates itself and nothing is
        uncheckable even without a concept.
        """
        found = resolve_item(
            item(
                concept_id=None,
                ground_truth_spans=[{"document_id": "star-2025", "page": 11, "clause_id": "9"}],
            ),
            index,
        )
        assert errors(found) == []
        assert unverifiable(found) == []


class TestConceptSeverity:
    def test_concept_absent_from_the_document_only_warns(self, index: CorpusIndex) -> None:
        """A vocabulary gap and a wrong label are indistinguishable to the index, so it
        must not block. Blocking here would push the labeller into fighting the tool.
        """
        found = resolve_item(
            item(
                concept_id="procedure.portability",
                ground_truth_spans=[{"document_id": "star-2025", "page": 14, "clause_id": "15"}],
            ),
            index,
        )
        assert errors(found) == []
        assert any("no anchor match anywhere" in w for w in warnings(found))

    def test_missing_clause_id_warns_about_page_only_matching(self, index: CorpusIndex) -> None:
        found = resolve_item(
            item(ground_truth_spans=[{"document_id": "star-2025", "page": 14}]),
            index,
        )
        assert any("only be matched by page" in w for w in warnings(found))
        assert any("2.7x in pagination" in w for w in warnings(found))


class TestReport:
    def test_unusable_ids_are_collected(self, index: CorpusIndex) -> None:
        good = item(id="good")
        bad = item(
            id="bad",
            ground_truth_spans=[{"document_id": "nope", "page": 1, "clause_id": "1"}],
        )
        report = resolve_golden_set(GoldenSet(items=(good, bad)), index)
        assert report.unusable_item_ids == {"bad"}
        assert not report.ok
        assert report.items_checked == 2

    def test_clean_set_is_ok(self, index: CorpusIndex) -> None:
        report = resolve_golden_set(GoldenSet(items=(item(),)), index)
        assert report.ok
        assert report.index_generated_at


class TestIndexIO:
    def test_load_round_trips(self, tmp_path: Path, index: CorpusIndex) -> None:
        import json

        path = tmp_path / "clause_index.json"
        path.write_text(
            json.dumps({"generated_at": index.generated_at, "documents": index.documents}),
            encoding="utf-8",
        )
        loaded = CorpusIndex.load(path)
        assert loaded.document_ids == {"star-2025", "star-2021"}
        assert loaded.pages("star-2025") == 48
        assert loaded.clause_pages("star-2025", "15") == [14, 34, 36, 42]

    def test_missing_document_returns_empty_rather_than_raising(self, index: CorpusIndex) -> None:
        assert index.pages("ghost") == 0
        assert index.clause_pages("ghost", "1") == []
        assert not index.has_clause("ghost", "1")


class TestRealIndex:
    @pytest.fixture
    def real(self, repo_root: Path) -> CorpusIndex:
        path = repo_root / "data" / "manifest" / "clause_index.json"
        if not path.exists():
            pytest.skip("run `arag-ingest clause-index` first")
        return CorpusIndex.load(path)

    def test_all_six_documents_indexed(self, real: CorpusIndex) -> None:
        assert len(real.document_ids) == 6

    def test_the_bare_clause_id_ambiguity_is_real(self, real: CorpusIndex) -> None:
        """Pins the measurement that forced the concept check to be an ERROR.

        If this ever returns a single page, bare ids became unambiguous and the severity
        split in _check_concept can be revisited.
        """
        pages = real.clause_pages("star-comprehensive-2025", "15")
        assert len(pages) > AMBIGUOUS_CLAUSE_PAGES, (
            "bare numeric clause ids are ambiguous in this corpus; that is why the "
            "concept anchor check carries the weight"
        )
        assert 14 in pages

    def test_the_cross_version_concept_join_resolves(self, real: CorpusIndex) -> None:
        """The §3 answer, as a test.

        The same concept must be locatable in BOTH Star versions - that is what makes
        concept_id a usable join key where clause_id is not.
        """
        from arag.eval.concepts import Concept

        for concept in (
            Concept.BENEFIT_BARIATRIC_SURGERY,
            Concept.LIMIT_ROOM_RENT,
            Concept.WAIT_PRE_EXISTING,
        ):
            old = real.concept_pages("star-comprehensive-2021", concept)
            new = real.concept_pages("star-comprehensive-2025", concept)
            assert old, f"{concept.value} not found in the 2021 wording"
            assert new, f"{concept.value} not found in the 2025 wording"

    def test_clause_ids_do_not_join_across_versions(self, real: CorpusIndex) -> None:
        """The negative half of the §3 answer, also as a test.

        Bariatric surgery is 'Section 6' in 2021 and item '15' in 2025. If a future change
        made these coincide, the concept_id requirement could be relaxed - so the claim is
        pinned rather than left as prose in a doc.
        """
        old_pages = real.concept_pages(
            "star-comprehensive-2021",
            __import__(
                "arag.eval.concepts", fromlist=["Concept"]
            ).Concept.BENEFIT_BARIATRIC_SURGERY,
        )
        assert real.has_clause("star-comprehensive-2021", "6")
        assert 6 in real.clause_pages("star-comprehensive-2021", "6")
        assert 6 in old_pages
        # ...and 6 is not the 2025 identifier for the same benefit.
        assert 14 not in real.clause_pages("star-comprehensive-2021", "6")


class TestIrdaiExclusionCodes:
    """The strongest identifier in this corpus.

    IRDAI mandates standardised exclusion wording with a code, so `Code Excl 02`
    appears verbatim in both Star versions. Unlike a bare numeric id it is
    unambiguous; unlike an insurer's own section numbering it is stable across
    versions and across insurers.
    """

    def test_canonicalisation_accepts_every_written_form(self) -> None:
        from arag.eval.concepts import validate_clause_id

        for written in ("excl.02", "Code Excl 02", "Excl 2", "excl 2", "EXCL.02"):
            assert validate_clause_id(written) == "excl.02", written

    def test_zero_padding_unifies_excl_2_and_excl_02(self) -> None:
        from arag.eval.concepts import validate_clause_id

        assert validate_clause_id("Excl 2") == validate_clause_id("excl.02")

    def test_rejects_non_codes(self) -> None:
        from arag.eval.concepts import validate_clause_id

        for bad in ("excl", "excl.999", "excl.abc"):
            with pytest.raises(ValueError):
                validate_clause_id(bad)

    def test_the_code_joins_both_star_versions(self, repo_root: Path) -> None:
        """The clean supersession case, as a test.

        If this fails, the code-level join has broken and those items must fall back
        to concept_id alone.
        """
        path = repo_root / "data" / "manifest" / "clause_index.json"
        if not path.exists():
            pytest.skip("run `arag-ingest clause-index` first")
        real = CorpusIndex.load(path)
        for code in ("excl.01", "excl.02"):
            old_pages = real.clause_pages("star-comprehensive-2021", code)
            new_pages = real.clause_pages("star-comprehensive-2025", code)
            assert old_pages, f"{code} missing from the 2021 wording"
            assert new_pages, f"{code} missing from the 2025 wording"

    def test_the_code_locates_itself(self, repo_root: Path) -> None:
        path = repo_root / "data" / "manifest" / "clause_index.json"
        if not path.exists():
            pytest.skip("run `arag-ingest clause-index` first")
        real = CorpusIndex.load(path)
        pages = real.clause_pages("star-comprehensive-2025", "excl.02")
        assert len(pages) <= AMBIGUOUS_CLAUSE_PAGES, (
            "excl codes must not trip the ambiguity path, or the clean supersession "
            "case would need a concept cross-check too"
        )
