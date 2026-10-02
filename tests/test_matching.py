"""Span matching tests.

``SpanMatcher.matches`` is the predicate underneath every retrieval metric, so a subtle
bug here would shift every published number without any other test noticing. The
dot-boundary case below is the specific bug this module was written to prevent.
"""

from __future__ import annotations

import pytest

from arag.agent.types import Citation
from arag.eval.matching import MatchMode, SpanMatcher, violates_must_not_cite
from arag.retrieval.types import DocSpan
from tests.conftest import chunk, truth


class TestClausePrefix:
    def test_parent_clause_covers_child(self) -> None:
        """A chunk covering clause 4.2 does contain ground truth 4.2.b."""
        m = SpanMatcher()
        assert m.matches(truth("d1", "4.2"), truth("d1", "4.2.b"))

    def test_child_satisfies_parent_truth(self) -> None:
        m = SpanMatcher()
        assert m.matches(truth("d1", "4.2.b"), truth("d1", "4.2"))

    def test_dot_boundary_prevents_false_credit(self) -> None:
        """Clause 4.21 is NOT inside clause 4.2.

        A naive ``startswith`` would credit this and inflate recall on any document whose
        numbering reaches double digits — which every policy wording does. This assertion
        is the reason the boundary check exists.
        """
        m = SpanMatcher()
        assert not m.matches(truth("d1", "4.21"), truth("d1", "4.2"))
        assert not m.matches(truth("d1", "4.2"), truth("d1", "4.21"))

    def test_different_documents_never_match(self) -> None:
        m = SpanMatcher()
        assert not m.matches(truth("star", "4.2"), truth("nivabupa", "4.2"))

    def test_clause_normalisation_is_case_and_dot_insensitive(self) -> None:
        m = SpanMatcher()
        assert m.matches(truth("d1", "4.2.B"), truth("d1", "4.2.b."))


class TestPageFallback:
    def test_falls_back_when_clause_missing(self) -> None:
        """Clause parsing fails on scanned annexures, so page labels must still score."""
        m = SpanMatcher()
        assert m.matches(DocSpan(document_id="d1", page=34), truth("d1", None, 34))

    def test_neighbouring_page_does_not_match_by_default(self) -> None:
        """page_tolerance defaults to 0 because widening it hides chunking defects."""
        m = SpanMatcher()
        assert not m.matches(DocSpan(document_id="d1", page=33), truth("d1", None, 34))

    def test_tolerance_can_be_widened_deliberately(self) -> None:
        m = SpanMatcher(page_tolerance=1)
        assert m.matches(DocSpan(document_id="d1", page=33), truth("d1", None, 34))

    def test_negative_tolerance_rejected(self) -> None:
        with pytest.raises(ValueError, match="page_tolerance"):
            SpanMatcher(page_tolerance=-1)

    def test_unusable_label_refuses_to_guess(self) -> None:
        """Neither clause nor page on one side: scoring is impossible.

        Returning True here would manufacture recall out of a badly-labelled item, so the
        matcher returns False and the bad label shows up as a miss to be investigated.
        """
        m = SpanMatcher()
        assert not m.matches(DocSpan(document_id="d1"), truth("d1", None, 34))


class TestModes:
    def test_exact_mode_rejects_parent_clause(self) -> None:
        m = SpanMatcher(mode=MatchMode.EXACT)
        assert not m.matches(truth("d1", "4.2"), truth("d1", "4.2.b"))
        assert m.matches(truth("d1", "4.2.b"), truth("d1", "4.2.b"))

    def test_page_mode_ignores_clauses_entirely(self) -> None:
        m = SpanMatcher(mode=MatchMode.PAGE)
        assert m.matches(truth("d1", "9.9", 34), truth("d1", "4.2", 34))


class TestCollectionHelpers:
    def test_relevance_vector_preserves_rank_order(self) -> None:
        m = SpanMatcher()
        chunks = [chunk("d1", "9.9"), chunk("d1", "4.2"), chunk("d1", "8.8")]
        assert m.relevance_vector(chunks, (truth("d1", "4.2"),)) == [0, 1, 0]

    def test_hit_truths_returns_distinct_truth_indices(self) -> None:
        m = SpanMatcher()
        chunks = [chunk("d1", "4.2", cid="a"), chunk("d1", "4.2", cid="b")]
        truths = (truth("d1", "4.2"), truth("d1", "2.1"))
        assert m.hit_truths(chunks, truths) == {0}


class TestMustNotCite:
    def _cite(self, doc: str, clause: str | None) -> Citation:
        return Citation(span=DocSpan(document_id=doc, clause_id=clause), chunk_id="c1")

    def test_document_level_pattern_blocks_any_clause(self) -> None:
        hits = violates_must_not_cite(
            (self._cite("star-comprehensive-2021", "4.1"),), ("star-comprehensive-2021",)
        )
        assert hits == ["star-comprehensive-2021"]

    def test_clause_level_pattern_blocks_descendants(self) -> None:
        """The supersession trap: citing 4.2.b when 4.2 was replaced is still a violation."""
        hits = violates_must_not_cite(
            (self._cite("star-comprehensive-2021", "4.2.b"),),
            ("star-comprehensive-2021#4.2",),
        )
        assert hits == ["star-comprehensive-2021#4.2"]

    def test_sibling_clause_is_not_a_violation(self) -> None:
        hits = violates_must_not_cite(
            (self._cite("star-comprehensive-2021", "4.3"),),
            ("star-comprehensive-2021#4.2",),
        )
        assert hits == []

    def test_current_version_is_allowed(self) -> None:
        hits = violates_must_not_cite(
            (self._cite("star-comprehensive-2024", "4.2"),),
            ("star-comprehensive-2021#4.2",),
        )
        assert hits == []

    def test_no_citations_no_violation(self) -> None:
        assert violates_must_not_cite((), ("star-comprehensive-2021",)) == []
