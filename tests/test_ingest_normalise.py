"""Normalisation tests.

Two properties matter most and both are asserted against the real corpus, not just
synthetic input:

* the ligature fix actually makes the corpus lexically matchable, and
* the numeric-invariance guard actually fires on the hazard it is built for.

The second needs synthetic input, because spike S5 measured zero occurrences of the
footnote hazard in the current corpus. A guard that has never been triggered is a guard
nobody has tested, so the hazard is constructed here deliberately.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest

from arag.ingest.normalise import (
    NBSP,
    SOFT_HYPHEN,
    NormalisationStats,
    NumericCorruptionError,
    normalise_query,
    normalise_text,
    numeric_spans,
    strip_footnote_markers,
)

FI = "ﬁ"  # LATIN SMALL LIGATURE FI
FFI = "ﬃ"
SUP1 = "¹"  # SUPERSCRIPT ONE
SUP2 = "²"


class TestLigatures:
    """The measured defect: 269 codepoints, 254 in star-comprehensive-2021."""

    def test_the_problem_restated_as_an_assertion(self) -> None:
        """Why this module exists at all. A ligature is not its component letters."""
        contaminated = f"bene{FI}t"
        assert contaminated != "benefit"
        assert "fi" not in contaminated
        assert len(contaminated) == 6
        assert len("benefit") == 7

    def test_ligature_is_folded(self) -> None:
        out, stats = normalise_text(f"The bene{FI}t is speci{FI}ed.")
        assert out == "The benefit is specified."
        assert stats.ligatures_folded == 2

    def test_three_letter_ligature_is_folded(self) -> None:
        out, _ = normalise_text(f"o{FFI}ce")
        assert out == "office"

    def test_clean_text_is_unchanged(self) -> None:
        text = "The benefit is specified herein."
        out, stats = normalise_text(text)
        assert out == text
        assert stats.total_changes == 0


class TestNumericInvarianceGuard:
    """N2. The guard has never fired on real data; these tests make sure it can."""

    def test_footnote_marker_after_a_monetary_value_is_stripped(self) -> None:
        """The exact hazard: '5,00,000/-<sup>1</sup>' must not become '5,00,0001'."""
        text = f"limit of Rs.5,00,000/-{SUP1} per policy period"
        out, stats = normalise_text(text)
        assert stats.footnote_markers_stripped == 1
        assert "5,00,0001" not in out
        assert "5,00,000" in out
        assert numeric_spans(out) == ["500000"]

    def test_guard_raises_when_a_numeric_value_would_change(self) -> None:
        """Prove the guard is real by bypassing the strip step that protects against it.

        strip_footnote_markers only removes a marker that FOLLOWS a digit or ) / -. A
        marker directly preceding a digit survives the strip, gets folded by NFKC, and
        must then be caught by the invariance check.
        """
        text = f"the limit is {SUP1}250000 rupees"
        with pytest.raises(NumericCorruptionError) as exc:
            normalise_text(text, context="synthetic-page-1")
        assert "synthetic-page-1" in str(exc.value)
        assert "footnote-marker fusion" in str(exc.value)

    def test_guard_message_names_the_changed_value(self) -> None:
        with pytest.raises(NumericCorruptionError) as exc:
            normalise_text(f"cap {SUP2}50000")
        assert "250000" in str(exc.value) or "50000" in str(exc.value)

    def test_guard_can_be_disabled_for_queries(self) -> None:
        """A user's typing must never be able to hard-fail their own request."""
        out = normalise_query(f"is {SUP1}250000 the cap?")
        assert out

    def test_thousands_separators_are_not_treated_as_corruption(self) -> None:
        """Commas are stripped before comparison, so a grouping change is not corruption
        while a changed VALUE still is. Without this the guard would fire on every page.
        """
        assert numeric_spans("Rs.5,00,000") == ["500000"]
        assert numeric_spans("Rs.500000") == ["500000"]

    def test_superscript_in_prose_is_preserved(self) -> None:
        """A conservative strip: 'm<sup>2</sup>' in a hospital-area definition is real
        content, not a footnote reference. Stripping every superscript unconditionally
        would silently destroy it.
        """
        out, count = strip_footnote_markers(f"area of 50 m{SUP2} per bed")
        assert count == 0, "a superscript after a letter is not a footnote marker"
        assert SUP2 in out

    def test_marker_after_closing_paren_is_stripped(self) -> None:
        out, count = strip_footnote_markers(f"(as amended){SUP1} applies")
        assert count == 1
        assert SUP1 not in out


class TestInvisibleCharacters:
    def test_soft_hyphen_is_removed(self) -> None:
        out, stats = normalise_text(f"hospi{SOFT_HYPHEN}talisation")
        assert out == "hospitalisation"
        assert stats.soft_hyphens_removed == 1

    def test_nbsp_becomes_a_space(self) -> None:
        out, stats = normalise_text(f"Rs.5,00,000{NBSP}per{NBSP}year")
        assert NBSP not in out
        assert out == "Rs.5,00,000 per year"
        assert stats.nbsp_folded == 2

    def test_zero_width_characters_are_removed(self) -> None:
        out, stats = normalise_text("bene​fit")
        assert out == "benefit"
        assert stats.zero_width_removed == 1


class TestPunctuationFolding:
    def test_curly_quotes_are_folded(self) -> None:
        """'policy's' and 'policy’s' are different BM25 tokens. Folding is free recall."""
        out, stats = normalise_text("the policy’s terms and “Insured Person”")
        assert out == 'the policy\'s terms and "Insured Person"'
        assert stats.punctuation_folded == 3

    def test_dashes_are_folded(self) -> None:
        out, _ = normalise_text("30–90 days")
        assert out == "30-90 days"

    def test_folding_can_be_disabled(self) -> None:
        out, _ = normalise_text("policy’s", fold_punctuation=False)
        assert "’" in out


class TestQueryPath:
    """N1: the query path must use the same normalisation as the index path."""

    def test_query_normalisation_matches_index_normalisation(self) -> None:
        """The property that makes N1 true.

        A user pasting a phrase out of the PDF submits U+FB01 too. If the query path
        normalised differently, retrieval would return nothing for that term and it would
        look like a relevance problem rather than an encoding one.
        """
        pasted = f"what is the bene{FI}t limit"
        indexed, _ = normalise_text(pasted)
        assert normalise_query(pasted) == indexed.strip()
        assert "benefit" in normalise_query(pasted)

    def test_query_is_stripped(self) -> None:
        assert normalise_query("  room rent limit  ") == "room rent limit"

    def test_query_path_is_the_same_function(self) -> None:
        """Asserted structurally, not by eyeball: two implementations of normalisation is
        how the index and the query silently drift apart.
        """
        import inspect

        from arag.ingest import normalise as mod

        source = inspect.getsource(mod.normalise_query)
        assert "normalise_text(" in source


class TestStats:
    def test_merge_accumulates(self) -> None:
        a = NormalisationStats(ligatures_folded=3, chars_in=10, chars_out=9)
        b = NormalisationStats(ligatures_folded=2, nbsp_folded=1, chars_in=5, chars_out=5)
        a.merge(b)
        assert a.ligatures_folded == 5
        assert a.nbsp_folded == 1
        assert a.chars_in == 15

    def test_empty_text_is_a_noop(self) -> None:
        out, stats = normalise_text("")
        assert out == ""
        assert stats.total_changes == 0


class TestAgainstRealCorpus:
    """Normalisation must actually fix the measured corpus defect."""

    @pytest.fixture
    def probe_report(self, repo_root: Path) -> list[dict[str, object]]:
        path = repo_root / "data" / "manifest" / "probe_report.json"
        if not path.exists():
            pytest.skip("corpus not probed; run `arag-ingest fetch && arag-ingest probe`")
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]

    def test_the_corpus_defect_is_real_and_measured(
        self, probe_report: list[dict[str, object]]
    ) -> None:
        star_2021 = next(d for d in probe_report if d["source_id"] == "star-comprehensive-2021")
        assert star_2021["ligature_chars"] >= 200

    def test_normalisation_clears_every_ligature_in_the_corpus(self, repo_root: Path) -> None:
        """End-to-end on the real PDF, not a synthetic fixture."""
        pymupdf = pytest.importorskip("pymupdf")
        path = repo_root / "data" / "raw" / "star-comprehensive-2021.pdf"
        if not path.exists():
            pytest.skip("corpus not fetched")

        from arag.ingest.probe import LIGATURES

        doc = pymupdf.open(path)
        total_before = 0
        with doc:
            for page in doc:
                raw = page.get_text("text")
                total_before += len(LIGATURES.findall(raw))
                cleaned, _ = normalise_text(raw, context=f"star-2021-p{page.number}")
                assert LIGATURES.findall(cleaned) == [], "a ligature survived normalisation"
        assert total_before >= 200, "the defect this test guards must still be present"

    def test_the_headline_query_term_becomes_matchable(self, repo_root: Path) -> None:
        """The concrete payoff: 'benefit' is findable in the 2021 wording after
        normalisation, and was not before.
        """
        pymupdf = pytest.importorskip("pymupdf")
        path = repo_root / "data" / "raw" / "star-comprehensive-2021.pdf"
        if not path.exists():
            pytest.skip("corpus not fetched")

        doc = pymupdf.open(path)
        with doc:
            raw = "\n".join(page.get_text("text") for page in doc)
        cleaned, stats = normalise_text(raw, context="star-2021-full")

        raw_hits = raw.lower().count("benefit")
        clean_hits = cleaned.lower().count("benefit")
        assert clean_hits > raw_hits, (
            f"normalisation must expose ligature-hidden occurrences: {raw_hits} -> {clean_hits}"
        )
        assert stats.ligatures_folded >= 200

    def test_no_numeric_corruption_across_the_whole_corpus(self, repo_root: Path) -> None:
        """The N2 guard must pass on every real page.

        This is the assertion that would have caught the hazard had it existed, and it is
        why the guard can be trusted going forward rather than only today.
        """
        pymupdf = pytest.importorskip("pymupdf")
        raw_dir = repo_root / "data" / "raw"
        if not raw_dir.exists() or not any(raw_dir.glob("*.pdf")):
            pytest.skip("corpus not fetched")

        for pdf in sorted(raw_dir.glob("*.pdf")):
            doc = pymupdf.open(pdf)
            with doc:
                for page in doc:
                    # Raises NumericCorruptionError on failure, which fails the test with
                    # the offending value named.
                    normalise_text(
                        page.get_text("text"),
                        context=f"{pdf.stem} p{page.number + 1}",
                    )

    def test_nfkc_is_what_does_the_work(self) -> None:
        """Pin the mechanism, so a future refactor cannot replace NFKC with something
        that looks equivalent and is not (NFC, for instance, does NOT fold ligatures).
        """
        assert unicodedata.normalize("NFKC", f"bene{FI}t") == "benefit"
        assert unicodedata.normalize("NFC", f"bene{FI}t") != "benefit"
