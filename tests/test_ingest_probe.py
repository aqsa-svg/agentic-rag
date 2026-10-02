"""Probe tests, built on synthetic PDFs plus the real corpus.

Synthetic PDFs are generated in-memory so the unit tests are deterministic and run without
network access. The real corpus is then asserted against the findings that were verified by
hand, which is what stops a future refactor from silently reintroducing any of the three
heuristics this probe got wrong on the first attempt:

1. space ratio flagged a clean table as corrupt
2. two-column layout was reported as damage when it extracts fine
3. table cell order was reported as prose column splicing

Each has a regression test below.
"""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from arag.ingest.probe import (
    FUSED_TOKEN_LEN,
    LIGATURES,
    DocumentProbe,
    PageProbe,
    _compact,
    probe_document,
)

pytestmark = pytest.mark.filterwarnings("ignore")


# --------------------------------------------------------------------------------------
# synthetic PDF builders
# --------------------------------------------------------------------------------------


def _write(tmp_path: Path, name: str, build) -> Path:  # type: ignore[no-untyped-def]
    doc = pymupdf.open()
    build(doc)
    path = tmp_path / name
    doc.save(path)
    doc.close()
    return path


def _prose(text: str, repeat: int = 12) -> str:
    return "\n".join(text for _ in range(repeat))


class TestCompact:
    def test_collapses_runs_into_ranges(self) -> None:
        assert _compact([1, 2, 3, 7, 9, 10]) == "1-3, 7, 9-10"

    def test_single_page(self) -> None:
        assert _compact([5]) == "5"

    def test_empty(self) -> None:
        assert _compact([]) == "-"

    def test_truncates_a_long_list(self) -> None:
        assert "more" in _compact(list(range(1, 60, 2)), limit=3)


class TestUnreadable:
    def test_missing_file_is_recorded_not_raised(self, tmp_path: Path) -> None:
        """A corpus scan must survive one bad document."""
        result = probe_document("ghost", tmp_path / "nope.pdf")
        assert not result.ok
        assert result.error
        assert result.breakages()[0].startswith("UNREADABLE")

    def test_garbage_file_is_recorded_not_raised(self, tmp_path: Path) -> None:
        path = tmp_path / "junk.pdf"
        path.write_bytes(b"this is not a pdf at all")
        result = probe_document("junk", path)
        assert not result.ok


class TestTextLayer:
    def test_blank_page_has_no_text_layer(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "blank.pdf", lambda d: d.new_page())
        result = probe_document("blank", path)
        assert result.pages_without_text == [1]
        assert result.total_chars == 0

    def test_prose_page_has_a_text_layer(self, tmp_path: Path) -> None:
        def build(doc: pymupdf.Document) -> None:
            page = doc.new_page()
            page.insert_text((72, 100), _prose("The insured person shall be covered."), fontsize=11)

        result = probe_document("prose", _write(tmp_path, "prose.pdf", build))
        assert result.pages_without_text == []
        assert result.total_chars > 200


class TestFusedTokenRegression:
    """Regression for correction 1: space ratio must not be the fusion signal.

    A table extracts one short cell per line, giving a low space ratio and perfectly clean
    text. The original heuristic called that corruption.
    """

    def test_table_like_short_lines_are_not_flagged_as_fused(self, tmp_path: Path) -> None:
        def build(doc: pymupdf.Document) -> None:
            page = doc.new_page()
            y = 90
            for row in ("Month 1", "7.00%", "10.40%", "12.30%") * 8:
                page.insert_text((72, y), row, fontsize=10)
                y += 14

        result = probe_document("tablish", _write(tmp_path, "tablish.pdf", build))
        page = result.page_probes[0]
        assert page.space_ratio < 0.10, "short cells genuinely produce a low space ratio"
        assert not page.fused_text, "low space ratio alone must NOT mean corrupt"
        assert result.fused_text_pages == []

    def test_genuinely_fused_tokens_are_flagged(self, tmp_path: Path) -> None:
        long_token = "Wheretheinsuredpersonhasbeencontinuouslycovered" * 2
        assert len(long_token) > FUSED_TOKEN_LEN

        def build(doc: pymupdf.Document) -> None:
            page = doc.new_page()
            y = 90
            for _ in range(20):
                page.insert_text((40, y), long_token, fontsize=8)
                y += 14

        result = probe_document("fused", _write(tmp_path, "fused.pdf", build))
        assert result.fused_text_pages == [1]
        assert result.page_probes[0].long_token_share > 0.5


class TestLigatures:
    """Ligature detection — the highest-impact real defect in this corpus.

    NOTE on why this is not tested through a synthetic PDF: PyMuPDF's built-in base-14
    fonts cannot render U+FB01, and silently substitute it (verified: inserting
    "beneﬁt" with Helvetica extracts back as "bene·t"). A synthetic fixture
    therefore cannot reproduce the defect. The detection logic is unit-tested directly
    here and end-to-end against the real corpus in TestRealCorpusFindings, which is where
    254 genuine occurrences live.
    """

    def test_regex_matches_the_ligature_block(self) -> None:
        # U+FB00..U+FB06: ff fi fl ffi ffl ft st
        assert LIGATURES.findall("beneﬁt and oﬃce and ﬂow") == [
            "ﬁ",
            "ﬃ",
            "ﬂ",
        ]

    def test_regex_ignores_clean_ascii(self) -> None:
        assert LIGATURES.findall("benefit and office and flow") == []

    def test_ligature_is_not_the_same_string_as_its_letters(self) -> None:
        """The whole reason this matters, asserted so nobody dismisses it as cosmetic.

        A BM25 index built on the raw extraction cannot match a query for "benefit",
        because the indexed token is a different string.
        """
        assert "beneﬁt" != "benefit"
        assert "fi" not in "beneﬁt"
        # ...and NFKC normalisation is the fix ingest must apply.
        import unicodedata

        assert unicodedata.normalize("NFKC", "beneﬁt") == "benefit"

    def test_page_probe_counts_and_reports_ligatures(self) -> None:
        probe = DocumentProbe(source_id="x", path=Path("x.pdf"), pages=1)
        page = PageProbe(number=1, chars=500, has_text_layer=True)
        page.ligature_chars = 12
        probe.page_probes = [page]

        assert probe.ligature_chars == 12
        assert probe.ligature_pages == [1]
        assert page.ligature_contaminated
        assert page.suspect, "ligature contamination must mark the page as suspect"
        assert any("LIGATURE CONTAMINATION" in b for b in probe.breakages())

    def test_clean_document_reports_no_ligature_breakage(self, tmp_path: Path) -> None:
        def build(doc: pymupdf.Document) -> None:
            page = doc.new_page()
            page.insert_text((72, 100), _prose("The benefit is specified herein."), fontsize=11)

        result = probe_document("clean", _write(tmp_path, "clean.pdf", build))
        assert result.ligature_chars == 0
        assert result.ligature_pages == []
        assert not any("LIGATURE" in b for b in result.breakages())


class TestBlankVersusCover:
    def test_cover_is_not_double_reported_as_empty(self, tmp_path: Path) -> None:
        """Reporting one page as both "cover excluded from OCR" and "empty page" is two
        lines of noise describing one benign fact.
        """
        probe = DocumentProbe(source_id="x", path=tmp_path / "x.pdf", pages=1)
        cover = PageProbe(number=1, chars=15, image_area_ratio=1.0, image_dominant=True)
        cover.likely_cover = True
        probe.page_probes = [cover]

        assert probe.cover_pages == [1]
        assert probe.pages_needing_ocr == [], "a cover must never be sent to OCR"
        assert probe.blank_pages == [], "a cover must not also be reported as empty"


# --------------------------------------------------------------------------------------
# real corpus assertions — skipped when the corpus has not been fetched
# --------------------------------------------------------------------------------------


@pytest.fixture
def probe_report(repo_root: Path) -> list[dict[str, object]]:
    path = repo_root / "data" / "manifest" / "probe_report.json"
    if not path.exists():
        pytest.skip("corpus not probed; run `arag-ingest fetch && arag-ingest probe`")
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _doc(report: list[dict[str, object]], source_id: str) -> dict[str, object]:
    for row in report:
        if row["source_id"] == source_id:
            return row
    pytest.skip(f"{source_id} not in probe report")


class TestRealCorpusFindings:
    """Locks in the numbers that were verified by hand against the PDFs."""

    def test_two_column_layout_is_not_treated_as_damage(
        self, probe_report: list[dict[str, object]]
    ) -> None:
        """Regression for correction 2.

        star-comprehensive-2025 has 32 two-column pages and every one extracts in correct
        reading order. If this ever reports splicing again, the detector has regressed.
        """
        doc = _doc(probe_report, "star-comprehensive-2025")
        assert len(doc["two_column_pages"]) > 20  # type: ignore[arg-type]
        assert doc["reading_order_broken_pages"] == []

    def test_table_cell_order_is_not_treated_as_prose_splicing(
        self, probe_report: list[dict[str, object]]
    ) -> None:
        """Regression for correction 3.

        Every Niva Bupa and IRDAI-annexure page that alternates columns does so only
        inside table regions. 18 of 22 originally-flagged pages were this.
        """
        for source_id in ("nivabupa-reassure2", "nivabupa-rise", "irdai-annexure-2024"):
            doc = _doc(probe_report, source_id)
            assert doc["reading_order_broken_pages"] == [], (
                f"{source_id}: table cell order is being misread as prose splicing again"
            )

    def test_the_one_genuinely_spliced_document_is_still_detected(
        self, probe_report: list[dict[str, object]]
    ) -> None:
        """The corrections must not have made the detector blind.

        star-comprehensive-2021 has 4 genuinely spliced prose pages outside table regions.
        """
        doc = _doc(probe_report, "star-comprehensive-2021")
        assert doc["reading_order_broken_pages"] == [5, 7, 8, 9]

    def test_ligature_contamination_is_detected_in_the_older_star_wording(
        self, probe_report: list[dict[str, object]]
    ) -> None:
        doc = _doc(probe_report, "star-comprehensive-2021")
        assert doc["ligature_chars"] >= 200
        assert any("LIGATURE" in b for b in doc["breakages"])  # type: ignore[union-attr]

    def test_the_corpus_needs_no_ocr(self, probe_report: list[dict[str, object]]) -> None:
        """A finding that changes the build plan: every page has a usable text layer, so
        the VLM OCR stage in DESIGN section 4 is not needed for this corpus.
        """
        assert all(doc["pages_needing_ocr"] == [] for doc in probe_report)

    def test_the_corpus_is_english_only(self, probe_report: list[dict[str, object]]) -> None:
        """DESIGN section 13 assumption 2, now measured rather than assumed."""
        assert all(doc["devanagari_chars"] == 0 for doc in probe_report)

    def test_tables_are_pervasive(self, probe_report: list[dict[str, object]]) -> None:
        total = sum(doc["table_candidates"] for doc in probe_report)  # type: ignore[misc]
        assert total > 100, "table handling is the dominant ingest problem in this corpus"

    def test_heading_grammar_differs_between_documents(
        self, probe_report: list[dict[str, object]]
    ) -> None:
        """Why a single heading regex cannot work.

        The two Star wordings of the SAME product disagree on heading style, so the
        chunker needs a per-document strategy chosen by measurement.
        """
        patterns = {doc["source_id"]: doc["dominant_heading_pattern"] for doc in probe_report}
        assert len(set(patterns.values())) > 1
        assert patterns["star-comprehensive-2025"] != patterns["star-comprehensive-2021"]
