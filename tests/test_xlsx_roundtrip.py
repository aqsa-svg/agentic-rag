"""The spreadsheet must read back as exactly what was written.

A reader and a writer can each be perfectly self-consistent and still disagree, and nothing
but a round-trip catches it. That happened here on the first run: `item_to_row` correctly
emitted `"excl.02|"` for an item whose second span has no clause, and `_split` discarded the
trailing empty — so this module rejected a sheet it had itself just produced, with a
perfectly accurate error message about positional alignment.

The convention under test: **one row per item, `|`-delimited spans**. Chosen over one row
per span with a repeated id, because a repeated-id sheet detaches its continuation rows the
moment someone sorts by strata — silently, with no error until load — and sorting is how a
spreadsheet is actually used.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arag.eval.schema import GoldenItem, GoldenSet
from arag.eval.xlsx import XlsxRowError, item_to_row, read_xlsx, row_to_item, write_xlsx

ROOT = Path(__file__).resolve().parents[1]


def item(**over: object) -> GoldenItem:
    base: dict[str, object] = {
        "id": "t-01",
        "question": "Is a nursing home with eight beds a hospital for my claim?",
        "strata": "definitional_carveout",
        "authored_by": "human",
        "expected_behaviour": "answer",
        "reference_answer": "Registration suffices; the bed count is an alternative limb.",
        "ground_truth_spans": [
            {"document_id": "star-comprehensive-2025", "page": 4, "clause_id": "def.hospital"}
        ],
    }
    base.update(over)
    return GoldenItem(**base)


class TestRoundTrip:
    def test_a_single_span_item_survives(self) -> None:
        original = item()
        back = row_to_item(item_to_row(original), row_number=2)
        assert back is not None
        assert back.model_dump(mode="json") == original.model_dump(mode="json")

    def test_a_span_with_no_clause_keeps_its_slot(self) -> None:
        """The defect this file was written for.

        h-02 has two spans and only the first has a clause. The writer emits `"excl.02|"`;
        a reader that discards empty parts turns two spans into one and silently re-pairs
        every span after the gap.
        """
        original = item(
            ground_truth_spans=[
                {"document_id": "star-comprehensive-2025", "page": 31, "clause_id": "excl.02"},
                {"document_id": "star-comprehensive-2025", "page": 32, "clause_id": None},
            ]
        )
        row = item_to_row(original)
        assert row["clause_id"] == "excl.02|", "the empty slot must be written"
        back = row_to_item(row, row_number=2)
        assert back is not None
        assert len(back.ground_truth_spans) == 2
        assert back.ground_truth_spans[1].clause_id is None
        assert back.ground_truth_spans[1].page == 32

    def test_a_multi_span_item_with_must_not_cite_survives(self) -> None:
        original = item(
            strata="supersession",
            concept_id="exclusion.permanent",
            must_not_cite=["star-comprehensive-2021"],
            ground_truth_spans=[
                {"document_id": "star-comprehensive-2025", "page": 33, "clause_id": "excl.06"}
            ],
        )
        back = row_to_item(item_to_row(original), row_number=2)
        assert back is not None
        assert back.model_dump(mode="json") == original.model_dump(mode="json")

    def test_the_shipped_workbook_round_trips_to_the_golden_set(self) -> None:
        """End to end on the real files, not a fixture.

        If data/golden/v1.xlsx and data/golden/v1.jsonl ever disagree, one of them has been
        edited without the other and the labelling workflow has two sources of truth.
        """
        workbook = ROOT / "data" / "golden" / "v1.xlsx"
        golden = ROOT / "data" / "golden" / "v1.jsonl"
        if not workbook.exists():
            pytest.skip("workbook not generated")

        items, problems = read_xlsx(workbook)
        assert not problems, problems
        from_file = {i.id: i.model_dump(mode="json") for i in GoldenSet.load(golden).items}
        from_sheet = {i.id: i.model_dump(mode="json") for i in items}
        assert from_sheet == from_file


class TestNotStartedIsNotMalformed:
    """The 33 pre-filled rows must not block an import.

    Each has an id, a stratum and an expected behaviour, and no question - because the
    question is the labeller's to write. All of them fail schema validation, correctly.
    Treating that as an error would mean nothing imports until all 40 are finished, which
    is the opposite of what incremental labelling needs.
    """

    def test_a_row_with_no_question_is_skipped(self) -> None:
        row = {
            "id": "h-33",
            "strata": "supersession",
            "expected_behaviour": "answer",
            "question": "",
        }
        assert row_to_item(row, row_number=9) is None

    def test_a_row_with_no_id_is_skipped(self) -> None:
        assert row_to_item({"id": "", "question": "stray text"}, row_number=9) is None

    def test_a_STARTED_row_that_is_wrong_still_fails(self) -> None:
        """Having a question is the marker of intent; from there, errors are real."""
        with pytest.raises(XlsxRowError, match="h-33"):
            row_to_item(
                {
                    "id": "h-33",
                    "strata": "supersession",
                    "expected_behaviour": "answer",
                    "question": "Is treatment for sleep apnea covered under my policy?",
                    # no reference_answer, no span: the schema refuses it
                },
                row_number=9,
            )


class TestSpanColumnAlignment:
    """The cost of one-row-per-item, made loud rather than silent."""

    def test_mismatched_clause_count_is_refused(self) -> None:
        with pytest.raises(XlsxRowError, match="positional"):
            row_to_item(
                {
                    "id": "t-01",
                    "question": "a question long enough to pass the schema's length check",
                    "strata": "definitional_carveout",
                    "expected_behaviour": "answer",
                    "reference_answer": "an answer",
                    "document_id": "a|b",
                    "page": "1|2",
                    "clause_id": "x",
                },
                row_number=4,
            )

    def test_mismatched_page_count_is_refused(self) -> None:
        with pytest.raises(XlsxRowError, match="one to one"):
            row_to_item(
                {
                    "id": "t-01",
                    "question": "a question long enough to pass the schema's length check",
                    "strata": "definitional_carveout",
                    "expected_behaviour": "answer",
                    "reference_answer": "an answer",
                    "document_id": "a|b",
                    "page": "1",
                    "clause_id": "x|y",
                },
                row_number=4,
            )

    def test_an_entirely_empty_clause_column_is_allowed(self) -> None:
        """Page-only spans are legal; the schema warns about them rather than refusing."""
        back = row_to_item(
            {
                "id": "t-01",
                "question": "a question long enough to pass the schema's length check",
                "strata": "definitional_carveout",
                "expected_behaviour": "answer",
                "reference_answer": "an answer",
                "document_id": "star-comprehensive-2025|star-comprehensive-2025",
                "page": "31|32",
                "clause_id": "",
            },
            row_number=4,
        )
        assert back is not None
        assert [s.clause_id for s in back.ground_truth_spans] == [None, None]

    def test_a_non_numeric_page_is_refused(self) -> None:
        with pytest.raises(XlsxRowError, match="not a number"):
            row_to_item(
                {
                    "id": "t-01",
                    "question": "a question long enough to pass the schema's length check",
                    "strata": "definitional_carveout",
                    "expected_behaviour": "answer",
                    "reference_answer": "an answer",
                    "document_id": "a",
                    "page": "page four",
                    "clause_id": "x",
                },
                row_number=4,
            )


class TestWriteThenRead:
    def test_a_written_workbook_reads_back_identically(self, tmp_path: Path) -> None:
        items = [item(), item(id="t-02", strata="procedural")]
        path = write_xlsx(
            tmp_path / "w.xlsx", items, pending=[{"id": "t-03", "strata": "procedural"}]
        )
        back, problems = read_xlsx(path)
        assert not problems
        assert [i.model_dump(mode="json") for i in back] == [
            i.model_dump(mode="json") for i in items
        ], "the pending row must be skipped, the complete ones must survive"
