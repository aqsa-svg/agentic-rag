"""`import-xlsx` must write only after validation passes.

The command's docstring has always promised "nothing is written when one row is unusable,
because a partial import leaves the golden set in a state nobody chose". For its first
version that promise was only prose: the JSONL was written, THEN validated, so an import
of five merged rows wrote 10 items to disk and afterwards reported three of them UNUSABLE
with exit 2. The file was left holding items the validator had just rejected.

Pattern 1 from docs/SILENT_WRONGNESS.md - code that is wrong - and the reason the fix
needed a test rather than a comment: a doc comment cannot fail a build.

Resolution runs against the committed data/manifest/clause_index.json, not the PDFs, so
these are ordinary PR-gate tests. The bad item is bad in a way ONLY resolution can see
(well-formed schema, clause on the wrong page), because that is the failure that got past
the original ordering - a schema-level failure was already refused before any write.

Mutation-checked: setting `staged = target` in the command (the defect, restored) fails
the byte-identical test and the positive control. The other three keep passing under that
mutation - they constrain the cleanup, not the ordering - which is worth knowing before
anyone reads six green tests as six independent guards.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from arag.eval.cli import app
from arag.eval.schema import GoldenItem
from arag.eval.xlsx import write_xlsx

runner = CliRunner()

# A span the committed clause index confirms: star-comprehensive-2025 p33 carries excl.08,
# and the exclusion.permanent anchors match that page.
RESOLVABLE = {
    "document_id": "star-comprehensive-2025",
    "page": 33,
    "clause_id": "excl.08",
}
PRIOR = '# a header line that must survive\n{"id": "x-01"}\n'


def book(tmp_path: Path, *, page: int) -> Path:
    item = GoldenItem(
        id="t-01",
        question="Is a tummy tuck after an accident covered, or is it cosmetic surgery?",
        strata="definitional_carveout",
        authored_by="human",
        expected_behaviour="answer",
        reference_answer="Reconstruction following an accident is carved out of excl.08.",
        concept_id="exclusion.permanent",
        ground_truth_spans=[{**RESOLVABLE, "page": page}],
    )
    return write_xlsx(tmp_path / f"p{page}.xlsx", [item], pending=[])


class TestNothingIsWrittenUntilValidationPasses:
    def test_a_failing_import_leaves_the_golden_set_byte_identical(self, tmp_path: Path) -> None:
        golden = tmp_path / "v.jsonl"
        golden.write_text(PRIOR, encoding="utf-8")
        before = golden.read_bytes()

        # Page 1 is the cover page: excl.08 is not on it and the concept anchors do not
        # match it, so resolution rejects the span.
        result = runner.invoke(
            app, ["import-xlsx", str(book(tmp_path, page=1)), "--golden", str(golden)]
        )

        assert result.exit_code == 2, result.stdout
        assert "NOTHING WRITTEN" in result.stdout
        assert golden.read_bytes() == before, "the rejected import changed the golden set"

    def test_a_failing_import_leaves_no_staged_file_behind(self, tmp_path: Path) -> None:
        """A leftover .staged is a second copy of the golden set that nothing validates."""
        golden = tmp_path / "v.jsonl"
        golden.write_text(PRIOR, encoding="utf-8")

        runner.invoke(app, ["import-xlsx", str(book(tmp_path, page=1)), "--golden", str(golden)])

        assert list(tmp_path.glob("*.staged")) == []

    def test_a_failing_import_into_a_missing_file_creates_nothing(self, tmp_path: Path) -> None:
        """The first import must not leave a half-set behind either."""
        golden = tmp_path / "fresh.jsonl"

        result = runner.invoke(
            app, ["import-xlsx", str(book(tmp_path, page=1)), "--golden", str(golden)]
        )

        assert result.exit_code == 2
        assert not golden.exists(), "a rejected first import created the golden set anyway"

    def test_a_passing_import_does_write(self, tmp_path: Path) -> None:
        """The control.

        Without it, a command that had been broken to never write at all would pass every
        assertion above. Same reason the chunker tests assert a positive match.
        """
        golden = tmp_path / "v.jsonl"
        golden.write_text(PRIOR, encoding="utf-8")

        result = runner.invoke(
            app, ["import-xlsx", str(book(tmp_path, page=33)), "--golden", str(golden)]
        )

        assert result.exit_code == 0, result.stdout
        body = golden.read_text(encoding="utf-8")
        assert '"t-01"' in body
        assert "a header line that must survive" in body
        assert list(tmp_path.glob("*.staged")) == []


@pytest.mark.parametrize("page", [1, 33])
def test_dry_run_never_writes(tmp_path: Path, page: int) -> None:
    golden = tmp_path / "v.jsonl"
    golden.write_text(PRIOR, encoding="utf-8")

    runner.invoke(
        app,
        ["import-xlsx", str(book(tmp_path, page=page)), "--golden", str(golden), "--dry-run"],
    )

    assert golden.read_text(encoding="utf-8") == PRIOR
