"""Report tests.

EVAL_LOG.md is the artefact the project is ultimately judged on, so the property that
matters most is that it is **derived, never hand-maintained**: the JSONL run log is the
source of truth and the Markdown is a view of it. These tests pin that relationship, plus
the two honesty features that are easy to lose in a refactor — the unverified-price banner
and the provenance slice.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arag.agent.stub import EchoEngine, NullEngine
from arag.eval.report import (
    load_runs,
    persist_run,
    render_console,
    render_eval_log,
    to_record,
    write_eval_log,
)
from arag.eval.runner import run_eval
from arag.eval.schema import GoldenSet
from arag.eval.thresholds import Thresholds


@pytest.fixture
def golden(golden_path: Path) -> GoldenSet:
    return GoldenSet.load(golden_path)


@pytest.fixture
def thresholds(thresholds_path: Path) -> Thresholds:
    return Thresholds.load(thresholds_path)


@pytest.fixture
async def run(golden: GoldenSet, thresholds: Thresholds):  # type: ignore[no-untyped-def]
    return await run_eval(golden, NullEngine(), thresholds, profile_name="v0_floor")


class TestRecord:
    async def test_carries_reproduction_metadata(self, run) -> None:  # type: ignore[no-untyped-def]
        rec = to_record(run)
        for key in ("git_sha", "profile", "price_table_version", "golden_source", "n_items"):
            assert key in rec, f"{key} missing: the row would be unfalsifiable without it"

    async def test_excludes_per_item_detail(self, run) -> None:  # type: ignore[no-untyped-def]
        """Per-item output changes every run and would make the log undiffable. It belongs
        in traces, not in the file that has to stay readable across dozens of runs.
        """
        assert "items" not in to_record(run)

    async def test_gate_outcome_is_recorded_with_its_reasons(self, run) -> None:  # type: ignore[no-untyped-def]
        rec = to_record(run)
        assert rec["gate_passed"] is True
        assert rec["gate_failures"] == []


class TestPersistence:
    async def test_append_only(self, run, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        """History must accumulate. An overwrite would destroy the v1 to v2 story, which
        is the single most valuable output of the project.
        """
        path = tmp_path / "runs.jsonl"
        persist_run(run, path)
        persist_run(run, path)
        assert len(load_runs(path)) == 2

    def test_missing_file_loads_as_empty(self, tmp_path: Path) -> None:
        assert load_runs(tmp_path / "nope.jsonl") == []

    async def test_creates_parent_directories(self, run, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "nested" / "deeper" / "runs.jsonl"
        persist_run(run, path)
        assert path.exists()


class TestRenderEvalLog:
    def test_empty_history_says_so(self) -> None:
        out = render_eval_log([])
        assert "No runs recorded yet" in out

    async def test_unverified_price_banner_is_present(self, run) -> None:  # type: ignore[no-untyped-def]
        """Cost figures must be labelled estimates until the rates are checked. Losing
        this banner is how an unsourced cost-per-query number reaches a README.
        """
        out = render_eval_log([to_record(run)])
        assert "Cost figures are estimates" in out
        assert "gemini-flash-paid" in out

    async def test_newest_run_appears_first_in_history(
        self, golden: GoldenSet, thresholds: Thresholds
    ) -> None:
        null_run = await run_eval(golden, NullEngine(), thresholds, profile_name="v0_floor")
        echo_run = await run_eval(golden, EchoEngine(), thresholds, profile_name="v0_floor")
        out = render_eval_log([to_record(null_run), to_record(echo_run)])
        history = out.split("## History")[1]
        assert history.index("echo") < history.index("null")

    async def test_includes_stratum_and_provenance_slices(self, run) -> None:  # type: ignore[no-untyped-def]
        out = render_eval_log([to_record(run)])
        assert "### By stratum" in out
        assert "multihop_temporal" in out
        assert "### By label provenance" in out
        assert "seed_unverified" in out

    async def test_surfaces_warnings(self, run) -> None:  # type: ignore[no-untyped-def]
        out = render_eval_log([to_record(run)])
        assert "### Warnings" in out
        assert "seed_unverified" in out

    async def test_marks_a_failed_gate_and_lists_reasons(
        self, golden: GoldenSet, thresholds: Thresholds
    ) -> None:
        failing = await run_eval(golden, NullEngine(), thresholds, profile_name="v1_baseline")
        out = render_eval_log([to_record(failing)])
        assert "**fail**" in out
        assert "### Gate failures" in out

    async def test_says_it_is_generated_not_hand_written(self, run) -> None:  # type: ignore[no-untyped-def]
        out = render_eval_log([to_record(run)])
        assert "Do not edit by hand" in out

    async def test_undefined_metrics_render_as_a_dash_not_zero(
        self, golden: GoldenSet, thresholds: Thresholds
    ) -> None:
        """A dash and a 0.000 mean different things, and conflating them in the published
        table would undo the whole undefined-versus-zero convention in the metrics.
        """
        run = await run_eval(golden, NullEngine(), thresholds, profile_name="v0_floor")
        rec = to_record(run)
        assert rec["overall"]["ungrounded_citation_rate"] is None
        assert "—" in render_eval_log([rec])


class TestWriteEvalLog:
    async def test_writes_a_file_from_the_run_log(self, run, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        runs = tmp_path / "runs.jsonl"
        out = tmp_path / "EVAL_LOG.md"
        persist_run(run, runs)
        written = write_eval_log(runs, out)
        assert written == out
        assert "## History" in out.read_text(encoding="utf-8")

    async def test_regenerating_is_idempotent(self, run, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        """Only the render timestamp may change; the data section must be stable, or every
        regeneration produces a noisy diff and people stop reading it.
        """
        runs = tmp_path / "runs.jsonl"
        out = tmp_path / "EVAL_LOG.md"
        persist_run(run, runs)
        first = write_eval_log(runs, out).read_text(encoding="utf-8").split("---")[0]
        second = write_eval_log(runs, out).read_text(encoding="utf-8").split("---")[0]
        assert first == second


class TestConsole:
    async def test_shows_metrics_denominators_and_gate(self, run) -> None:  # type: ignore[no-untyped-def]
        out = render_console(run)
        assert "recall@10" in out
        assert "engine=null" in out
        assert "retrieval-n/a=4" in out, "skipped-item count must be visible, not implied"
        assert "GATE PASS" in out

    async def test_shows_the_calibration_tally(self, run) -> None:  # type: ignore[no-untyped-def]
        out = render_console(run)
        assert "calibration" in out
        assert "50:1" in out

    async def test_shows_both_cost_numbers(self, run) -> None:  # type: ignore[no-untyped-def]
        out = render_console(run)
        assert "marginal" in out
        assert "shadow" in out
