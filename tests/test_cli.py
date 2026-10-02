"""CLI tests, focused on exit codes.

CI reads the exit code and nothing else, so the code *is* the contract:

* 0 gate passed
* 1 gate failed — a quality regression
* 2 the run could not be trusted — invalid golden set, stale cassette, bad config

Collapsing 1 and 2 into a single failure code is how teams learn to re-run a build instead
of reading it, so each is asserted separately here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from arag.config import Settings
from arag.eval import cli as cli_module
from arag.eval.cli import app
from arag.eval.sampling import FAST_SUBSET_SIZE

GOLDEN_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden_seed.jsonl"

runner = CliRunner()


@pytest.fixture
def sandbox(
    tmp_path: Path,
    golden_path: Path,
    thresholds_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Point the CLI at a temp directory so tests never write repo files."""
    settings = Settings(
        golden_path=golden_path,
        thresholds_path=thresholds_path,
        cassette_dir=tmp_path / "cassettes",
        eval_log_path=tmp_path / "EVAL_LOG.md",
    )
    monkeypatch.setattr(cli_module, "settings", lambda: settings)
    monkeypatch.setattr(cli_module, "RUNS_PATH", tmp_path / "eval_runs.jsonl")
    return tmp_path


class TestValidate:
    def test_reports_composition_and_exits_zero(self, sandbox: Path) -> None:
        """--no-resolve isolates the composition report from corpus resolution.

        The seed fixture's spans are invented placeholders, so resolution correctly
        rejects them - see test_resolution_rejects_the_invented_fixture_spans below.
        """
        result = runner.invoke(app, ["validate", "--no-resolve"])
        assert result.exit_code == 0
        assert "by stratum" in result.stdout
        assert "NOT YET PUBLISHABLE" in result.stdout

    def test_warns_that_seed_metrics_are_not_results(self, sandbox: Path) -> None:
        result = runner.invoke(app, ["validate", "--no-resolve"])
        assert "must not be quoted as results" in result.stdout

    def test_says_so_loudly_when_spans_are_not_resolved(
        self, sandbox: Path, tmp_path: Path
    ) -> None:
        """A missing clause index must be reported, not silently skipped.

        Silently skipping would let a labeller believe their spans were checked when only
        the schema was, which is the exact false confidence this layer exists to remove.
        """
        result = runner.invoke(app, ["validate", "--index", str(tmp_path / "absent.json")])
        assert "SPANS NOT RESOLVED" in result.stdout

    def test_resolution_rejects_the_invented_fixture_spans(
        self, sandbox: Path, repo_root: Path
    ) -> None:
        """The seed fixture was written before the corpus existed, so its clause ids are
        guesses. Resolution must reject them and exit 2 - proof the layer works on real
        data rather than only on synthetic tests.
        """
        index = repo_root / "data" / "manifest" / "clause_index.json"
        if not index.exists():
            pytest.skip("run `arag-ingest clause-index` first")
        result = runner.invoke(app, ["validate", "--index", str(index)])
        assert result.exit_code == 2
        assert "UNUSABLE until fixed" in result.stdout

    def test_missing_golden_set_is_exit_two(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["validate", "--golden", str(tmp_path / "nope.jsonl")])
        assert result.exit_code == 2

    def test_invalid_golden_set_is_exit_two(self, tmp_path: Path) -> None:
        """A broken eval set is a broken harness (2), not a quality regression (1)."""
        bad = tmp_path / "bad.jsonl"
        bad.write_text(
            '{"id": "x", "question": "q", "strata": "flat_lookup", '
            '"authored_by": "human", "expected_behaviour": "answer"}\n',
            encoding="utf-8",
        )
        result = runner.invoke(app, ["validate", "--golden", str(bad)])
        assert result.exit_code == 2


class TestRun:
    def test_null_engine_passes_the_floor_gate(self, sandbox: Path) -> None:
        result = runner.invoke(
            app, ["run", "--engine", "null", "--profile", "v0_floor", "--no-log"]
        )
        assert result.exit_code == 0, result.stdout
        assert "GATE PASS" in result.stdout

    def test_gate_failure_is_exit_one(self, sandbox: Path) -> None:
        result = runner.invoke(
            app, ["run", "--engine", "null", "--profile", "v1_baseline", "--no-log"]
        )
        assert result.exit_code == 1
        assert "GATE FAIL" in result.stdout

    def test_unknown_engine_is_exit_two(self, sandbox: Path) -> None:
        result = runner.invoke(app, ["run", "--engine", "does-not-exist", "--no-log"])
        assert result.exit_code == 2

    def test_bad_thresholds_file_is_exit_two(self, sandbox: Path, tmp_path: Path) -> None:
        bad = tmp_path / "t.yaml"
        bad.write_text("not: a: valid: mapping\n", encoding="utf-8")
        result = runner.invoke(
            app, ["run", "--engine", "null", "--thresholds", str(bad), "--no-log"]
        )
        assert result.exit_code == 2

    def test_fast_subset_runs_fewer_or_equal_items(self, sandbox: Path) -> None:
        result = runner.invoke(
            app, ["run", "--engine", "null", "--profile", "v0_floor", "--fast", "--no-log"]
        )
        assert result.exit_code == 0
        # Derived from the fixture rather than hardcoded. This assertion has now been
        # broken twice by legitimate fixture growth - s-017 for conditional_override, then
        # s-018 for supersession - and each time the fix was to retype the number, which
        # is a habit worth removing: a test whose failure is routinely "update the count"
        # trains people to update counts without reading what changed.
        #
        # The property is still asserted, not weakened: while the fixture is under
        # FAST_SUBSET_SIZE the fast subset must contain EVERY item, so the expected count
        # is the fixture size. If the fixture ever exceeds the target this assertion
        # becomes wrong rather than merely stale, so it fails loudly with instructions.
        n_fixture = sum(
            1
            for line in GOLDEN_FIXTURE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        )
        assert n_fixture <= FAST_SUBSET_SIZE, (
            f"the fixture now holds {n_fixture} items, over the {FAST_SUBSET_SIZE}-item "
            "fast-subset target, so 'fast runs everything' no longer holds. Assert the "
            "stratified subset's composition here instead of the total."
        )
        assert f"items={n_fixture}" in result.stdout

    def test_limit_zero_is_exit_two(self, sandbox: Path) -> None:
        result = runner.invoke(app, ["run", "--engine", "null", "--limit", "0", "--no-log"])
        assert result.exit_code == 2

    def test_log_writes_run_history_and_renders_the_markdown(self, sandbox: Path) -> None:
        result = runner.invoke(app, ["run", "--engine", "null", "--profile", "v0_floor", "--log"])
        assert result.exit_code == 0
        assert (sandbox / "eval_runs.jsonl").exists()
        assert "## History" in (sandbox / "EVAL_LOG.md").read_text(encoding="utf-8")

    def test_echo_engine_reports_the_opposite_failure(self, sandbox: Path) -> None:
        result = runner.invoke(
            app, ["run", "--engine", "echo", "--profile", "v0_floor", "--no-log"]
        )
        assert result.exit_code == 0
        assert "false abst   0.000" in result.stdout


class TestJudgeAgreement:
    def _labels(self, path: Path, rows: list[tuple[str, str, str]]) -> Path:
        path.write_text(
            "\n".join(f'{{"item_id": "{i}", "human": "{h}", "judge": "{j}"}}' for i, h, j in rows),
            encoding="utf-8",
        )
        return path

    def test_acceptable_kappa_exits_zero(self, sandbox: Path, tmp_path: Path) -> None:
        # kappa = 0.615 (see test_agreement.py for the arithmetic)
        labels = self._labels(
            tmp_path / "labels.jsonl",
            [("a", "1", "1"), ("b", "1", "0"), ("c", "0", "0"), ("d", "0", "0"), ("e", "1", "1")],
        )
        result = runner.invoke(app, ["judge-agreement", str(labels)])
        assert result.exit_code == 0
        assert "judge accepted" in result.stdout
        assert "Publish this kappa alongside the metrics" in result.stdout

    def test_insufficient_kappa_exits_one(self, sandbox: Path, tmp_path: Path) -> None:
        """A judge that always says "correct" agrees 80% of the time and knows nothing.

        This is the case the gate exists for: the run must fail rather than let
        judge-derived faithfulness numbers be published.
        """
        rows = [(f"i{n}", "1", "1") for n in range(8)] + [("i8", "0", "1"), ("i9", "0", "1")]
        labels = self._labels(tmp_path / "labels.jsonl", rows)
        result = runner.invoke(app, ["judge-agreement", str(labels)])
        assert result.exit_code == 1
        assert "INSUFFICIENT" in result.stdout

    def test_warns_on_a_small_sample(self, sandbox: Path, tmp_path: Path) -> None:
        labels = self._labels(
            tmp_path / "labels.jsonl",
            [("a", "1", "1"), ("b", "1", "0"), ("c", "0", "0"), ("d", "0", "0"), ("e", "1", "1")],
        )
        result = runner.invoke(app, ["judge-agreement", str(labels)])
        assert "target is 25" in result.stdout

    def test_lists_disagreements_for_review(self, sandbox: Path, tmp_path: Path) -> None:
        labels = self._labels(
            tmp_path / "labels.jsonl",
            [("a", "1", "1"), ("b", "1", "0"), ("c", "0", "0"), ("d", "0", "0"), ("e", "1", "1")],
        )
        result = runner.invoke(app, ["judge-agreement", str(labels)])
        assert "disagreement" in result.stdout
        assert "b" in result.stdout

    def test_missing_file_is_exit_two(self, sandbox: Path, tmp_path: Path) -> None:
        result = runner.invoke(app, ["judge-agreement", str(tmp_path / "nope.jsonl")])
        assert result.exit_code == 2

    def test_malformed_row_is_exit_two(self, sandbox: Path, tmp_path: Path) -> None:
        labels = tmp_path / "labels.jsonl"
        labels.write_text('{"item_id": "a", "human": "1"}\n', encoding="utf-8")
        result = runner.invoke(app, ["judge-agreement", str(labels)])
        assert result.exit_code == 2

    def test_empty_file_is_exit_two(self, sandbox: Path, tmp_path: Path) -> None:
        labels = tmp_path / "labels.jsonl"
        labels.write_text("# only a comment\n", encoding="utf-8")
        result = runner.invoke(app, ["judge-agreement", str(labels)])
        assert result.exit_code == 2


class TestReportCommand:
    def test_renders_from_an_empty_history(self, sandbox: Path) -> None:
        result = runner.invoke(app, ["report"])
        assert result.exit_code == 0
        assert "No runs recorded yet" in (sandbox / "EVAL_LOG.md").read_text(encoding="utf-8")
