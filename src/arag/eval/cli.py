"""``arag-eval`` — the single command the whole project is measured by.

Exit codes matter here because CI reads them:

* ``0`` gate passed
* ``1`` gate failed (a quality regression, the intended CI failure)
* ``2`` the run could not be trusted (invalid golden set, cassette miss, config error)

Separating 1 from 2 is not pedantry. "Retrieval got worse" and "the harness is broken"
demand different responses, and a CI job that returns the same code for both trains people
to re-run the build instead of reading it.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer

from arag.agent.protocols import QueryEngine
from arag.agent.stub import EchoEngine, NullEngine
from arag.config import settings
from arag.eval.cassettes import CassetteMiss, CassetteMode, CassetteStore
from arag.eval.metrics.agreement import cohens_kappa
from arag.eval.report import persist_run, render_console, write_eval_log
from arag.eval.resolve import CorpusIndex, resolve_golden_set
from arag.eval.runner import run_eval
from arag.eval.sampling import FAST_SUBSET_SIZE, stratified_subset
from arag.eval.schema import GoldenSet, SetTargets
from arag.eval.thresholds import Thresholds
from arag.obs import configure_logging, get_logger

app = typer.Typer(add_completion=False, help="Evaluation harness for the agentic RAG system.")
log = get_logger(__name__)

# Registry rather than an import-time switch, so a new engine is one line here and the
# CLI never imports LangGraph until an engine that needs it is actually selected.
ENGINES: dict[str, type[QueryEngine]] = {
    "null": NullEngine,
    "echo": EchoEngine,
}

# Engines that need the corpus built are constructed lazily, by name, so that importing
# this CLI never pulls the ingest stack (and PyMuPDF) into the eval package. The
# import-boundary test requires arag.eval to depend on protocols rather than
# implementations, and a module-scope import here would break that for every command
# including `validate`, which has no business touching a PDF.
LAZY_ENGINES = ("bm25", "bm25-row-nl")


def _build_lazy_engine(key: str, top_k: int) -> QueryEngine:
    from arag.agent.retrieval_only import RetrievalOnlyEngine
    from arag.config import REPO_ROOT
    from arag.index.build import build_corpus
    from arag.retrieval.lexical import LexicalRetriever

    serialisation = "row_nl" if key.endswith("row-nl") else "markdown"
    corpus = build_corpus(
        REPO_ROOT / "data" / "manifest" / "sources.jsonl",
        REPO_ROOT / "data" / "raw",
        serialisation=serialisation,
    )
    log.info(
        "corpus_built",
        serialisation=serialisation,
        chunks=len(corpus.chunks),
    )
    return RetrievalOnlyEngine(LexicalRetriever(corpus.index, corpus.chunks), top_k=top_k, name=key)


RUNS_PATH = Path("data/eval_runs.jsonl")
CLAUSE_INDEX = Path("data/manifest/clause_index.json")


def _load_golden(path: Path | None) -> GoldenSet:
    resolved = path or settings().golden_path
    if not resolved.exists():
        typer.secho(f"golden set not found: {resolved}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    try:
        return GoldenSet.load(resolved)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc


def _load_thresholds(path: Path | None) -> Thresholds:
    resolved = path or settings().thresholds_path
    try:
        return Thresholds.load(resolved)
    except Exception as exc:
        typer.secho(f"invalid thresholds file {resolved}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc


@app.command("import-xlsx")
def import_xlsx(
    xlsx: Annotated[Path, typer.Argument(help="The labelling workbook to read.")],
    golden: Annotated[Path | None, typer.Option(help="Golden JSONL to write.")] = None,
    dry_run: Annotated[
        bool, typer.Option(help="Parse and report without writing the JSONL.")
    ] = False,
) -> None:
    """Read the labelling spreadsheet back into the golden set, then validate it.

    The spreadsheet is where labelling actually happens - reading six PDFs and typing
    findings is not a JSONL activity - but JSONL stays the storage format, so this is a
    one-way gate rather than a second source of truth.

    **A malformed row fails exactly as it would in the file.** The row is assembled into
    the same dict the JSONL loader builds and handed to `GoldenItem`; the schema is not
    reimplemented here, so the rules, the messages and the strictness are identical. The
    only extra checks are the ones the spreadsheet format introduces and the schema cannot
    see: the span columns are positional and must align.

    Exit 2 on any bad row - nothing is written when one row is unusable, because a partial
    import leaves the golden set in a state nobody chose.
    """
    configure_logging()
    from arag.eval.xlsx import read_xlsx, write_jsonl

    target = golden or settings().golden_path
    try:
        items, problems = read_xlsx(xlsx)
    except Exception as exc:
        typer.secho(f"  cannot read {xlsx}: {exc}", fg=typer.colors.RED)
        raise typer.Exit(2) from exc

    typer.echo(f"\n  {len(items)} complete item(s) in {xlsx.name}")
    if problems:
        typer.secho(f"  {len(problems)} unusable row(s):", fg=typer.colors.RED)
        for problem in problems:
            typer.secho(f"    ! {problem}", fg=typer.colors.RED)
        typer.secho(
            "\n  Nothing written. A partial import leaves the golden set in a state nobody chose.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(2)

    ids = [i.id for i in items]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        typer.secho(f"  duplicate id(s): {duplicates}", fg=typer.colors.RED)
        raise typer.Exit(2)

    if dry_run:
        typer.secho("  --dry-run: parsed cleanly, nothing written.", fg=typer.colors.GREEN)
        return

    header = ""
    if target.exists():
        existing = target.read_text(encoding="utf-8").splitlines()
        header = "\n".join(line for line in existing if line.startswith("#"))
    write_jsonl(target, items, header)
    typer.secho(f"  wrote {len(items)} item(s) -> {target}", fg=typer.colors.GREEN)
    typer.echo("  now validating, same as `arag-eval validate`:")
    validate(golden=target, index_path=None, resolve=True)


@app.command()
def validate(
    golden: Annotated[Path | None, typer.Option(help="Path to the golden JSONL.")] = None,
    index_path: Annotated[
        Path | None, typer.Option("--index", help="clause_index.json for corpus resolution.")
    ] = None,
    resolve: Annotated[bool, typer.Option(help="Check every span against the real corpus.")] = True,
) -> None:
    """Validate the golden set: schema, corpus resolution, then composition.

    Two layers, and the second is what catches silent failures. Schema validation proves a
    span is well-formed; corpus resolution proves it is true. A well-formed span pointing
    at a page where the clause does not exist loads fine, counts in the denominator, and
    scores zero forever with no error anywhere.

    Exit 2 when any item is unusable, because an unusable item in a hand-labelled set is a
    broken harness, not a quality regression.
    """
    configure_logging()
    gs = _load_golden(golden)

    typer.echo("")
    typer.echo(f"  {len(gs.items)} items from {gs.source}")
    typer.echo("")
    typer.echo("  by stratum")
    for name, count in sorted(gs.strata_counts().items()):
        typer.echo(f"    {name:<26}{count:>4}")
    typer.echo("")
    typer.echo("  by provenance")
    for name, count in sorted(gs.provenance_counts().items()):
        typer.echo(f"    {name:<26}{count:>4}")

    gaps = gs.coverage_gaps()
    if gaps:
        typer.echo("")
        typer.secho(f"  no items for strata: {', '.join(gaps)}", fg=typer.colors.YELLOW)

    unusable: set[str] = set()
    if resolve:
        resolved = index_path or CLAUSE_INDEX
        typer.echo("")
        if not resolved.exists():
            typer.secho(
                f"  SPANS NOT RESOLVED: {resolved} is missing. Run "
                "`arag-ingest clause-index` to check labelled spans against the real "
                "corpus. Schema validation alone cannot tell you a clause_id points at "
                "the wrong page.",
                fg=typer.colors.YELLOW,
            )
        else:
            index = CorpusIndex.load(resolved)
            report = resolve_golden_set(gs, index)
            unusable = report.unusable_item_ids
            typer.echo(
                f"  corpus resolution: {len(index.document_ids)} document(s), "
                f"index built {report.index_generated_at}"
            )
            if report.hard_errors:
                typer.secho(
                    f"    {len(report.hard_errors)} ERROR(S) - demonstrably wrong:",
                    fg=typer.colors.RED,
                )
                for finding in report.hard_errors:
                    typer.secho(f"      x {finding}", fg=typer.colors.RED)
            if report.unverifiable:
                # Printed in its own block, above the warnings. An unverifiable label is
                # indistinguishable from a wrong one, and the whole point is that it must
                # not be skimmed past in a long run.
                typer.echo("")
                typer.secho(
                    f"    {len(report.unverifiable)} UNVERIFIABLE - nothing can check "
                    "these, which is indistinguishable from them being wrong:",
                    fg=typer.colors.RED,
                    bold=True,
                )
                for finding in report.unverifiable:
                    typer.secho(f"      ? {finding}", fg=typer.colors.RED, bold=True)
                typer.echo("")
            if report.warnings:
                typer.secho(f"    {len(report.warnings)} warning(s):", fg=typer.colors.YELLOW)
                for finding in report.warnings:
                    typer.secho(f"      ! {finding}", fg=typer.colors.YELLOW)
            if report.ok and not report.warnings:
                typer.secho("    every span resolves against the corpus", fg=typer.colors.GREEN)

    shortfall = SetTargets().shortfall(gs)
    typer.echo("")
    if shortfall:
        typer.secho("  NOT YET PUBLISHABLE - composition shortfall:", fg=typer.colors.YELLOW)
        for line in shortfall:
            typer.echo(f"    - {line}")
        typer.echo("")
        typer.echo(
            "  These are development-signal numbers. Metrics from this set must not be "
            "quoted as results."
        )
        typer.echo("")
    else:
        typer.secho("  composition targets met", fg=typer.colors.GREEN)
        typer.echo("")

    if unusable:
        typer.secho(
            f"  {len(unusable)} item(s) are UNUSABLE until fixed: {sorted(unusable)}",
            fg=typer.colors.RED,
        )
        typer.echo("")
        raise typer.Exit(2)


@app.command()
def run(
    engine: Annotated[str, typer.Option(help="Engine key from the registry.")] = "null",
    profile: Annotated[str | None, typer.Option(help="Threshold profile override.")] = None,
    golden: Annotated[Path | None, typer.Option()] = None,
    thresholds_path: Annotated[Path | None, typer.Option("--thresholds")] = None,
    fast: Annotated[
        bool, typer.Option(help="Stratified subset, cassette replay, no network.")
    ] = False,
    limit: Annotated[int | None, typer.Option(help="Cap item count (debugging only).")] = None,
    concurrency: Annotated[int, typer.Option(help="Parallel items. Keep low: 15 RPM ceiling.")] = 1,
    record: Annotated[bool, typer.Option(help="Record provider calls into cassettes.")] = False,
    log_run: Annotated[
        bool, typer.Option("--log/--no-log", help="Append to data/eval_runs.jsonl.")
    ] = True,
) -> None:
    """Run the suite and enforce the gate."""
    configure_logging(settings().log_level)

    if engine not in ENGINES and engine not in LAZY_ENGINES:
        typer.secho(
            f"unknown engine {engine!r}; available: {sorted([*ENGINES, *LAZY_ENGINES])}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(2)

    gs = _load_golden(golden)
    th = _load_thresholds(thresholds_path)

    if fast:
        gs = stratified_subset(gs, FAST_SUBSET_SIZE)
    if limit is not None:
        gs = GoldenSet(items=gs.items[:limit], source=gs.source)
    if not gs.items:
        typer.secho("no items to run", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    mode = CassetteMode.RECORD if record else CassetteMode.REPLAY if fast else CassetteMode.OFF
    store = CassetteStore(settings().cassette_dir, mode)

    built: QueryEngine = (
        _build_lazy_engine(engine, th.retrieval.k) if engine in LAZY_ENGINES else ENGINES[engine]()
    )

    try:
        result = asyncio.run(
            run_eval(
                gs,
                built,
                th,
                profile_name=profile,
                concurrency=concurrency,
                cassette_stats=store.stats(),
            )
        )
    except CassetteMiss as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc

    result.cassette_stats = store.stats()
    typer.echo(render_console(result))

    if log_run:
        persist_run(result, RUNS_PATH)
        path = write_eval_log(RUNS_PATH, settings().eval_log_path)
        typer.echo(f"  recorded -> {RUNS_PATH}, rendered -> {path}\n")

    if result.gate and not result.gate.passed:
        raise typer.Exit(1)


@app.command("judge-agreement")
def judge_agreement(
    labels: Annotated[Path, typer.Argument(help="JSONL of {item_id, human, judge}.")],
    thresholds_path: Annotated[Path | None, typer.Option("--thresholds")] = None,
) -> None:
    """Report Cohen's kappa between human labels and the local judge.

    DESIGN §5.8 makes this a precondition for quoting any judge-derived metric. A free
    judge that has never been validated produces numbers that look like evidence.
    """
    configure_logging()
    th = _load_thresholds(thresholds_path)

    if not labels.exists():
        typer.secho(f"labels file not found: {labels}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    ids: list[str] = []
    human: list[str] = []
    judge: list[str] = []
    for lineno, line in enumerate(labels.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            row = json.loads(line)
            ids.append(str(row["item_id"]))
            human.append(str(row["human"]))
            judge.append(str(row["judge"]))
        except (KeyError, json.JSONDecodeError) as exc:
            typer.secho(f"{labels.name}:{lineno}: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(2) from exc

    if not ids:
        typer.secho("no labelled rows found", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    report = cohens_kappa(human, judge, item_ids=ids)
    typer.echo(f"\n  {report.summary()}")
    typer.echo(f"  label distribution (human): {report.per_label_counts}")

    want = th.judge.agreement_sample_size
    if report.n < want:
        typer.secho(
            f"  sample is {report.n}, target is {want} — kappa on a small sample is unstable",
            fg=typer.colors.YELLOW,
        )

    if report.disagreements:
        typer.echo(f"\n  {len(report.disagreements)} disagreement(s):")
        for item_id, h, j in report.disagreements[:20]:
            typer.echo(f"    {item_id:<12} human={h:<12} judge={j}")

    typer.echo("")
    if report.kappa < th.judge.min_kappa:
        typer.secho(
            f"  INSUFFICIENT: kappa {report.kappa:.3f} < {th.judge.min_kappa}. "
            "Judge-derived metrics are not trustworthy at this agreement level; switch to a "
            "hosted judge before quoting faithfulness or answer-correctness numbers.\n",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    typer.secho(
        f"  judge accepted ({report.band}). Publish this kappa alongside the metrics.\n",
        fg=typer.colors.GREEN,
    )


@app.command()
def report() -> None:
    """Regenerate EVAL_LOG.md from the recorded run log."""
    configure_logging()
    path = write_eval_log(RUNS_PATH, settings().eval_log_path)
    typer.echo(f"  rendered -> {path}")


if __name__ == "__main__":  # pragma: no cover
    app()
