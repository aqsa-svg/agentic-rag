"""``arag-ingest`` — fetch the corpus and report what is wrong with it.

Two commands, deliberately separate:

* ``fetch`` touches the network and writes ``data/raw/``. It records an outcome per source
  and exits non-zero only if *nothing* was fetched, because a partially available corpus is
  still worth probing.
* ``probe`` is offline and read-only. It answers "what breaks per document" without fixing
  anything, so the ingest work that follows is driven by measurements rather than by
  assumptions about what an insurance PDF looks like.
"""

from __future__ import annotations

import asyncio
import json
import re
import unicodedata
from pathlib import Path
from typing import Annotated

import pymupdf
import typer

from arag.config import REPO_ROOT, settings
from arag.ingest.clause_index import build_clause_index, concept_mapping
from arag.ingest.fetch import FetchResult, fetch_all
from arag.ingest.manifest import Manifest
from arag.ingest.probe import DocumentProbe, probe_document
from arag.obs import configure_logging, get_logger

app = typer.Typer(add_completion=False, help="Corpus ingest: fetch sources and probe them.")
log = get_logger(__name__)

MANIFEST_PATH = REPO_ROOT / "data" / "manifest" / "sources.jsonl"
RAW_DIR = REPO_ROOT / "data" / "raw"
FETCH_REPORT = REPO_ROOT / "data" / "manifest" / "fetch_report.json"
PROBE_REPORT = REPO_ROOT / "data" / "manifest" / "probe_report.json"
CLAUSE_INDEX = REPO_ROOT / "data" / "manifest" / "clause_index.json"


def _load_manifest(path: Path | None) -> Manifest:
    resolved = path or MANIFEST_PATH
    if not resolved.exists():
        typer.secho(f"manifest not found: {resolved}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    try:
        return Manifest.load(resolved)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc


@app.command()
def fetch(
    manifest_path: Annotated[Path | None, typer.Option("--manifest")] = None,
    force: Annotated[bool, typer.Option(help="Refetch even if a cached copy matches.")] = False,
    concurrency: Annotated[int, typer.Option()] = 3,
) -> None:
    """Fetch every source in the manifest. Per-source failures are recorded, not raised."""
    configure_logging(settings().log_level)
    manifest = _load_manifest(manifest_path)

    results = asyncio.run(fetch_all(manifest, RAW_DIR, force=force, concurrency=concurrency))

    typer.echo("")
    typer.echo(f"  {'source':<28}{'outcome':<16}{'size':>10}  sha256")
    for r in results:
        size = f"{r.size_bytes / 1024:,.0f}KB" if r.size_bytes else "-"
        colour = typer.colors.GREEN if r.usable else typer.colors.RED
        typer.secho(
            f"  {r.source_id:<28}{r.outcome.value:<16}{size:>10}  {(r.sha256 or '-')[:16]}",
            fg=colour,
        )
        if r.detail:
            typer.echo(f"      {r.detail}")
        if r.mirror_drift:
            typer.secho(
                "      MIRROR DRIFT: a declared mirror has different bytes to the primary. "
                "For a regulatory instrument this needs a human look before either is trusted.",
                fg=typer.colors.YELLOW,
            )
        for attempt in r.attempts:
            if attempt.error:
                typer.echo(f"      tried {attempt.url[:70]} -> {attempt.error}")

    usable = [r for r in results if r.usable]
    typer.echo("")
    typer.echo(f"  {len(usable)}/{len(results)} sources usable")

    FETCH_REPORT.parent.mkdir(parents=True, exist_ok=True)
    FETCH_REPORT.write_text(
        json.dumps([_fetch_record(r) for r in results], indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    typer.echo(f"  report -> {FETCH_REPORT}")

    # Hashes are printed rather than written back into the manifest automatically: pinning a
    # hash is a deliberate act, and a silent rewrite would defeat the point of pinning.
    if usable:
        typer.echo("\n  to pin these versions, add expected_sha256 to the manifest:")
        for r in usable:
            typer.echo(f'    {r.source_id}: "{r.sha256}"')
    typer.echo("")

    if not usable:
        raise typer.Exit(1)


@app.command()
def probe(
    manifest_path: Annotated[Path | None, typer.Option("--manifest")] = None,
    source: Annotated[str | None, typer.Option(help="Probe a single source id.")] = None,
    max_pages: Annotated[int | None, typer.Option(help="Cap pages per document.")] = None,
    verbose: Annotated[bool, typer.Option(help="Per-page detail for suspect pages.")] = False,
) -> None:
    """Report what breaks per document. Read-only; fixes nothing."""
    configure_logging(settings().log_level)
    manifest = _load_manifest(manifest_path)

    sources = [s for s in manifest.sources if source is None or s.id == source]
    if not sources:
        typer.secho(f"no source matched {source!r}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    probes: list[DocumentProbe] = []
    for src in sources:
        path = RAW_DIR / f"{src.id}.pdf"
        if not path.exists():
            typer.secho(
                f"  {src.id:<28}NOT FETCHED (run `arag-ingest fetch` first)",
                fg=typer.colors.YELLOW,
            )
            continue
        probes.append(probe_document(src.id, path, max_pages=max_pages))

    if not probes:
        typer.secho("nothing to probe", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    _render_summary(probes)
    _render_breakages(probes, verbose=verbose)

    PROBE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    PROBE_REPORT.write_text(
        json.dumps([_probe_record(p) for p in probes], indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    typer.echo(f"  report -> {PROBE_REPORT}\n")


def _render_summary(probes: list[DocumentProbe]) -> None:
    typer.echo("")
    header = (
        f"  {'source':<28}{'pp':>4}{'chars':>9}{'med/pp':>8}{'ocr':>5}"
        f"{'2col':>6}{'splic':>7}{'lig':>6}{'fused':>7}{'tbl':>5}  heading pattern"
    )
    typer.echo(header)
    typer.echo("  " + "-" * (len(header) - 2))
    for p in probes:
        if not p.ok:
            typer.secho(f"  {p.source_id:<28}UNREADABLE: {p.error}", fg=typer.colors.RED)
            continue
        typer.echo(
            f"  {p.source_id:<28}{p.pages:>4}{p.total_chars:>9,}"
            f"{p.median_chars_per_page:>8,.0f}{len(p.pages_needing_ocr):>5}"
            f"{len(p.two_column_pages):>6}{len(p.reading_order_broken_pages):>7}"
            f"{p.ligature_chars:>6}{len(p.fused_text_pages):>7}"
            f"{p.table_candidates:>5}  {p.dominant_heading_pattern or '-'}"
        )
    typer.echo("")
    for line in (
        "pp=pages   med/pp=median chars per page",
        "ocr=no text layer, image-dominant, and NOT a cover page",
        "2col=two-column layout. NOT itself a defect - most extract fine",
        "splic=the subset of 2col PROSE pages (tables excluded) whose extraction order",
        "      INTERLEAVES columns. This IS a defect: spliced clauses read fluently",
        "      while being wrong.",
        "lig=ligature codepoints (U+FB00-06). Breaks BM25: 'benefit' != 'bene<fi>t'.",
        "fused=pages with run-together tokens   tbl=table candidates",
    ):
        typer.echo(f"  {line}")


def _render_breakages(probes: list[DocumentProbe], *, verbose: bool) -> None:
    typer.echo("")
    typer.secho("  WHAT BREAKS, PER DOCUMENT", bold=True)
    for p in probes:
        typer.echo("")
        typer.secho(f"  {p.source_id}", bold=True)
        for line in p.breakages():
            colour = typer.colors.GREEN if line.startswith("no blocking") else typer.colors.RED
            typer.secho(f"    ! {line}", fg=colour)
        for line in p.notes():
            typer.secho(f"    . {line}", fg=typer.colors.CYAN)
        if p.ok:
            totals = p.heading_totals
            if totals:
                shown = ", ".join(f"{k}={v}" for k, v in list(totals.items())[:4])
                typer.echo(f"    heading hits: {shown}")
            if p.producer:
                typer.echo(f"    producer: {p.producer[:60]}")
        if verbose and p.ok:
            for page in p.page_probes:
                if page.suspect or page.needs_ocr:
                    typer.echo(
                        f"      p{page.number}: chars={page.chars} "
                        f"longtok={page.long_token_share:.3f} cid={page.cid_artefacts} "
                        f"img={page.image_area_ratio:.2f} "
                        f"2col={page.two_column} switches={page.column_switches} "
                        f"tables={page.table_candidates}"
                    )
    typer.echo("")


def _fetch_record(r: FetchResult) -> dict[str, object]:
    return {
        "source_id": r.source_id,
        "outcome": r.outcome.value,
        "usable": r.usable,
        "sha256": r.sha256,
        "size_bytes": r.size_bytes,
        "mirror_drift": r.mirror_drift,
        "detail": r.detail,
        "attempts": [
            {
                "url": a.url,
                "status": a.status,
                "content_type": a.content_type,
                "bytes": a.bytes_downloaded,
                "sha256": a.sha256,
                "error": a.error,
            }
            for a in r.attempts
        ],
    }


def _probe_record(p: DocumentProbe) -> dict[str, object]:
    return {
        "source_id": p.source_id,
        "ok": p.ok,
        "error": p.error,
        "pages": p.pages,
        "encrypted": p.encrypted,
        "has_outline": p.has_outline,
        "outline_entries": p.outline_entries,
        "producer": p.producer,
        "total_chars": p.total_chars,
        "median_chars_per_page": p.median_chars_per_page,
        "scanned_fraction": p.scanned_fraction,
        "pages_needing_ocr": p.pages_needing_ocr,
        "pages_without_text": p.pages_without_text,
        "suspect_pages": p.suspect_pages,
        "two_column_pages": p.two_column_pages,
        "reading_order_broken_pages": p.reading_order_broken_pages,
        "cover_pages": p.cover_pages,
        "fused_text_pages": p.fused_text_pages,
        "cid_pages": p.cid_pages,
        "table_candidates": p.table_candidates,
        "pages_with_tables": p.pages_with_tables,
        "ligature_chars": p.ligature_chars,
        "ligature_pages": p.ligature_pages,
        "soft_hyphens": p.soft_hyphens,
        "nbsp_chars": p.nbsp_chars,
        "table_driven_switch_pages": p.table_driven_switch_pages,
        "blank_pages": p.blank_pages,
        "devanagari_chars": p.devanagari_chars,
        "heading_totals": p.heading_totals,
        "dominant_heading_pattern": p.dominant_heading_pattern,
        "breakages": p.breakages(),
        "notes": p.notes(),
    }


@app.command()
def locate(
    phrase: Annotated[str, typer.Argument(help="Text to find, case-insensitive.")],
    source: Annotated[str | None, typer.Option(help="Restrict to one source id.")] = None,
    context: Annotated[int, typer.Option(help="Characters of surrounding text.")] = 160,
    limit: Annotated[int, typer.Option(help="Max hits per document.")] = 6,
) -> None:
    """Find a phrase in the corpus and print page + nearest clause heading.

    This exists to support hand-labelling. Writing a golden item requires the clause the
    answer actually lives in, and guessing it from a PDF viewer produces two failure modes
    that the eval harness cannot distinguish from a retrieval miss:

    * a page number from the summary/index table rather than the clause body, and
    * a clause id in a format the SpanMatcher silently fails to parse.

    The output is deliberately shaped to be copied straight into a golden item.
    """
    configure_logging(settings().log_level)
    manifest = _load_manifest(None)
    needle = unicodedata.normalize("NFKC", phrase).lower()

    heading = re.compile(
        r"^\s{0,6}(?:(Section\s+[IVXLCDM0-9]+)|(\d{1,2}(?:\.\d{1,3}){0,3})[\.\)]?)\s+(\S.{0,60})",
        re.MULTILINE,
    )

    total = 0
    for src in manifest.sources:
        if source is not None and src.id != source:
            continue
        path = RAW_DIR / f"{src.id}.pdf"
        if not path.exists():
            continue

        doc = pymupdf.open(path)
        hits = 0
        with doc:
            for pno in range(doc.page_count):
                raw = doc[pno].get_text("text")
                haystack = unicodedata.normalize("NFKC", raw).lower()
                start = haystack.find(needle)
                if start < 0:
                    continue

                # Nearest heading ABOVE the hit, which is the clause the text sits in.
                nearest = ""
                for m in heading.finditer(raw[:start]):
                    nearest = (m.group(1) or m.group(2) or "").strip()
                snippet = " ".join(
                    raw[max(0, start - context // 2) : start + len(needle) + context].split()
                )

                typer.secho(f"  {src.id}  p{pno + 1}", bold=True)
                typer.echo(f"    nearest heading above : {nearest or '(none found)'}")
                typer.echo(f"    authority_rank        : {src.authority_rank}")
                if src.superseded_by:
                    typer.secho(
                        f"    SUPERSEDED BY         : {src.superseded_by} "
                        "(citing this document is a hard failure for current-date questions)",
                        fg=typer.colors.YELLOW,
                    )
                typer.echo(f"    ...{snippet}...")

                # A bare one-or-two-digit "heading" is indistinguishable from a running
                # page number ("1 of 18"), and every page in this corpus has one. Emitting
                # it as a clause_id would hand the labeller a confidently wrong value, so
                # low-confidence inferences are surfaced as FIXME instead. A labelling aid
                # that guesses is worse than one that refuses.
                confident = bool(nearest) and (
                    nearest.lower().startswith("section") or "." in nearest
                )
                clause = nearest if confident else "FIXME"
                if not confident:
                    typer.secho(
                        "    clause_id: NOT INFERRED - the nearest heading was "
                        f"{nearest or 'absent'!r}, which cannot be told apart from a page "
                        "number. Read the page and supply the clause yourself.",
                        fg=typer.colors.YELLOW,
                    )
                typer.echo(
                    f'    span: {{"document_id": "{src.id}", "page": {pno + 1}, '
                    f'"clause_id": "{clause}"}}'
                )
                typer.echo("")
                hits += 1
                total += 1
                if hits >= limit:
                    break

    if total == 0:
        typer.secho(f"  {phrase!r} not found in any fetched document", fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    typer.echo(f"  {total} hit(s). clause_id must be canonical - see arag.eval.concepts.")


@app.command("clause-index")
def clause_index_cmd(
    manifest_path: Annotated[Path | None, typer.Option("--manifest")] = None,
    show_mapping: Annotated[
        bool, typer.Option(help="Print the star-2021 <-> star-2025 concept mapping.")
    ] = True,
) -> None:
    """Build data/manifest/clause_index.json and show the cross-version mapping.

    The index is what lets `arag-eval validate` check a labelled span against the real
    corpus instead of only against the schema. Without it, a well-formed clause_id
    pointing at the wrong page passes validation and then silently misses at SpanMatcher
    time - present in the file, absent from every metric, no error anywhere.
    """
    configure_logging(settings().log_level)
    manifest = _load_manifest(manifest_path)

    index = build_clause_index(manifest, RAW_DIR)
    if not index.documents:
        typer.secho("no documents indexed - run `arag-ingest fetch` first", fg=typer.colors.RED)
        raise typer.Exit(2)

    path = index.save(CLAUSE_INDEX)
    typer.echo("")
    typer.echo(f"  {'document':<28}{'pages':>7}{'clause ids':>12}{'concepts':>10}")
    typer.echo("  " + "-" * 55)
    for source_id, doc in sorted(index.documents.items()):
        typer.echo(f"  {source_id:<28}{doc.pages:>7}{len(doc.clauses):>12}{len(doc.concepts):>10}")
    typer.echo(f"\n  index -> {path}")

    if show_mapping:
        rows = concept_mapping(index, "star-comprehensive-2021", "star-comprehensive-2025")
        if not rows:
            typer.secho("\n  no shared concepts - cannot demonstrate the join", fg=typer.colors.RED)
            raise typer.Exit(1)

        typer.echo("")
        typer.secho(
            "  CROSS-VERSION JOIN: star-comprehensive-2021 <-> star-comprehensive-2025",
            bold=True,
        )
        typer.echo(
            "  The join is on CONCEPT. The clause columns are what differ - which is why\n"
            "  a clause-level join is impossible and concept_id exists.\n"
        )
        typer.echo(
            f"  {'concept':<38}{'2021 pp':>9}  {'2021 clauses':<22}{'2025 pp':>9}  2025 clauses"
        )
        typer.echo("  " + "-" * 108)
        identical = 0
        for concept, a_pp, a_cl, b_pp, b_cl in rows:
            a_ids = ",".join(a_cl[:4]) or "-"
            b_ids = ",".join(b_cl[:4]) or "-"
            if a_cl and a_cl == b_cl:
                identical += 1
            typer.echo(
                f"  {concept:<38}{a_pp[:3]!s:>9}  {a_ids[:21]:<22}{b_pp[:3]!s:>9}  {b_ids[:30]}"
            )
        typer.echo("  " + "-" * 108)
        typer.echo(
            f"  {len(rows)} concept(s) present in BOTH documents -> the join key works.\n"
            f"  {identical} of them share an identical clause-id set -> a clause-level join "
            f"would resolve {identical}/{len(rows)}."
        )
    typer.echo("")


if __name__ == "__main__":  # pragma: no cover
    app()
