"""Table tests: T1 stitching, T2 header guarantee, T3 serialisations.

Tables are the load-bearing ingest problem in this corpus (165 candidates on 131 of 197
pages), so each of the three rules gets its own tests, because each fails independently:

* T1 fails silently — an unstitched table looks complete and is missing rows.
* T2 fails catastrophically — a header-less row is a number with no meaning.
* T3 is a measurement, so the tests only pin that both serialisations exist and behave.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import pytest

from arag.ingest.tables import (
    COLUMN_X_TOL,
    PROSE_HEADER_CHARS,
    ExtractedTable,
    HeaderlessTableError,
    StitchClass,
    TableChunk,
    TableFragment,
    classify_stitch,
    classify_stitches,
    stitch,
    table_chunks,
    to_markdown,
    to_row_sentences,
)

PAGE_H = 842.0


def frag(
    page: int,
    *,
    rows: tuple[tuple[str, ...], ...],
    header: tuple[str, ...] = ("Sum Insured", "Room Rent Limit"),
    y0: float = 100.0,
    y1: float = 400.0,
    x0: float = 50.0,
    x1: float = 550.0,
    col_xs: tuple[float, ...] = (50.0, 300.0),
    external: bool = True,
    index: int = 0,
) -> TableFragment:
    return TableFragment(
        source_id="doc",
        page=page,
        index_on_page=index,
        bbox=(x0, y0, x1, y1),
        col_xs=col_xs,
        header=header,
        header_external=external,
        rows=rows,
        page_height=PAGE_H,
    )


def table(
    *,
    header: tuple[str, ...] = ("Sum Insured", "Room Rent Limit"),
    rows: tuple[tuple[str, ...], ...] = (("5,00,000", "1% of SI"), ("10,00,000", "2% of SI")),
    pages: tuple[int, ...] = (7,),
) -> ExtractedTable:
    return ExtractedTable(
        table_id="doc:p7:t0", source_id="doc", pages=pages, header=header, rows=rows
    )


class TestT1Stitching:
    """A page-spanning table must become one table, not two plausible halves."""

    def test_continuation_is_merged(self) -> None:
        a = frag(7, rows=(("5,00,000", "1%"),), y0=400.0, y1=800.0)
        b = frag(8, rows=(("10,00,000", "2%"),), y0=70.0, y1=300.0)
        out = stitch([a, b])
        assert len(out) == 1
        assert out[0].pages == (7, 8)
        assert out[0].stitched_from == 2
        assert len(out[0].rows) == 2
        assert out[0].spans_pages

    def test_repeated_header_on_the_continuation_is_dropped(self) -> None:
        """Policy wordings repeat the header on each page. Keeping it would inject a
        header row into the middle of the data, where it reads as a row of values.
        """
        a = frag(7, rows=(("5,00,000", "1%"),), y0=400.0, y1=800.0)
        b = frag(
            8,
            rows=(("Sum Insured", "Room Rent Limit"), ("10,00,000", "2%")),
            y0=70.0,
            y1=300.0,
        )
        out = stitch([a, b])
        assert len(out) == 1
        assert out[0].rows == (("5,00,000", "1%"), ("10,00,000", "2%"))
        assert out[0].header_repeats_dropped == 1

    def test_non_adjacent_pages_are_not_merged(self) -> None:
        a = frag(7, rows=(("5,00,000", "1%"),), y0=400.0, y1=800.0)
        b = frag(9, rows=(("10,00,000", "2%"),), y0=70.0, y1=300.0)
        assert len(stitch([a, b])) == 2

    def test_table_not_reaching_the_bottom_is_not_continued(self) -> None:
        """A table that ends mid-page has finished. Whatever is at the top of the next
        page is a different table.
        """
        a = frag(7, rows=(("5,00,000", "1%"),), y0=100.0, y1=300.0)
        b = frag(8, rows=(("10,00,000", "2%"),), y0=70.0, y1=300.0)
        assert len(stitch([a, b])) == 2

    def test_different_column_count_is_not_merged(self) -> None:
        a = frag(7, rows=(("5,00,000", "1%"),), y0=400.0, y1=800.0)
        b = frag(
            8,
            header=("A", "B", "C"),
            rows=(("x", "y", "z"),),
            y0=70.0,
            y1=300.0,
            col_xs=(50.0, 200.0, 400.0),
        )
        assert len(stitch([a, b])) == 2

    def test_misaligned_columns_are_not_merged(self) -> None:
        """The discriminating check.

        Two unrelated tables on consecutive pages routinely share page adjacency and
        column count. Only their x-geometry separates them, which is why column positions
        are compared and not just counted.
        """
        a = frag(7, rows=(("5,00,000", "1%"),), y0=400.0, y1=800.0, col_xs=(50.0, 300.0))
        b = frag(
            8,
            rows=(("10,00,000", "2%"),),
            y0=70.0,
            y1=300.0,
            col_xs=(50.0 + COLUMN_X_TOL * 3, 300.0 + COLUMN_X_TOL * 3),
        )
        assert len(stitch([a, b])) == 2

    def test_three_page_table_merges_into_one(self) -> None:
        frags = [
            frag(7, rows=(("a", "1"),), y0=400.0, y1=800.0),
            frag(8, rows=(("b", "2"),), y0=70.0, y1=800.0),
            frag(9, rows=(("c", "3"),), y0=70.0, y1=300.0),
        ]
        out = stitch(frags)
        assert len(out) == 1
        assert out[0].pages == (7, 8, 9)
        assert len(out[0].rows) == 3


class TestT2HeaderGuarantee:
    """A chunk cannot exist without a header. Structural, not remembered."""

    def test_chunk_construction_refuses_an_empty_header(self) -> None:
        with pytest.raises(HeaderlessTableError, match="no header"):
            TableChunk(
                chunk_id="c1",
                table_id="doc:p1:t0",
                source_id="doc",
                pages=(1,),
                text="| 25,000 |",
                serialisation="markdown",
                header=("", ""),
                row_span=(0, 0),
            )

    def test_headerless_table_is_refused_not_degraded(self) -> None:
        bare = table(header=("", ""))
        with pytest.raises(HeaderlessTableError):
            table_chunks(bare, serialisation="markdown")
        with pytest.raises(HeaderlessTableError):
            table_chunks(bare, serialisation="row_nl")

    def test_error_message_explains_the_consequence(self) -> None:
        """A labeller or maintainer must understand WHY this is fatal, or they will
        "fix" it by loosening the check.
        """
        with pytest.raises(HeaderlessTableError) as exc:
            table_chunks(table(header=("",)), serialisation="markdown")
        assert "number with no meaning" in str(exc.value)
        assert "per-eye limit" in str(exc.value)

    @pytest.mark.parametrize("mode", ["markdown", "row_nl"])
    def test_every_derived_chunk_carries_the_header(self, mode: str) -> None:
        chunks = table_chunks(table(), serialisation=mode)
        assert chunks
        for chunk in chunks:
            assert any(cell.strip() for cell in chunk.header)

    def test_oversized_table_splits_with_the_header_repeated(self) -> None:
        """This is what "never split a table" was too coarse to express.

        A table larger than the chunk budget MUST split, and every group must carry the
        header - otherwise the second half is unusable.
        """
        rows = tuple((f"{i},00,000", f"{i}% of SI", "x" * 60) for i in range(1, 40))
        big = ExtractedTable(
            table_id="doc:p7:t0",
            source_id="doc",
            pages=(7,),
            header=("Sum Insured", "Limit", "Notes"),
            rows=rows,
        )
        chunks = table_chunks(big, serialisation="markdown", max_chars=600)
        assert len(chunks) > 1, "an oversized table must split"
        for chunk in chunks:
            assert chunk.header == ("Sum Insured", "Limit", "Notes")
            assert "Sum Insured" in chunk.text, "the header must be IN the chunk text"
            assert chunk.meta["split"] is True
        # Row coverage must be complete and non-overlapping.
        spans = sorted(c.row_span for c in chunks)
        assert spans[0][0] == 0
        assert spans[-1][1] == len(rows) - 1
        for (_, end), (start, _) in pairwise(spans):
            assert start == end + 1

    def test_empty_table_yields_no_chunks(self) -> None:
        assert table_chunks(table(rows=()), serialisation="markdown") == []


class TestT3Serialisations:
    def test_markdown_includes_header_and_every_row(self) -> None:
        md = to_markdown(table())
        assert "| Sum Insured | Room Rent Limit |" in md
        assert "5,00,000" in md and "10,00,000" in md
        assert "page 7" in md, "provenance must travel with the chunk"

    def test_markdown_escapes_pipes_in_cells(self) -> None:
        md = to_markdown(table(rows=(("a|b", "1%"),)))
        assert "a/b" in md, "an unescaped pipe would corrupt the table structure"

    def test_row_sentences_name_the_header_inline(self) -> None:
        """The property that should make this serialisation win.

        A query for "room rent limit for 5 lakh sum insured" shares real tokens with the
        sentence and almost nothing with "| 5,00,000 | 1% of SI |".
        """
        sentences = to_row_sentences(table())
        assert len(sentences) == 2
        assert "For Sum Insured 5,00,000" in sentences[0]
        assert "Room Rent Limit is 1% of SI" in sentences[0]
        assert "doc" in sentences[0] and "page 7" in sentences[0]

    def test_row_sentences_skip_empty_rows(self) -> None:
        assert len(to_row_sentences(table(rows=(("", ""), ("5,00,000", "1%"))))) == 1

    def test_caption_as_header_is_recovered_so_both_axes_survive(self) -> None:
        """h-10: find_tables took the TITLE row as the header, losing the column axes.

        The star-2025 Delivery table has a title row ("Delivery and New Born") that
        find_tables returned AS the header, pushing the real column names - Sum Insured,
        Normal Delivery, Delivery by Ceasarean Section - into data rows. Every column then
        serialised as "value is X", so "caesarean" never appeared and a query for it matched
        nothing; row-NL missed h-10. The serialiser now promotes the real column-name rows
        (merging a column group over its sub-header) and keeps the title as the caption.
        """
        caption_headed = table(
            header=("Delivery and New Born", "", "", ""),
            rows=(
                ("Sum Insured Rs.", "Limit for Delivery", "", "New Born liability"),
                ("", "Normal Delivery Rs.", "Delivery by Ceasarean Section Rs.", ""),
                ("10,00,000 to 25,00,000", "30,000", "50,000", "1,00,000"),
            ),
        )
        sentences = to_row_sentences(caption_headed)
        # one DATA row only; the two header rows were consumed, not serialised as data
        assert len(sentences) == 1
        s = sentences[0]
        # both axes present: the sum-insured band AND the caesarean column name + its value
        assert "10,00,000 to 25,00,000" in s
        assert "Ceasarean" in s and "50,000" in s
        assert "value is" not in s, "columns must be named, not labelled 'value'"
        # the title survives as context, carrying the subject term for retrieval
        assert "Delivery and New Born" in s

    def test_an_ordinary_header_is_left_untouched(self) -> None:
        """The recovery only fires on a single-cell caption; a normal table is unchanged."""
        sentences = to_row_sentences(table())
        assert "For Sum Insured 5,00,000" in sentences[0]
        assert "Room Rent Limit is 1% of SI" in sentences[0]

    def test_row_nl_produces_one_chunk_per_row(self) -> None:
        rows = tuple((f"{i}", f"{i}%") for i in range(5))
        chunks = table_chunks(table(rows=rows), serialisation="row_nl")
        assert len(chunks) == 5
        assert all(c.serialisation == "row_nl" for c in chunks)
        assert [c.row_span for c in chunks] == [(i, i) for i in range(5)]

    def test_markdown_keeps_a_small_table_whole(self) -> None:
        chunks = table_chunks(table(), serialisation="markdown")
        assert len(chunks) == 1
        assert chunks[0].meta["split"] is False

    def test_unknown_serialisation_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown serialisation"):
            table_chunks(table(), serialisation="yaml")

    def test_page_spanning_flag_reaches_the_chunk(self) -> None:
        """A reviewer must be able to tell a stitched table's chunks apart from a
        single-page table's, because stitching is the step most likely to be wrong.
        """
        spanning = table(pages=(7, 8))
        for mode in ("markdown", "row_nl"):
            for chunk in table_chunks(spanning, serialisation=mode):
                assert chunk.meta["spans_pages"] is True
                assert chunk.pages == (7, 8)


class TestAgainstRealCorpus:
    @pytest.fixture
    def probe_report(self, repo_root: Path) -> list[dict[str, object]]:
        path = repo_root / "data" / "manifest" / "probe_report.json"
        if not path.exists():
            pytest.skip("run `arag-ingest probe` first")
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]

    def test_the_corpus_really_is_table_heavy(self, probe_report: list[dict[str, object]]) -> None:
        """Pins the measurement that made tables the priority. If this drops sharply, the
        table work has stopped being the right place to spend effort.
        """
        total = sum(int(d["table_candidates"]) for d in probe_report)  # type: ignore[arg-type]
        assert total > 100

    @pytest.mark.slow
    def test_extraction_and_chunking_over_the_real_corpus(self, repo_root: Path) -> None:
        """End-to-end on real PDFs: stitching runs, and every chunk has a header.

        Two tables in the IRDAI circular have no detectable header and are REFUSED. That
        is the intended behaviour, so the test asserts the refusal count rather than
        requiring perfect extraction - per the standing rule that tables stop when the
        header and stitch guarantees hold, not when extraction is flawless.
        """
        pymupdf = pytest.importorskip("pymupdf")
        raw = repo_root / "data" / "raw"
        if not raw.exists() or not any(raw.glob("*.pdf")):
            pytest.skip("corpus not fetched")

        from arag.ingest.tables import extract_tables

        stitched = refused = chunk_count = 0
        for pdf in sorted(raw.glob("*.pdf")):
            doc = pymupdf.open(pdf)
            with doc:
                tables, report = extract_tables(pdf.stem, doc)
            stitched += report.stitched
            for t in tables:
                try:
                    for mode in ("markdown", "row_nl"):
                        for chunk in table_chunks(t, serialisation=mode):
                            assert any(h.strip() for h in chunk.header)
                            chunk_count += 1
                except HeaderlessTableError:
                    refused += 1

        assert stitched > 0, "page-spanning tables exist in this corpus and must be merged"
        assert chunk_count > 500
        assert refused <= 4, (
            "a rising refusal count means extraction regressed; investigate before "
            "loosening the header guarantee"
        )


class TestStitchClassification:
    """Pins 'N genuine / M detected' as an assertion, not a reporting convention.

    A stitched region's header separates cleanly by length on this corpus: real column
    headers measure at most 57 characters, mis-detected prose paragraphs 313 to 2286.
    The threshold sits in a 250-character gap.

    The first version of this classifier used 45 characters and was WRONG - it called
    legitimately verbose column names prose and reported 4 genuine grids instead of 7.
    These tests exist so that mistake cannot recur silently in either direction.
    """

    def test_verbose_but_real_column_names_are_a_data_grid(self) -> None:
        """The false positive that broke the first classifier."""
        cls, why = classify_stitch(
            table(
                header=(
                    "Base Sum Insured",
                    "1st paid Claim",
                    "ReAssure+ is triggered when the base sum insured is exhausted",
                )
            )
        )
        assert cls is StitchClass.DATA_GRID, why
        assert cls.is_reliable

    def test_caption_only_header_is_its_own_class(self) -> None:
        """T2 passes - a header exists - but the header is uninformative, so a row
        sentence reads "For Table - B2 27%: ..." and cannot be retrieved. Naming this
        as a distinct class is what keeps the T2 guarantee honestly scoped.
        """
        cls, why = classify_stitch(table(header=("Table - B2",)))
        assert cls is StitchClass.CAPTION_HEADER
        assert not cls.is_reliable
        assert "caption" in why

    def test_paragraph_header_is_spurious(self) -> None:
        cls, why = classify_stitch(table(header=("x" * (PROSE_HEADER_CHARS + 1),)))
        assert cls is StitchClass.SPURIOUS
        assert "paragraph" in why

    def test_empty_header_is_spurious(self) -> None:
        assert classify_stitch(table(header=("", "")))[0] is StitchClass.SPURIOUS

    def test_threshold_sits_inside_the_measured_gap(self) -> None:
        """Real headers max at 57 chars, prose starts at 313. A threshold outside
        that gap would be a tuned parameter rather than a measured separation.
        """
        assert 57 < PROSE_HEADER_CHARS < 313

    def test_single_page_tables_are_not_classified(self) -> None:
        single = table()
        assert single.stitched_from == 1
        grouped = classify_stitches([single])
        assert all(not entries for entries in grouped.values())

    @pytest.mark.slow
    def test_real_corpus_stitch_counts(self, repo_root: Path) -> None:
        """The counts themselves, pinned.

        16 page-spanning tables are detected; only 7 are trustworthy data grids, 3 have
        caption-only headers, and 6 are prose that find_tables mis-detected. Reporting
        "16 stitched" would overstate the result almost 2.3x, so the split is asserted.

        If these numbers move, extraction changed and the claim in the write-up needs
        re-deriving - which is the point of pinning them.
        """
        pymupdf = pytest.importorskip("pymupdf")
        raw = repo_root / "data" / "raw"
        if not raw.exists() or not any(raw.glob("*.pdf")):
            pytest.skip("corpus not fetched")

        from arag.ingest.tables import extract_tables

        totals = dict.fromkeys(StitchClass, 0)
        for pdf in sorted(raw.glob("*.pdf")):
            doc = pymupdf.open(pdf)
            with doc:
                tables, _ = extract_tables(pdf.stem, doc)
            for cls, entries in classify_stitches(tables).items():
                totals[cls] += len(entries)

        detected = sum(totals.values())
        assert detected == 16, f"stitch detection changed: {totals}"
        assert totals[StitchClass.DATA_GRID] == 7, f"genuine grid count changed: {totals}"
        assert totals[StitchClass.CAPTION_HEADER] == 3, f"caption headers changed: {totals}"
        assert totals[StitchClass.SPURIOUS] == 6, f"spurious stitches changed: {totals}"
