"""Build-report and T3-harness tests.

Every number asserted here is hand-computed in the test's own docstring, because the
numbers are the deliverable: ``pages_uncovered`` is the evidence for R2 (the two refused
tables on irdai-master-circular-2024 p10-p11 must still be covered by prose) and
``tokens_at_k`` is the cost half of the markdown-vs-row_nl decision. A test that asserted
"whatever build_report returned today" would let both of those drift to nonsense while
staying green, which is precisely the failure the harness exists to prevent.

The chunker, the BM25 index and the lexical retriever live in modules owned elsewhere.
These tests therefore exercise the pure arithmetic directly and the whole-corpus path only
behind ``@pytest.mark.slow`` + ``importorskip``, so the file is green before those land
and stays green after.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from arag.index import build
from arag.index.build import (
    BuildReport,
    SerialisationComparison,
    build_report,
    estimate_tokens,
    index_size_bytes,
    latency_summary,
    measure_queries,
    percentile,
    read_page_census,
    uncovered_pages,
)
from arag.ingest.chunk_types import Chunk, ChunkKind
from arag.ingest.manifest import Manifest
from arag.retrieval.types import DocSpan, RetrievalFilters, RetrievedChunk

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "data" / "manifest" / "sources.jsonl"
RAW_DIR = REPO_ROOT / "data" / "raw"

# The corpus PDFs are gitignored (copyrighted, DESIGN §2), so the full-corpus tests are
# skipped rather than failed where they are absent - a CI checkout has the manifest but
# not the documents. Presence is measured against the PRODUCTION sources, so the
# adversarial fixture PDF (built locally into data/raw) neither satisfies nor breaks the
# count - "6 real documents present" is the condition, by kind not by file tally.
_PRODUCTION_IDS = (
    {s.id for s in Manifest.load(MANIFEST).production_sources} if MANIFEST.exists() else set()
)
CORPUS_PRESENT = bool(_PRODUCTION_IDS) and all(
    (RAW_DIR / f"{sid}.pdf").exists() for sid in _PRODUCTION_IDS
)


def make_chunk(
    chunk_id: str,
    *,
    source_id: str = "star-comprehensive-2025",
    kind: ChunkKind = ChunkKind.PROSE,
    chars: int = 1,
    pages: tuple[int, ...] = (1,),
) -> Chunk:
    """A chunk with a text of exactly ``chars`` characters, so char totals are countable."""
    return Chunk(
        chunk_id=chunk_id,
        source_id=source_id,
        kind=kind,
        text="x" * chars,
        span=DocSpan(document_id=source_id, page=pages[0]),
        pages=pages,
    )


def retrieved(chars: int, *, chunk_id: str = "c") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        span=DocSpan(document_id="doc", page=1),
        text="x" * chars,
    )


class FakeRetriever:
    """A ``Retriever`` that returns canned results and counts calls.

    Canned lists are returned *unsliced* on purpose: the harness must apply ``top_k``
    itself, because a retriever over-returning would otherwise inflate the token estimate
    that the whole T3 decision rests on.
    """

    name = "fake"

    def __init__(
        self,
        results: dict[str, list[RetrievedChunk]],
        *,
        later_calls: list[RetrievedChunk] | None = None,
    ) -> None:
        self.results = results
        self.later_calls = later_calls
        self.calls: list[tuple[str, int]] = []

    async def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
        top_k: int = 30,
    ) -> list[RetrievedChunk]:
        seen_before = any(q == query for q, _ in self.calls)
        self.calls.append((query, top_k))
        if seen_before and self.later_calls is not None:
            return self.later_calls
        return self.results[query]


class FakeIndex:
    """Stands in for ``Bm25Index`` where only its serialised payload matters.

    A fake rather than a real index, because the arithmetic under test is "how many bytes
    is this dict", and hand-computing the JSON of a real postings table would produce a
    magic number nobody could check. The real ``to_dict`` contract is exercised by
    ``TestIndexSizeBytes.test_a_real_index_serialises_to_a_growing_payload``.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def to_dict(self) -> dict[str, Any]:
        return self._payload


# --------------------------------------------------------------------------------------
# percentiles
# --------------------------------------------------------------------------------------


class TestPercentile:
    def test_p95_of_twenty_samples_is_the_nineteenth_smallest(self) -> None:
        """Nearest rank: index = ceil(q * n) - 1.

        n = 20, q = 0.95 -> ceil(19.0) = 19 -> index 18 -> the 19th smallest value.
        With samples 1.0 .. 20.0 that is 19.0, NOT the 20.0 maximum and not an
        interpolated 19.05.
        """
        samples = [float(v) for v in range(20, 0, -1)]
        assert percentile(samples, 0.95) == 19.0

    def test_p95_of_five_samples_is_the_maximum(self) -> None:
        """n = 5, q = 0.95 -> ceil(4.75) = 5 -> index 4 -> the largest sample.

        This is the degenerate case that ``measure_queries(repeats=...)`` exists for: with
        few samples the p95 column is just the worst observation, and reading it as a tail
        latency would overstate typical cost.
        """
        assert percentile([12.0, 4.0, 30.0, 16.0, 8.0], 0.95) == 30.0

    def test_p50_of_an_even_sample_is_the_lower_median_not_the_average(self) -> None:
        """n = 2, q = 0.5 -> ceil(1.0) = 1 -> index 0 -> 10.0.

        Nearest-rank, so 15.0 (the interpolated median of 10 and 20) is deliberately wrong
        here: every reported figure is a latency that actually happened.
        """
        assert percentile([20.0, 10.0], 0.5) == 10.0

    def test_p100_is_the_maximum(self) -> None:
        """n = 3, q = 1.0 -> ceil(3.0) = 3 -> index 2 -> 9.0."""
        assert percentile([1.0, 9.0, 5.0], 1.0) == 9.0

    def test_empty_sample_raises(self) -> None:
        with pytest.raises(ValueError, match="empty sample"):
            percentile([], 0.95)

    @pytest.mark.parametrize("q", [0.0, -0.1, 1.5])
    def test_q_outside_the_open_unit_interval_raises(self, q: float) -> None:
        with pytest.raises(ValueError, match="must be in"):
            percentile([1.0], q)


class TestLatencySummary:
    def test_mean_p50_and_p95_of_a_known_sample(self) -> None:
        """samples = [12, 4, 30, 16, 8] ms, sorted [4, 8, 12, 16, 30], n = 5.

        mean = (12 + 4 + 30 + 16 + 8) / 5 = 70 / 5 = 14.0
        p50  = ceil(0.5 * 5) = 3 -> index 2 -> 12.0
        p95  = ceil(0.95 * 5) = 5 -> index 4 -> 30.0
        """
        assert latency_summary([12.0, 4.0, 30.0, 16.0, 8.0]) == {
            "mean": 14.0,
            "p50": 12.0,
            "p95": 30.0,
        }

    def test_empty_sample_raises_rather_than_reporting_zero(self) -> None:
        """A "mean 0.0 ms" row is indistinguishable from an impossibly fast index."""
        with pytest.raises(ValueError, match="no latency samples"):
            latency_summary([])


# --------------------------------------------------------------------------------------
# token estimation
# --------------------------------------------------------------------------------------


class TestEstimateTokens:
    def test_total_chars_divided_by_four_floored(self) -> None:
        """chars 10 + 21 = 31; 31 // 4 = 7 (7.75 floored)."""
        assert estimate_tokens(["x" * 10, "x" * 21]) == 7

    def test_splitting_a_chunk_does_not_change_the_estimate(self) -> None:
        """Total-then-divide: 30 // 4 = 7 whether the 30 chars arrive as one text or three.

        Divide-then-total would give 7 for one text and 3 * (10 // 4) = 6 for three, which
        would make row_nl look cheaper than markdown purely because it splits more.
        """
        assert estimate_tokens(["x" * 30]) == estimate_tokens(["x" * 10] * 3) == 7

    def test_no_texts_is_zero_tokens(self) -> None:
        assert estimate_tokens([]) == 0


# --------------------------------------------------------------------------------------
# build report
# --------------------------------------------------------------------------------------


class TestBuildReportTotals:
    @staticmethod
    def nine_chunks() -> list[Chunk]:
        """3 prose (10, 20, 30 chars), 2 definition (40, 50), 4 table (60, 70, 5, 5)."""
        return [
            make_chunk("p1", kind=ChunkKind.PROSE, chars=10),
            make_chunk("p2", kind=ChunkKind.PROSE, chars=20),
            make_chunk("p3", kind=ChunkKind.PROSE, chars=30),
            make_chunk("d1", kind=ChunkKind.DEFINITION, chars=40),
            make_chunk("d2", kind=ChunkKind.DEFINITION, chars=50),
            make_chunk("t1", kind=ChunkKind.TABLE_MARKDOWN, chars=60),
            make_chunk("t2", kind=ChunkKind.TABLE_MARKDOWN, chars=70),
            make_chunk("t3", kind=ChunkKind.TABLE_ROW_NL, chars=5),
            make_chunk("t4", kind=ChunkKind.TABLE_ROW_NL, chars=5),
        ]

    def report(self) -> BuildReport:
        return build_report(
            self.nine_chunks(),
            serialisation="markdown",
            documents=1,
            page_census={"star-comprehensive-2025": 1},
            index_bytes=123,
            build_seconds=1.23456,
            tables_refused=2,
        )

    def test_parts_sum_to_the_total(self) -> None:
        """prose 3 + definition 2 + table 4 = 9 = total_chunks."""
        report = self.report()
        assert (report.prose_chunks, report.definition_chunks, report.table_chunks) == (3, 2, 4)
        assert report.prose_chunks + report.definition_chunks + report.table_chunks == 9
        assert report.total_chunks == 9

    def test_total_chars_is_the_sum_of_every_chunk_text(self) -> None:
        """(10 + 20 + 30) + (40 + 50) + (60 + 70 + 5 + 5) = 60 + 90 + 140 = 290."""
        assert self.report().total_chars == 290

    def test_mean_chunk_chars(self) -> None:
        """290 / 9 = 32.222... -> 32.2 to one decimal."""
        assert self.report().mean_chunk_chars == 32.2

    def test_both_table_kinds_land_in_table_chunks(self) -> None:
        """TABLE_MARKDOWN and TABLE_ROW_NL are two kinds and one bucket.

        Counting only TABLE_MARKDOWN would report 2 table chunks out of 4 and understate
        the row_nl arm by exactly the amount that matters to the T3 cost comparison.
        """
        assert self.report().table_chunks == 4

    def test_every_chunk_kind_is_accounted_for_by_some_bucket(self) -> None:
        """The partition guard, stated as a property over ``ChunkKind``.

        ``build_report`` raises when the three buckets do not sum to the total, so a new
        ChunkKind added without a report field fails the build. This test names the reason
        so the failure is not mistaken for an arithmetic bug.
        """
        for kind in ChunkKind:
            assert kind in (ChunkKind.PROSE, ChunkKind.DEFINITION) or kind.is_table

    def test_build_seconds_is_rounded_not_truncated(self) -> None:
        """1.23456 s -> 1.235 s at three decimals."""
        assert self.report().build_seconds == 1.235

    def test_passthrough_fields_are_not_recomputed(self) -> None:
        report = self.report()
        assert report.serialisation == "markdown"
        assert report.documents == 1
        assert report.index_bytes == 123
        assert report.tables_refused == 2

    def test_unknown_serialisation_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown serialisation"):
            build_report(
                [],
                serialisation="csv",
                documents=0,
                page_census={},
                index_bytes=0,
                build_seconds=0.0,
                tables_refused=0,
            )


class TestPageCoverage:
    CIRCULAR = "irdai-master-circular-2024"

    def circular_chunks(self, *, cover_refused_pages: bool) -> list[Chunk]:
        """Prose chunks over the circular's 17 pages.

        The two refused tables sit on p10 and p11, so those two pages arrive as one prose
        chunk spanning both - which is exactly the R2 fallback shape.
        """
        chunks = [
            make_chunk(f"c{page}", source_id=self.CIRCULAR, pages=(page,))
            for page in list(range(1, 10)) + list(range(12, 18))
        ]
        if cover_refused_pages:
            chunks.append(make_chunk("c10-11", source_id=self.CIRCULAR, pages=(10, 11)))
        return chunks

    def test_refused_table_pages_covered_by_prose_leave_no_gap(self) -> None:
        """R2 satisfied: pages 1-9 + (10, 11) + 12-17 = all 17 census pages."""
        gaps = uncovered_pages(self.circular_chunks(cover_refused_pages=True), {self.CIRCULAR: 17})
        assert gaps == {}

    def test_refused_table_pages_with_no_prose_fallback_are_reported(self) -> None:
        """R2 violated: pages 10 and 11 have no chunk at all, so they are named."""
        gaps = uncovered_pages(self.circular_chunks(cover_refused_pages=False), {self.CIRCULAR: 17})
        assert gaps == {self.CIRCULAR: [10, 11]}

    def test_a_fully_covered_source_is_omitted_from_the_report(self) -> None:
        """An empty dict means full coverage, so ``fully_covered`` is a one-line check."""
        report = build_report(
            self.circular_chunks(cover_refused_pages=True),
            serialisation="markdown",
            documents=1,
            page_census={self.CIRCULAR: 17},
            index_bytes=0,
            build_seconds=0.0,
            tables_refused=2,
        )
        assert report.pages_uncovered == {}
        assert report.fully_covered is True

    def test_a_source_with_no_chunks_lists_every_page(self) -> None:
        """nivabupa-rise has 34 pages; none chunked -> pages 1..34, all 34 of them.

        This is the missing-PDF case: build_corpus skips the document with a warning and
        the coverage report states the consequence instead of the build dying.
        """
        gaps = uncovered_pages([], {"nivabupa-rise": 34})
        assert gaps == {"nivabupa-rise": list(range(1, 35))}
        assert len(gaps["nivabupa-rise"]) == 34

    def test_a_chunk_from_a_source_outside_the_census_creates_no_gap(self) -> None:
        """Uncheckable is not the same as uncovered, so 'mystery' is not invented as a gap."""
        chunks = [make_chunk("m1", source_id="mystery", pages=(1,))]
        assert uncovered_pages(chunks, {"nivabupa-rise": 2}) == {"nivabupa-rise": [1, 2]}


class TestPageCensus:
    def test_the_probe_report_is_the_197_page_census(self) -> None:
        """48 + 18 + 34 + 34 + 17 + 46 = 197 pages across 6 documents."""
        census = read_page_census(MANIFEST)
        assert len(census) == 6
        assert census["star-comprehensive-2025"] == 48
        assert census["irdai-master-circular-2024"] == 17
        assert sum(census.values()) == 197

    def test_a_missing_probe_report_is_a_hard_failure(self, tmp_path: Path) -> None:
        """Coverage that cannot be verified must not be reported as verified."""
        with pytest.raises(FileNotFoundError, match="no page census"):
            read_page_census(tmp_path / "sources.jsonl")

    def test_a_malformed_probe_entry_is_named(self, tmp_path: Path) -> None:
        (tmp_path / "probe_report.json").write_text(
            json.dumps([{"source_id": "x"}]), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="missing 'source_id'/'pages'"):
            read_page_census(tmp_path / "sources.jsonl")


# --------------------------------------------------------------------------------------
# index size
# --------------------------------------------------------------------------------------


class TestIndexSizeBytes:
    def test_a_small_payload_is_counted_exactly(self) -> None:
        """json.dumps({"n": 1}, sort_keys=True) == '{"n": 1}'.

        Eight characters - {, ", n, ", :, space, 1, } - all ASCII, so 8 bytes.
        """
        assert index_size_bytes(FakeIndex({"n": 1})) == 8

    def test_non_ascii_terms_are_counted_as_utf8_not_as_escapes(self) -> None:
        """With ensure_ascii=False: '{"₹": 1}' is 8 characters of which ₹ takes 3 bytes,
        so 7 + 3 = 10 bytes.

        The escaped form json.dumps writes by default - '{"\\u20b9": 1}' - would be 13
        bytes for the same index, overstating the shipped payload by 30% on a corpus whose
        terms are full of rupee signs and Devanagari.
        """
        assert index_size_bytes(FakeIndex({"₹": 1})) == 10

    def test_key_order_does_not_change_the_count(self) -> None:
        """sort_keys=True, so two builds of one corpus report one byte count.

        '{"a": 1, "b": 2}' is 16 bytes whichever order the dict was populated in; without
        sorting, an index_bytes delta between two runs could be pure dict ordering.
        """
        assert index_size_bytes(FakeIndex({"a": 1, "b": 2})) == 16
        assert index_size_bytes(FakeIndex({"b": 2, "a": 1})) == 16

    def test_a_real_index_serialises_to_a_growing_payload(self) -> None:
        """The contract check: ``Bm25Index.to_dict()`` exists and its payload grows.

        Monotonicity rather than a magic byte count - a second chunk adds a doc_len entry
        and at least one posting, so the serialised index cannot get smaller. This is the
        assumption most likely to break if the index's persistence format changes.
        """
        lexical = pytest.importorskip("arag.index.lexical")
        index = lexical.Bm25Index()
        index.add(make_chunk("c1", chars=5))
        index.finalise()
        one = index_size_bytes(index)

        bigger = lexical.Bm25Index()
        bigger.add(make_chunk("c1", chars=5))
        bigger.add(make_chunk("c2", chars=5))
        bigger.finalise()

        assert one > 0
        assert index_size_bytes(bigger) > one


# --------------------------------------------------------------------------------------
# query measurement
# --------------------------------------------------------------------------------------


class TestMeasureQueries:
    QUERY_A = "room rent limit for 5 lakh sum insured"
    QUERY_B = "cataract waiting period"

    def retriever(self) -> FakeRetriever:
        """Query A returns 40, 60 and 1000 char hits; query B returns 25 and 25."""
        return FakeRetriever(
            {
                self.QUERY_A: [
                    retrieved(40, chunk_id="a1"),
                    retrieved(60, chunk_id="a2"),
                    retrieved(1000, chunk_id="a3"),
                ],
                self.QUERY_B: [retrieved(25, chunk_id="b1"), retrieved(25, chunk_id="b2")],
            }
        )

    async def test_tokens_at_k_is_the_floored_mean_of_per_query_estimates(self) -> None:
        """top_k = 2.

        query A: 40 + 60 = 100 chars -> 100 // 4 = 25 tokens (the 1000-char third hit is
                 outside top_k and must not be counted)
        query B: 25 + 25 = 50 chars  ->  50 // 4 = 12 tokens (12.5 floored)
        mean:    (25 + 12) // 2 = 37 // 2 = 18 tokens
        """
        measurement = await measure_queries(self.retriever(), [self.QUERY_A, self.QUERY_B], top_k=2)
        assert measurement.tokens_at_k == 18
        assert measurement.top_k == 2
        assert measurement.queries == 2

    async def test_over_returning_retriever_is_sliced_to_top_k(self) -> None:
        """top_k = 3 admits the 1000-char hit: 40 + 60 + 1000 = 1100 -> 1100 // 4 = 275."""
        measurement = await measure_queries(self.retriever(), [self.QUERY_A], top_k=3)
        assert measurement.tokens_at_k == 275

    async def test_top_k_is_passed_through_to_the_retriever(self) -> None:
        retriever = self.retriever()
        await measure_queries(retriever, [self.QUERY_A], top_k=4)
        assert retriever.calls == [(self.QUERY_A, 4)]

    async def test_repeats_time_more_passes_but_cost_is_counted_once(self) -> None:
        """The fake returns a 4000-char hit on any repeat call.

        Were later passes counted, the estimate would move off 18; it must not, because
        cost per query does not change when the same query is timed again.
        """
        retriever = FakeRetriever(
            {
                self.QUERY_A: [retrieved(40, chunk_id="a1"), retrieved(60, chunk_id="a2")],
                self.QUERY_B: [retrieved(25, chunk_id="b1"), retrieved(25, chunk_id="b2")],
            },
            later_calls=[retrieved(4000, chunk_id="huge")],
        )
        measurement = await measure_queries(
            retriever, [self.QUERY_A, self.QUERY_B], top_k=2, repeats=2
        )
        assert measurement.tokens_at_k == 18
        assert measurement.queries == 2
        assert len(retriever.calls) == 4

    async def test_latency_reports_three_statistics(self) -> None:
        measurement = await measure_queries(self.retriever(), [self.QUERY_A], top_k=2)
        assert sorted(measurement.latency_ms) == ["mean", "p50", "p95"]
        assert all(value >= 0.0 for value in measurement.latency_ms.values())

    async def test_no_queries_raises(self) -> None:
        with pytest.raises(ValueError, match="no queries to measure"):
            await measure_queries(self.retriever(), [], top_k=2)

    async def test_zero_repeats_raises(self) -> None:
        with pytest.raises(ValueError, match="repeats must be >= 1"):
            await measure_queries(self.retriever(), [self.QUERY_A], repeats=0)


# --------------------------------------------------------------------------------------
# the T3 comparison record
# --------------------------------------------------------------------------------------


def fixture_report(serialisation: str, *, total_chunks: int, total_chars: int) -> BuildReport:
    return BuildReport(
        serialisation=serialisation,
        documents=6,
        prose_chunks=total_chunks,
        definition_chunks=0,
        table_chunks=0,
        total_chunks=total_chunks,
        total_chars=total_chars,
        index_bytes=total_chars,
        build_seconds=1.0,
        tables_refused=2,
        pages_uncovered={},
    )


class TestSerialisationComparison:
    def comparison(self) -> SerialisationComparison:
        """The measured corpus shape: 222 markdown chunks vs 1735 row_nl chunks.

        Token figures are chosen to make the harness's point: row_nl emits ~8x the chunks
        yet sends *less* context at the same top_k, because its chunks are single rows.
        """
        return SerialisationComparison(
            markdown=fixture_report("markdown", total_chunks=222, total_chars=600_000),
            row_nl=fixture_report("row_nl", total_chunks=1735, total_chars=420_000),
            markdown_latency_ms={"mean": 8.0, "p50": 7.0, "p95": 12.0},
            row_nl_latency_ms={"mean": 11.0, "p50": 10.0, "p95": 19.0},
            markdown_tokens_at_k=1500,
            row_nl_tokens_at_k=480,
            top_k=6,
            queries=2,
        )

    def test_chunk_ratio(self) -> None:
        """1735 / 222 = 7.815315... (222 * 7 = 1554, remainder 181; 181 / 222 = 0.815315...)."""
        assert self.comparison().chunk_ratio == pytest.approx(7.815315, abs=1e-6)

    def test_token_ratio_is_not_the_chunk_ratio(self) -> None:
        """480 / 1500 = 0.32.

        8x the chunks and 0.32x the context at k. Inferring cost from the chunk count -
        the shortcut this field exists to prevent - would have been wrong by a factor
        of 24.
        """
        comparison = self.comparison()
        assert comparison.token_ratio == pytest.approx(0.32, abs=1e-12)
        assert comparison.token_ratio < 1.0 < comparison.chunk_ratio

    def test_ratios_are_zero_rather_than_undefined_on_an_empty_markdown_arm(self) -> None:
        """A ZeroDivisionError inside a reporting property would lose the whole comparison."""
        empty = SerialisationComparison(
            markdown=fixture_report("markdown", total_chunks=0, total_chars=0),
            row_nl=fixture_report("row_nl", total_chunks=1735, total_chars=420_000),
            markdown_latency_ms={"mean": 0.0, "p50": 0.0, "p95": 0.0},
            row_nl_latency_ms={"mean": 1.0, "p50": 1.0, "p95": 1.0},
            markdown_tokens_at_k=0,
            row_nl_tokens_at_k=480,
            top_k=6,
            queries=1,
        )
        assert empty.chunk_ratio == 0.0
        assert empty.token_ratio == 0.0


# --------------------------------------------------------------------------------------
# graceful degradation while the collaborating modules land
# --------------------------------------------------------------------------------------


class TestMissingCollaborators:
    def test_build_corpus_without_the_chunker_raises_runtimeerror(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """RuntimeError naming ``arag.ingest.chunk``, not ImportError.

        ``_MISSING_MODULES`` is monkeypatched rather than relying on the module genuinely
        being absent, so this test asserts the same thing before and after the chunker
        lands. tmp_path holds no manifest, which also pins that the check fires before any
        filesystem work.
        """
        monkeypatch.setitem(
            build._MISSING_MODULES, build.CHUNKER_MODULE, "No module named 'arag.ingest.chunk'"
        )
        with pytest.raises(RuntimeError) as excinfo:
            build.build_corpus(tmp_path / "sources.jsonl", tmp_path / "raw")

        assert build.CHUNKER_MODULE in str(excinfo.value)
        assert not isinstance(excinfo.value, ImportError)

    def test_the_error_lists_every_missing_module_at_once(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """One message, not a fix-one-retry loop through three modules."""
        monkeypatch.setitem(build._MISSING_MODULES, build.CHUNKER_MODULE, "absent")
        monkeypatch.setitem(build._MISSING_MODULES, build.LEXICAL_INDEX_MODULE, "absent")
        with pytest.raises(RuntimeError) as excinfo:
            build.build_corpus(tmp_path / "sources.jsonl", tmp_path / "raw")

        message = str(excinfo.value)
        assert build.CHUNKER_MODULE in message
        assert build.LEXICAL_INDEX_MODULE in message

    def test_compare_serialisations_names_the_missing_retriever(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setitem(
            build._MISSING_MODULES,
            build.LEXICAL_RETRIEVER_MODULE,
            "No module named 'arag.retrieval.lexical'",
        )
        with pytest.raises(RuntimeError) as excinfo:
            build.compare_serialisations(tmp_path / "sources.jsonl", tmp_path / "raw", ["q"])

        assert build.LEXICAL_RETRIEVER_MODULE in str(excinfo.value)
        assert not isinstance(excinfo.value, ImportError)

    def test_no_error_when_nothing_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The degradation path must not fire once the collaborators exist."""
        monkeypatch.setattr(build, "_MISSING_MODULES", {})
        assert build._require(build.CHUNKER_MODULE, build.LEXICAL_INDEX_MODULE) is None

    def test_compare_serialisations_rejects_an_empty_query_list(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(build, "_MISSING_MODULES", {})
        with pytest.raises(ValueError, match="at least one query"):
            build.compare_serialisations(tmp_path / "sources.jsonl", tmp_path / "raw", [])


# --------------------------------------------------------------------------------------
# full corpus (nightly)
# --------------------------------------------------------------------------------------


T3_QUERIES = [
    "what is the room rent limit for a 5 lakh sum insured policy",
    "how long is the waiting period for cataract surgery",
    "is bariatric surgery covered",
]


@pytest.mark.slow
@pytest.mark.skipif(not CORPUS_PRESENT, reason="data/raw PDFs are gitignored; not in a checkout")
class TestFullCorpusBuild:
    def test_every_page_of_every_document_is_covered(self) -> None:
        """R2 over the real corpus: all 197 pages, including the refused table regions."""
        pytest.importorskip("pymupdf")
        pytest.importorskip("arag.ingest.chunk")
        pytest.importorskip("arag.index.lexical")

        corpus = build.build_corpus(MANIFEST, RAW_DIR, serialisation="markdown")

        assert corpus.report.documents == 6
        assert corpus.report.total_chunks == len(corpus.chunks)
        assert corpus.report.pages_uncovered == {}
        # 2 headerless tables are known-refused on irdai-master-circular-2024 p10-p11.
        assert corpus.report.tables_refused >= 2

    def test_the_adversarial_fixture_is_unreachable_from_the_default_corpus(self) -> None:
        """A poisoned document must not reach a retriever by accident.

        Right now the only thing between a payload and the index is that nobody wired it in.
        That is not a guarantee: the fixture is declared in the manifest and its PDF is in
        data/raw, so a `build_corpus` that iterated `sources` instead of `production_sources`
        would chunk it and put four prompt-injection payloads one cosine-similarity away from
        a user's question. This asserts the default build excludes it BY KIND, and that the
        only way in is the deliberate `include_adversarial=True` opt-in.
        """
        pytest.importorskip("pymupdf")
        pytest.importorskip("arag.ingest.chunk")
        pytest.importorskip("arag.index.lexical")

        fixtures = {s.id for s in Manifest.load(MANIFEST).adversarial_fixtures}
        assert fixtures, "this test is vacuous unless the manifest declares a fixture"

        default = build.build_corpus(MANIFEST, RAW_DIR, serialisation="markdown")
        assert not (
            {c.source_id for c in default.chunks.values()} & fixtures
        ), "a fixture reached the default corpus - the production_sources exclusion is broken"

        opted_in = build.build_corpus(
            MANIFEST, RAW_DIR, serialisation="markdown", include_adversarial=True
        )
        assert {c.source_id for c in opted_in.chunks.values()} & fixtures, (
            "include_adversarial=True did not surface the fixture - the opt-in is the only "
            "path in, so if it does not work the fixture can never be evaluated at all"
        )

    def test_row_nl_emits_far_more_table_chunks_than_markdown(self) -> None:
        """The 8x effect is in the TABLE chunks: 1735 vs 222 measured, a ratio of 7.8.

        Asserted on ``table_chunk_ratio``, not ``chunk_ratio``. Both arms produce prose and
        definitions identically (802 of them), so corpus-wide the same effect measures only
        2537 / 1024 = 2.5 - and an assertion of "> 4.0" on that figure fails against a
        perfectly healthy build. Pinning the wrong one of the two ratios is the specific
        mistake ``chunk_ratio``'s docstring warns about, so the two bounds are asserted
        together here, with the dilution between them stated as the relationship it is.

        Bounds are loose deliberately: the exact counts move with the chunker's size
        budget, the order of magnitude is the finding.
        """
        pytest.importorskip("pymupdf")
        pytest.importorskip("arag.ingest.chunk")
        pytest.importorskip("arag.index.lexical")
        pytest.importorskip("arag.retrieval.lexical")

        comparison = build.compare_serialisations(MANIFEST, RAW_DIR, T3_QUERIES, top_k=6)

        assert comparison.row_nl.table_chunks > comparison.markdown.table_chunks
        assert comparison.table_chunk_ratio > 4.0
        # Diluted by the prose and definitions the serialisation does not touch, so
        # strictly between 1x and the table-region effect.
        assert 1.0 < comparison.chunk_ratio < comparison.table_chunk_ratio
        assert comparison.markdown_tokens_at_k > 0
        assert comparison.row_nl_tokens_at_k > 0
        assert comparison.markdown.pages_uncovered == {}
        assert comparison.row_nl.pages_uncovered == {}

    def test_eight_times_the_chunks_is_not_eight_times_the_context(self) -> None:
        """The finding the whole harness exists to produce.

        Measured at k=6: 1393 row_nl tokens vs 1450 markdown, a token ratio of 0.96 against
        a table-chunk ratio of 7.8. So row_nl sends slightly LESS context than markdown
        while emitting ~8x the chunks, because each of its chunks is a single row. The
        chunk-count shortcut would have predicted 8x the bill and been wrong by ~8x.

        Asserted as a bound rather than an equality: the token figure moves with the
        chunker, but "not proportional to the chunk count" is the claim being defended.
        """
        pytest.importorskip("pymupdf")
        pytest.importorskip("arag.ingest.chunk")
        pytest.importorskip("arag.index.lexical")
        pytest.importorskip("arag.retrieval.lexical")

        comparison = build.compare_serialisations(MANIFEST, RAW_DIR, T3_QUERIES, top_k=6)

        assert comparison.token_ratio < 2.0
        assert comparison.token_ratio < comparison.table_chunk_ratio / 2
