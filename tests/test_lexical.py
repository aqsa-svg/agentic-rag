"""BM25 tests whose expected values were computed by hand, on paper, first.

The point of this file is not that ``search`` returns something. It is that the number it
returns is the number BM25 defines. The project claims "true BM25 with tunable k1/b", and
a test asserting against whatever the implementation happens to produce today would
certify a wrong formula just as happily as a right one - it would lock the bug in rather
than catch it. So every expected value below is written out as arithmetic in the
docstring, and the literal it evaluates to is asserted too, which means a change to either
the formula or the tokeniser has to be argued for rather than merely re-recorded.

The three parameter tests (idf, b, k1) exist for the same reason. Each drives its
parameter to a value where the correct answer is derivable without BM25 at all:

* ``b = 0`` removes length normalisation, so a term appearing once must score exactly its
  idf no matter how long the document is;
* ``k1 = 0`` collapses the saturation term, so a document containing a term three times
  must score exactly the same as one containing it once;
* idf must fall monotonically as document frequency rises, and must never go negative.

If a knob is silently ignored, one of those degenerate cases breaks. A test that only
checked "k1=2.0 gives different numbers than k1=1.5" would pass for an implementation that
multiplied by k1 in the wrong place.
"""

from __future__ import annotations

import json
from datetime import date
from math import log

import pytest

from arag.index.lexical import BYTES_PER_POSTING, Bm25Index, tokenise
from arag.ingest.chunk_types import Chunk, ChunkKind
from arag.retrieval.lexical import LexicalRetriever
from arag.retrieval.protocols import Retriever
from arag.retrieval.types import ChunkMeta, DocSpan, RetrievalFilters

# The toy corpus. Token counts after tokenise (lower-cased, split on non-alphanumerics,
# nothing shorter than two characters kept - every word below survives):
#   c1 -> waiting period for cataract surgery                             = 5 tokens
#   c2 -> cataract surgery is covered after the waiting period ends       = 9 tokens
#   c3 -> maternity expenses are excluded                                 = 4 tokens
# N = 3, total = 18, avgdl = 6.0
TOY = {
    "c1": "waiting period for cataract surgery",
    "c2": "cataract surgery is covered after the waiting period ends",
    "c3": "maternity expenses are excluded",
}

LN_1_6 = log(1.6)  # idf of a term in 2 of 3 documents
LN_1_2 = log(1.2)  # idf of a term in 2 of 2 documents

# Two documents of exactly six tokens each, so avgdl = 6.0 and the length term is 1 for
# both whatever b is - which isolates the term-frequency saturation the k1 tests probe.
# The only difference is f(grace): 3 in a1, 1 in a2. n(grace) = 2 of N = 2, so
# idf = ln(1 + 0.5/2.5) = ln(1.2) = 0.1823215567939546.
SAME_LENGTH = {
    "a1": "grace period grace period grace period",
    "a2": "grace waiting period cover benefit limit",
}


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    document_id: str = "star-comprehensive-2025",
    page: int = 1,
    kind: ChunkKind = ChunkKind.PROSE,
    meta: ChunkMeta | None = None,
) -> Chunk:
    """A chunk carrying only the fields the index and the filters read.

    A helper rather than a fixture: these tests turn on *specific* corpora, and a fixture
    returning a canned one would hide at the assertion site which documents are being
    scored - which is exactly the information a reader needs to check the arithmetic.
    """
    return Chunk(
        chunk_id=chunk_id,
        source_id=document_id,
        kind=kind,
        text=text,
        span=DocSpan(document_id=document_id, page=page),
        pages=(page,),
        meta=meta or ChunkMeta(),
    )


def build(corpus: dict[str, str], *, k1: float = 1.5, b: float = 0.75) -> Bm25Index:
    index = Bm25Index(k1=k1, b=b)
    for chunk_id, text in corpus.items():
        index.add(make_chunk(chunk_id, text))
    index.finalise()
    return index


class TestTokenise:
    def test_lowercases_and_splits_on_non_alphanumerics(self) -> None:
        assert tokenise("Pre-existing Disease (PED): 36 months") == [
            "pre",
            "existing",
            "disease",
            "ped",
            "36",
            "months",
        ]

    def test_folds_the_ligature_that_would_otherwise_break_every_benefit_query(self) -> None:
        """S5 measured 269 ligature codepoints in this corpus.

        U+FB01 is one character that renders as "fi". Without NFKC folding, the index would
        hold a token no query can produce, and the miss would look like a relevance
        problem. This is the single most load-bearing line in the tokeniser.
        """
        assert tokenise("Beneﬁt Payable") == ["benefit", "payable"]
        assert tokenise("Beneﬁt") == tokenise("Benefit")

    def test_drops_tokens_shorter_than_two_characters(self) -> None:
        assert tokenise("a policy") == ["policy"]

    def test_underscore_separates_rather_than_glues(self) -> None:
        """``clause_4_2`` must not become one token, and the two digits fall to the length
        floor, leaving only the word.
        """
        assert tokenise("clause_4_2") == ["clause"]

    def test_a_monetary_figure_tokenises_into_its_digit_groups(self) -> None:
        """A known and accepted limitation, asserted so it cannot change unnoticed.

        Indian digit grouping plus the two-character floor means "5,00,000" indexes as
        {00, 000}: the leading "5" is dropped and the groups never rejoin. BM25 therefore
        cannot answer a sum-insured question by term overlap - that is what the table
        chunks and the reranker are for. Pretending otherwise would be the false claim.
        """
        assert tokenise("5,00,000/-") == ["00", "000"]

    @pytest.mark.parametrize("text", ["", "   ", "a", "%", "​"])
    def test_text_with_nothing_indexable_yields_no_tokens(self, text: str) -> None:
        assert tokenise(text) == []


class TestIdf:
    def test_hand_computed_idf_values(self) -> None:
        """N = 3.

        cataract  in c1, c2   -> n = 2: ln(1 + (3-2+0.5)/(2+0.5)) = ln(1.6)   = 0.470003629
        maternity in c3       -> n = 1: ln(1 + (3-1+0.5)/(1+0.5)) = ln(8/3)   = 0.980829253
        surgery   in c1, c2   -> n = 2: ln(1.6)                               = 0.470003629
        """
        index = build(TOY)

        assert index.doc_frequency("cataract") == 2
        assert index.doc_frequency("maternity") == 1
        assert index.idf("cataract") == pytest.approx(0.47000362924573563)
        assert index.idf("maternity") == pytest.approx(log(8 / 3))
        assert index.idf("maternity") == pytest.approx(0.9808292530117263)
        assert index.idf("surgery") == index.idf("cataract")

    def test_idf_falls_as_a_term_appears_in_more_documents(self) -> None:
        """Three documents; ``rare`` in one, ``mid`` in two, ``common`` in all three.

        n = 1: ln(1 + 2.5/1.5) = ln(8/3)   = 0.980829253
        n = 2: ln(1 + 1.5/2.5) = ln(1.6)   = 0.470003629
        n = 3: ln(1 + 0.5/3.5) = ln(8/7)   = 0.133531393
        """
        index = build(
            {
                "r1": "rare mid common alpha",
                "r2": "mid common beta gamma",
                "r3": "common delta epsilon zeta",
            }
        )

        assert index.idf("rare") == pytest.approx(0.9808292530117263)
        assert index.idf("mid") == pytest.approx(0.47000362924573563)
        assert index.idf("common") == pytest.approx(0.13353139262452257)
        assert index.idf("rare") > index.idf("mid") > index.idf("common")

    def test_idf_stays_positive_for_a_term_in_every_document(self) -> None:
        """Why the ``1 +`` inside the logarithm is not decoration.

        The classic idf, ln((N - n + 0.5)/(n + 0.5)), is negative for n = 3, N = 3:
        ln(0.5/3.5) = -1.945. A negative idf makes a common term *penalise* the documents
        containing it, so ranking has to be rescued by an undocumented clamp at zero. The
        variant used here needs no clamp, and this assertion is what keeps it that way.
        """
        index = build({"r1": "common alpha", "r2": "common beta", "r3": "common gamma"})
        assert index.idf("common") > 0.0


class TestBm25Score:
    def test_hand_computed_single_term_score(self) -> None:
        """Query "cataract" against the toy corpus, k1 = 1.5, b = 0.75.

        N = 3, avgdl = (5 + 9 + 4) / 3 = 6.0
        n(cataract) = 2, f(cataract, c1) = f(cataract, c2) = 1

        idf   = ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1.6) = 0.47000362924573563
        numer = f * (k1 + 1) = 1 * 2.5 = 2.5

        c1, |d| = 5:
            denom = f + k1 * (1 - b + b * |d|/avgdl)
                  = 1 + 1.5 * (1 - 0.75 + 0.75 * 5/6)
                  = 1 + 1.5 * (0.25 + 0.625) = 1 + 1.5 * 0.875 = 2.3125
            score = 0.47000362924573563 * 2.5 / 2.3125
                  = 0.47000362924573563 * 1.0810810810810811
                  = 0.5081120316170116

        c2, |d| = 9:
            denom = 1 + 1.5 * (0.25 + 0.75 * 9/6) = 1 + 1.5 * 1.375 = 3.0625
            score = 0.47000362924573563 * 2.5 / 3.0625
                  = 0.47000362924573563 * 0.8163265306122449
                  = 0.3836764320373352

        c3 contains neither token, so it must be absent rather than present with 0.0.
        """
        index = build(TOY)

        result = index.search("cataract")

        assert [chunk_id for chunk_id, _ in result] == ["c1", "c2"]
        assert result[0][1] == pytest.approx(LN_1_6 * 2.5 / 2.3125)
        assert result[0][1] == pytest.approx(0.5081120316170116)
        assert result[1][1] == pytest.approx(LN_1_6 * 2.5 / 3.0625)
        assert result[1][1] == pytest.approx(0.3836764320373352)

    def test_multi_term_score_is_the_sum_over_query_terms(self) -> None:
        """Query "waiting period" against c1.

        Both terms are in c1 and c2 only, so both have n = 2 and idf = ln(1.6), and both
        occur once in c1 - the same f, the same |d|, the same idf as "cataract" above.
        The score is therefore exactly twice the single-term score:

            2 * 0.5081120316170116 = 1.0162240632340231
        """
        index = build(TOY)

        result = dict(index.search("waiting period"))

        assert result["c1"] == pytest.approx(1.0162240632340231)
        assert result["c1"] == pytest.approx(2 * 0.5081120316170116)
        assert "c3" not in result

    def test_a_repeated_query_term_is_not_counted_twice(self) -> None:
        """The formula carries no query-term-frequency factor, so "cataract cataract" must
        score identically to "cataract". Counting the repeat would make score depend on how
        emphatically a user typed.
        """
        index = build(TOY)
        assert index.search("cataract cataract") == index.search("cataract")

    def test_a_term_absent_from_the_corpus_matches_nothing(self) -> None:
        index = build(TOY)
        assert index.search("dental") == []

    def test_top_k_truncates_after_ranking(self) -> None:
        index = build(TOY)
        assert [chunk_id for chunk_id, _ in index.search("cataract", top_k=1)] == ["c1"]

    def test_equal_scores_break_ties_on_chunk_id(self) -> None:
        """Two chunks with identical text score identically. Without a tie-break the order
        would follow dict insertion, and an eval number would move when ingest order
        changed - drift for a non-reason.
        """
        index = build({"zz": "cataract surgery", "aa": "cataract surgery"})
        assert [chunk_id for chunk_id, _ in index.search("cataract")] == ["aa", "zz"]


class TestLengthNormalisation:
    def test_b_zero_removes_the_length_penalty_entirely(self) -> None:
        """With b = 0 the length term collapses to 1 - 0 + 0 = 1, so for f = 1:

            score = idf * (1 * 2.5) / (1 + 1.5 * 1) = idf * 2.5 / 2.5 = idf = ln(1.6)

        c1 (5 tokens) and c2 (9 tokens) must therefore score *the same*, which is the
        cleanest possible proof that b is the only thing that reads document length.
        """
        index = build(TOY, b=0.0)

        result = dict(index.search("cataract"))

        assert result["c1"] == result["c2"]
        assert result["c1"] == pytest.approx(LN_1_6)
        assert result["c1"] == pytest.approx(0.47000362924573563)

    def test_default_b_penalises_the_longer_document(self) -> None:
        """Same term, same frequency, different lengths: 5 tokens vs 9, avgdl 6.

        c1: idf * 2.5 / 2.3125 = idf * 1.0810810810810811   (shorter than average, boosted)
        c2: idf * 2.5 / 3.0625 = idf * 0.8163265306122449   (longer than average, damped)
        ratio c1/c2 = 3.0625 / 2.3125 = 1.3243243243243243
        """
        index = build(TOY)

        result = dict(index.search("cataract"))

        assert result["c1"] > result["c2"]
        assert result["c1"] / result["c2"] == pytest.approx(3.0625 / 2.3125)
        assert result["c1"] / result["c2"] == pytest.approx(1.3243243243243243)


class TestSaturation:
    def test_k1_zero_makes_term_frequency_irrelevant(self) -> None:
        """With k1 = 0 the saturation term degenerates:

            score = idf * (f * (0 + 1)) / (f + 0 * length_norm) = idf * f / f = idf

        So a document containing "grace" three times scores *exactly* the same as one
        containing it once: ln(1.2) = 0.1823215567939546. Asserted with ``==`` rather than
        ``approx`` because f/f is exact in binary floating point, and an implementation
        that had k1 anywhere but where the formula puts it would miss by more than an ulp.
        """
        index = build(SAME_LENGTH, k1=0.0)

        result = dict(index.search("grace"))

        assert result["a1"] == result["a2"]
        assert result["a1"] == LN_1_2
        assert result["a1"] == pytest.approx(0.1823215567939546)

    def test_default_k1_rewards_repetition_but_saturates_it(self) -> None:
        """Same corpus, k1 = 1.5, length term = 1 for both documents.

        a1, f = 3: idf * (3 * 2.5) / (3 + 1.5) = idf * 7.5 / 4.5 = idf * 1.6666666666666667
        a2, f = 1: idf * (1 * 2.5) / (1 + 1.5) = idf * 2.5 / 2.5 = idf

        Three times the term frequency buys 1.667x the score, not 3x. That gap *is* the
        saturation BM25 is chosen for, so the ratio is asserted, not just the ordering.
        """
        index = build(SAME_LENGTH)

        result = dict(index.search("grace"))

        assert result["a1"] == pytest.approx(0.30386926132325764)
        assert result["a2"] == pytest.approx(LN_1_2)
        assert result["a1"] / result["a2"] == pytest.approx(5 / 3)


class TestPersistence:
    def test_round_trip_through_json_preserves_scores_exactly(self) -> None:
        """``==``, not ``approx``: ``to_dict`` writes integer counts only and ``from_dict``
        recomputes idf and avgdl from them, so the reloaded index performs bit-identical
        arithmetic. Serialising the floats instead would put decimal formatting between
        two runs of the same query.
        """
        original = build(TOY)
        expected = original.search("cataract surgery waiting")

        restored = Bm25Index.from_dict(json.loads(json.dumps(original.to_dict())))

        assert restored.search("cataract surgery waiting") == expected
        assert restored.stats == original.stats

    def test_round_trip_preserves_the_tuning_parameters(self) -> None:
        """k1 and b are the whole point of owning the implementation; an index that
        reloaded with the defaults would silently discard a tuning result.
        """
        original = build(TOY, k1=0.9, b=0.35)

        restored = Bm25Index.from_dict(original.to_dict())

        assert (restored.k1, restored.b) == (0.9, 0.35)
        assert restored.search("cataract") == original.search("cataract")

    def test_an_index_written_by_another_format_version_is_refused(self) -> None:
        raw = build(TOY).to_dict()
        raw["version"] = 99

        with pytest.raises(ValueError, match="unsupported bm25 index format"):
            Bm25Index.from_dict(raw)


class TestStats:
    def test_hand_computed_stats(self) -> None:
        """Corpus: s1 = "grace period", s2 = "grace waiting". Both 2 tokens, avgdl = 2.0.

        postings: grace -> {s1, s2}, period -> {s1}, waiting -> {s2}  = 4 postings
        n_terms = 3 (grace, period, waiting)
        term bytes  = len("grace") + len("period") + len("waiting") = 5 + 6 + 7 = 18
        id bytes    = len("s1") + len("s2")                         = 2 + 2     = 4
        bytes_estimate = 18 + 4 + 12 * 4 = 70
        """
        index = build({"s1": "grace period", "s2": "grace waiting"})

        assert BYTES_PER_POSTING == 12
        assert index.stats == {
            "n_chunks": 2,
            "n_terms": 3,
            "avgdl": 2.0,
            "postings": 4,
            "bytes_estimate": 70,
        }


class TestIndexLifecycle:
    def test_search_before_finalise_refuses_to_guess(self) -> None:
        """Scoring without avgdl would divide by zero or, worse, silently normalise
        against a stale average. Raising is the only honest option.
        """
        index = Bm25Index()
        index.add(make_chunk("c1", "cataract surgery"))

        with pytest.raises(RuntimeError, match="before finalise"):
            index.search("cataract")

    def test_adding_after_finalise_invalidates_the_index(self) -> None:
        index = build(TOY)
        index.add(make_chunk("c4", "cataract cover"))

        with pytest.raises(RuntimeError, match="before finalise"):
            index.search("cataract")

        index.finalise()
        assert len(index.search("cataract")) == 3

    def test_a_repeated_chunk_id_is_refused(self) -> None:
        index = Bm25Index()
        index.add(make_chunk("c1", "cataract surgery"))

        with pytest.raises(ValueError, match="already indexed"):
            index.add(make_chunk("c1", "cataract surgery"))

    def test_an_empty_index_returns_no_results_rather_than_raising(self) -> None:
        index = Bm25Index()
        index.finalise()

        assert index.search("cataract") == []
        assert index.stats["n_chunks"] == 0
        assert index.stats["avgdl"] == 0.0

    def test_a_chunk_with_no_indexable_token_still_counts_towards_the_corpus(self) -> None:
        """ "5 %" tokenises to nothing (the digit is one character). The chunk is counted in
        N and in avgdl because it *is* a document in the corpus; it is simply unreachable
        lexically, which ``add`` logs as a warning.
        """
        index = build({"c1": "cataract surgery", "c2": "5 %"})

        assert index.stats["n_chunks"] == 2
        assert index.stats["avgdl"] == 1.0
        assert [chunk_id for chunk_id, _ in index.search("cataract")] == ["c1"]

    @pytest.mark.parametrize("query", ["", "   ", "a"])
    def test_a_query_with_no_tokens_returns_nothing(self, query: str) -> None:
        assert build(TOY).search(query) == []

    def test_non_positive_top_k_returns_nothing(self) -> None:
        assert build(TOY).search("cataract", top_k=0) == []


class TestLexicalRetriever:
    @staticmethod
    def retriever(chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75) -> LexicalRetriever:
        index = Bm25Index(k1=k1, b=b)
        for chunk in chunks:
            index.add(chunk)
        index.finalise()
        return LexicalRetriever(index, {chunk.chunk_id: chunk for chunk in chunks})

    def test_satisfies_the_retriever_protocol(self) -> None:
        assert isinstance(self.retriever([make_chunk("c1", "cataract surgery")]), Retriever)

    async def test_sets_lexical_score_and_orders_best_first(self) -> None:
        """The same toy corpus as the hand-computed score test, so the two expected values
        are the ones already derived there: 0.5081120316170116 then 0.3836764320373352.
        """
        chunks = [make_chunk(chunk_id, text) for chunk_id, text in TOY.items()]

        results = await self.retriever(chunks).retrieve("cataract")

        assert [result.chunk_id for result in results] == ["c1", "c2"]
        assert results[0].lexical_score == pytest.approx(0.5081120316170116)
        assert results[1].lexical_score == pytest.approx(0.3836764320373352)
        assert results[0].lexical_score is not None
        assert results[1].lexical_score is not None
        assert results[0].lexical_score > results[1].lexical_score
        # The other score slots stay empty: attribution across stages is the reason
        # RetrievedChunk keeps them separate.
        assert results[0].dense_score is None
        assert results[0].rerank_score is None
        assert results[0].text == TOY["c1"]
        assert results[0].span.document_id == "star-comprehensive-2025"

    async def test_document_ids_filter_excludes_other_documents(self) -> None:
        chunks = [
            make_chunk("a1", "cataract surgery cover", document_id="star-comprehensive-2025"),
            make_chunk("b1", "cataract surgery cover", document_id="irdai-master-circular-2024"),
        ]

        results = await self.retriever(chunks).retrieve(
            "cataract", filters=RetrievalFilters(document_ids=("irdai-master-circular-2024",))
        )

        assert [result.chunk_id for result in results] == ["b1"]

    async def test_the_filter_runs_before_top_k_not_after(self) -> None:
        """The bug this ordering exists to prevent.

        Three short chunks in one document out-score a long chunk in another. Asking for
        top_k = 1 scoped to the second document must return that document's chunk. An
        implementation that scored first, cut to 1, then filtered would return an empty
        list and look exactly like "no such clause".
        """
        chunks = [
            make_chunk("a1", "cataract surgery", document_id="star-comprehensive-2025"),
            make_chunk("a2", "cataract surgery", document_id="star-comprehensive-2025"),
            make_chunk("a3", "cataract surgery", document_id="star-comprehensive-2025"),
            make_chunk(
                "b1",
                "cataract surgery is covered after the stated waiting period has elapsed "
                "in full and every other condition of this clause is satisfied",
                document_id="irdai-master-circular-2024",
            ),
        ]

        results = await self.retriever(chunks).retrieve(
            "cataract",
            filters=RetrievalFilters(document_ids=("irdai-master-circular-2024",)),
            top_k=1,
        )

        assert [result.chunk_id for result in results] == ["b1"]

    async def test_insurer_and_product_filters_apply(self) -> None:
        chunks = [
            make_chunk(
                "a1",
                "cataract surgery cover",
                meta=ChunkMeta(insurer="Star Health", product="Comprehensive"),
            ),
            make_chunk(
                "b1",
                "cataract surgery cover",
                meta=ChunkMeta(insurer="HDFC Ergo", product="Optima Secure"),
            ),
        ]
        retriever = self.retriever(chunks)

        by_insurer = await retriever.retrieve(
            "cataract", filters=RetrievalFilters(insurer="Star Health")
        )
        by_product = await retriever.retrieve(
            "cataract", filters=RetrievalFilters(product="Optima Secure")
        )

        assert [result.chunk_id for result in by_insurer] == ["a1"]
        assert [result.chunk_id for result in by_product] == ["b1"]

    async def test_tables_only_keeps_table_chunks_from_either_signal(self) -> None:
        """``meta.is_table`` and ``kind.is_table`` are supposed to agree. The filter takes
        either, so a chunk whose metadata flag was missed at ingest is still returned to
        the one query type that asked specifically for tables.
        """
        chunks = [
            make_chunk("p1", "cataract surgery prose"),
            make_chunk("t1", "cataract surgery table row", kind=ChunkKind.TABLE_MARKDOWN),
            make_chunk("t2", "cataract surgery grid", meta=ChunkMeta(is_table=True)),
        ]

        results = await self.retriever(chunks).retrieve(
            "cataract", filters=RetrievalFilters(tables_only=True)
        )

        assert sorted(result.chunk_id for result in results) == ["t1", "t2"]

    async def test_as_of_excludes_a_clause_outside_its_validity_window(self) -> None:
        """The supersession control. ``old`` was replaced on 2025-03-31, ``new`` took effect
        on 2025-04-01, so a query as of 2025-06-01 must see only ``new``.
        """
        chunks = [
            make_chunk(
                "old",
                "cataract surgery waiting period twenty four months",
                meta=ChunkMeta(
                    effective_from=date(2021, 4, 1),
                    effective_to=date(2025, 3, 31),
                    superseded_by="new",
                ),
            ),
            make_chunk(
                "new",
                "cataract surgery waiting period twelve months",
                meta=ChunkMeta(effective_from=date(2025, 4, 1)),
            ),
        ]
        retriever = self.retriever(chunks)

        current = await retriever.retrieve(
            "cataract", filters=RetrievalFilters(as_of=date(2025, 6, 1))
        )
        historical = await retriever.retrieve(
            "cataract", filters=RetrievalFilters(as_of=date(2023, 6, 1))
        )

        assert [result.chunk_id for result in current] == ["new"]
        assert [result.chunk_id for result in historical] == ["old"]

    async def test_a_chunk_with_no_dates_is_visible_at_any_as_of(self) -> None:
        """Most of the corpus carries no effective dates. Treating "undated" as "not in
        force" would empty the corpus for every ``as_of`` query, which is the failure mode
        of an over-eager temporal filter.
        """
        chunks = [make_chunk("c1", "cataract surgery cover")]

        results = await self.retriever(chunks).retrieve(
            "cataract", filters=RetrievalFilters(as_of=date(2025, 6, 1))
        )

        assert [result.chunk_id for result in results] == ["c1"]

    async def test_filters_that_match_nothing_return_empty_not_the_whole_corpus(self) -> None:
        """Guards the ``None`` (no restriction) versus empty-set (nothing matched)
        distinction. Conflating them turns a filter that excludes everything into a search
        over everything - a wrong answer instead of no answer.
        """
        chunks = [make_chunk("c1", "cataract surgery cover")]

        results = await self.retriever(chunks).retrieve(
            "cataract", filters=RetrievalFilters(insurer="Nobody At All")
        )

        assert results == []

    @pytest.mark.parametrize("query", ["", "   "])
    async def test_an_empty_query_returns_empty_rather_than_raising(self, query: str) -> None:
        """The protocol requires the abstain path to be exercised, not a 500."""
        chunks = [make_chunk("c1", "cataract surgery cover")]

        assert await self.retriever(chunks).retrieve(query) == []

    async def test_an_empty_corpus_returns_empty(self) -> None:
        index = Bm25Index()
        index.finalise()

        assert await LexicalRetriever(index, {}).retrieve("cataract") == []

    def test_a_chunk_store_that_does_not_cover_the_index_is_refused(self) -> None:
        """An index and a store from different ingest runs would cite text that was never
        indexed. Caught at construction rather than mid-request.
        """
        index = Bm25Index()
        index.add(make_chunk("c1", "cataract surgery"))
        index.finalise()

        with pytest.raises(ValueError, match="chunk store is missing 1 of 1"):
            LexicalRetriever(index, {})
