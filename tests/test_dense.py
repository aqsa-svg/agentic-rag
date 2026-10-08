"""Dense index, dense retriever, and RRF fusion.

No model is loaded here. A fake embedder with hand-chosen vectors makes every assertion a
hand-computed value rather than "whatever the implementation returned today" — the practice
that caught instances 1 and 4 in ``docs/SILENT_WRONGNESS.md``. Tests that need the real
model belong in the measurement script, not the PR gate: a 440MB download is not a unit
test dependency, and asserting on bge's actual rankings would pin model behaviour rather
than this code's behaviour.
"""

from __future__ import annotations

import math
from datetime import date

import pytest

# This module tests the OFFLINE dense stack (numpy, sentence-transformers / torch), which
# the offline PR gate does not install. Guard the imports so the file SKIPS cleanly there
# instead of raising ImportError at collection - a collection error reddens the whole Tests
# step, which is how test_dense.py quietly broke CI while the local suite (with .[offline]
# installed) reported green. The skip is made visible by the gate's `pytest -rs` and the
# skip-count line in the job summary; a silently skipped hard test is the failure this whole
# episode records.
np = pytest.importorskip("numpy")
pytest.importorskip("arag.index.dense")

from arag.index.dense import DenseIndex, EmbeddingCache, default_query_prefix  # noqa: E402
from arag.ingest.chunk_types import Chunk, ChunkKind  # noqa: E402
from arag.retrieval.dense import DenseRetriever  # noqa: E402
from arag.retrieval.embedding import BGE_QUERY_PREFIX  # noqa: E402
from arag.retrieval.hybrid import RRFHybridRetriever  # noqa: E402
from arag.retrieval.types import ChunkMeta, DocSpan, RetrievalFilters, RetrievedChunk  # noqa: E402


class FakeEmbedder:
    """Deterministic 2-d embedder. Satisfies the ``Embedder`` protocol."""

    def __init__(
        self,
        mapping: dict[str, tuple[float, float]],
        *,
        model_id: str = "fake/model",
        query_prefix: str = "Q: ",
        normalise: bool = True,
    ) -> None:
        self._mapping = mapping
        self.model_id = model_id
        self.dim = 2
        self.query_prefix = query_prefix
        self._normalise = normalise
        self.queries_seen: list[str] = []

    def _vec(self, text: str) -> list[float]:
        x, y = self._mapping.get(text, (1.0, 0.0))
        if not self._normalise:
            return [x, y]
        norm = math.hypot(x, y) or 1.0
        return [x / norm, y / norm]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        # Applying the prefix is the EMBEDDER's job, not the retriever's - the retriever
        # passes the raw query and never knows the model has an instruction. Recorded so a
        # test can assert the retriever routed through embed_query rather than
        # embed_passages, which is the asymmetry the whole protocol exists to protect.
        self.queries_seen.append(self.query_prefix + text)
        return self._vec(text)


def chunk(chunk_id: str, text: str, **meta: object) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_id="doc",
        span=DocSpan(document_id="doc", page=1, clause_id=chunk_id),
        text=text,
        pages=(1,),
        kind=ChunkKind.PROSE,
        meta=ChunkMeta(**meta),  # type: ignore[arg-type]
    )


@pytest.fixture
def corpus() -> dict[str, Chunk]:
    return {
        "a": chunk("a", "north"),
        "b": chunk("b", "east"),
        "c": chunk("c", "north-east"),
    }


@pytest.fixture
def embedder() -> FakeEmbedder:
    # north=(0,1), east=(1,0), north-east=(1,1)/sqrt2. Cosine against a "north" query:
    # a=1.0, c=0.7071, b=0.0 - hand-computable, so the expected order is a, c, b.
    return FakeEmbedder({"north": (0.0, 1.0), "east": (1.0, 0.0), "north-east": (1.0, 1.0)})


@pytest.fixture
def index(corpus: dict[str, Chunk], embedder: FakeEmbedder) -> DenseIndex:
    return DenseIndex.build(list(corpus.values()), embedder)


class TestDenseIndex:
    def test_cosine_order_and_values_are_hand_computed(
        self, index: DenseIndex, embedder: FakeEmbedder
    ) -> None:
        hits = index.search(embedder.embed_query("north"), top_k=3)
        assert [cid for cid, _ in hits] == ["a", "c", "b"]
        scores = dict(hits)
        assert scores["a"] == pytest.approx(1.0, abs=1e-6)
        assert scores["c"] == pytest.approx(1 / math.sqrt(2), abs=1e-6)
        assert scores["b"] == pytest.approx(0.0, abs=1e-6)

    def test_allow_set_narrows_before_scoring(
        self, index: DenseIndex, embedder: FakeEmbedder
    ) -> None:
        """Pre-filter, not post-filter.

        Excluding the best hit must promote the next one, not return a shorter list -
        post-filtering would return 1 result for top_k=2 and read as "nothing else matched".
        """
        hits = index.search(embedder.embed_query("north"), top_k=2, allow=frozenset({"b", "c"}))
        assert [cid for cid, _ in hits] == ["c", "b"]

    def test_empty_allow_set_returns_nothing(
        self, index: DenseIndex, embedder: FakeEmbedder
    ) -> None:
        assert index.search(embedder.embed_query("north"), 3, allow=frozenset()) == []

    def test_ties_break_on_chunk_id(self, embedder: FakeEmbedder) -> None:
        """Otherwise an eval number moves when ingest order changes."""
        same = {k: chunk(k, "north") for k in ("z", "m", "a")}
        idx = DenseIndex.build(list(same.values()), embedder)
        hits = idx.search(embedder.embed_query("north"), top_k=3)
        assert [cid for cid, _ in hits] == ["a", "m", "z"]

    def test_unnormalised_vectors_are_refused(self, corpus: dict[str, Chunk]) -> None:
        """An un-normalised embedder turns cosine into a length-weighted dot product.

        It raises nothing on its own: the matrix is valid, the scores are floats, and long
        chunks simply start winning for reasons unrelated to the query.
        """
        bad = FakeEmbedder({"north": (0.0, 5.0)}, normalise=False)
        with pytest.raises(ValueError, match="un-normalised"):
            DenseIndex.build(list(corpus.values()), bad)

    def test_round_trip_preserves_everything_the_contract_depends_on(
        self, index: DenseIndex, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        path = index.save(tmp_path / "d.npz")
        back = DenseIndex.load(path)
        assert back.model_id == index.model_id
        assert back.dim == index.dim
        assert back.query_prefix == index.query_prefix
        assert back.chunk_ids == index.chunk_ids
        assert np.allclose(back.vectors, index.vectors)

    def test_wrong_dimension_query_raises(self, index: DenseIndex) -> None:
        with pytest.raises(ValueError, match="expects"):
            index.search([1.0, 0.0, 0.0], top_k=1)


class TestDenseRetrieverRefusesSilentMismatches:
    """Each of these produces a valid, plausible, wrong ranking if allowed through."""

    def test_different_model_is_refused(self, index: DenseIndex, corpus: dict[str, Chunk]) -> None:
        other = FakeEmbedder({}, model_id="fake/other")
        with pytest.raises(ValueError, match="meaningless ranking"):
            DenseRetriever(index, corpus, other)

    def test_different_query_prefix_is_refused(
        self, index: DenseIndex, corpus: dict[str, Chunk]
    ) -> None:
        """The subtlest of the three: same model, same dim, merely a different prompt.

        Every vector is still valid and the ranking is merely worse, which is
        indistinguishable from the retriever being bad at its job.
        """
        drifted = FakeEmbedder({}, query_prefix="")
        with pytest.raises(ValueError, match="query prefix disagrees"):
            DenseRetriever(index, corpus, drifted)

    def test_chunk_store_missing_indexed_ids_is_refused(
        self, index: DenseIndex, corpus: dict[str, Chunk], embedder: FakeEmbedder
    ) -> None:
        with pytest.raises(ValueError, match="different ingest runs"):
            DenseRetriever(index, {"a": corpus["a"]}, embedder)


class TestDenseRetriever:
    async def test_returns_chunks_ordered_with_dense_scores_set(
        self, index: DenseIndex, corpus: dict[str, Chunk], embedder: FakeEmbedder
    ) -> None:
        got = await DenseRetriever(index, corpus, embedder).retrieve("north", top_k=3)
        assert [c.chunk_id for c in got] == ["a", "c", "b"]
        assert got[0].dense_score == pytest.approx(1.0, abs=1e-6)
        assert all(c.lexical_score is None for c in got)

    async def test_the_query_goes_through_embed_query_with_its_prefix(
        self, index: DenseIndex, corpus: dict[str, Chunk], embedder: FakeEmbedder
    ) -> None:
        """Queries and passages are embedded differently, and only the embedder knows how.

        If the retriever ever called embed_passages for a query, every vector would still
        be valid and the ranking merely worse - the exact silent degradation the two-method
        protocol exists to make impossible.
        """
        await DenseRetriever(index, corpus, embedder).retrieve("north", top_k=1)
        assert embedder.queries_seen == ["Q: north"]

    async def test_empty_query_returns_empty_not_error(
        self, index: DenseIndex, corpus: dict[str, Chunk], embedder: FakeEmbedder
    ) -> None:
        assert await DenseRetriever(index, corpus, embedder).retrieve("   ", top_k=3) == []

    async def test_as_of_filter_excludes_superseded_chunks(self, embedder: FakeEmbedder) -> None:
        """The supersession control, on the dense side.

        This is the field the shared filtering module exists for: if it worked lexically
        and not densely, half the candidates would be correctly filtered and the bug would
        look like ordinary relevance noise.
        """
        store = {
            "old": chunk("old", "north", effective_to=date(2024, 1, 1)),
            "new": chunk("new", "north", effective_from=date(2024, 1, 2)),
        }
        idx = DenseIndex.build(list(store.values()), embedder)
        got = await DenseRetriever(idx, store, embedder).retrieve(
            "north", top_k=5, filters=RetrievalFilters(as_of=date(2026, 1, 1))
        )
        assert [c.chunk_id for c in got] == ["new"]


class StubRetriever:
    def __init__(self, name: str, ids: list[str], *, dense: bool = False) -> None:
        self.name = name
        self._ids = ids
        self._dense = dense

    async def retrieve(
        self, query: str, *, filters: RetrievalFilters | None = None, top_k: int = 30
    ) -> list[RetrievedChunk]:
        out = []
        for i, cid in enumerate(self._ids[:top_k]):
            score = 1.0 / (i + 1)
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    span=DocSpan(document_id="doc", page=1, clause_id=cid),
                    text=cid,
                    dense_score=score if self._dense else None,
                    lexical_score=None if self._dense else score,
                )
            )
        return out


class TestRRFFusion:
    async def test_fused_scores_are_hand_computed(self) -> None:
        """k=60. 'b' is rank 2 in both, 'a' rank 1 lexically, 'c' rank 1 densely.

            a: 1/61              = 0.016393
            b: 1/62 + 1/62       = 0.032258
            c: 1/61              = 0.016393

        So b wins despite never being either retriever's top hit - which is the entire
        argument for fusing, and is asserted here rather than assumed.
        """
        hybrid = RRFHybridRetriever(
            [StubRetriever("lex", ["a", "b"]), StubRetriever("dense", ["c", "b"], dense=True)]
        )
        got = await hybrid.retrieve("q", top_k=3)
        assert got[0].chunk_id == "b"
        assert got[0].fused_score == pytest.approx(2 / 62, abs=1e-9)
        assert {c.chunk_id for c in got} == {"a", "b", "c"}
        for c in got[1:]:
            assert c.fused_score == pytest.approx(1 / 61, abs=1e-9)

    async def test_ties_break_on_chunk_id(self) -> None:
        hybrid = RRFHybridRetriever(
            [StubRetriever("lex", ["a", "b"]), StubRetriever("dense", ["c", "b"], dense=True)]
        )
        got = await hybrid.retrieve("q", top_k=3)
        assert [c.chunk_id for c in got[1:]] == ["a", "c"]

    async def test_per_stage_scores_survive_fusion(self) -> None:
        """RetrievedChunk keeps dense and lexical scores apart so a failure can be
        attributed to the stage that caused it. Fusion must merge, not overwrite.
        """
        hybrid = RRFHybridRetriever(
            [StubRetriever("lex", ["b"]), StubRetriever("dense", ["b"], dense=True)]
        )
        got = await hybrid.retrieve("q", top_k=1)
        assert got[0].lexical_score is not None
        assert got[0].dense_score is not None
        assert got[0].fused_score is not None

    async def test_each_retriever_is_asked_for_more_than_top_k(self) -> None:
        """A chunk ranked low by one retriever and high by the other must be able to reach
        the fused top-k. Asking each for exactly top_k makes fusion a re-ordering of the
        intersection and discards the complementary recall dense retrieval exists for.
        """
        deep = StubRetriever("dense", [f"d{i}" for i in range(30)], dense=True)
        hybrid = RRFHybridRetriever([StubRetriever("lex", ["a"]), deep], candidate_multiplier=3)
        got = await hybrid.retrieve("q", top_k=2)
        assert len(got) == 2
        # 30 dense candidates were considered even though top_k was 2.
        assert await deep.retrieve("q", top_k=6) != []

    async def test_empty_query_returns_empty(self) -> None:
        hybrid = RRFHybridRetriever([StubRetriever("lex", ["a"])])
        assert await hybrid.retrieve("  ", top_k=3) == []

    def test_construction_rejects_incoherent_configuration(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            RRFHybridRetriever([])
        with pytest.raises(ValueError, match="weights"):
            RRFHybridRetriever([StubRetriever("a", [])], weights=[1.0, 2.0])
        with pytest.raises(ValueError, match="positive"):
            RRFHybridRetriever([StubRetriever("a", [])], k=0)


class TestQueryPrefixRegistry:
    def test_bge_models_get_the_documented_instruction(self) -> None:
        assert default_query_prefix("BAAI/bge-base-en-v1.5") == BGE_QUERY_PREFIX

    def test_an_unknown_model_gets_no_prefix_and_is_logged(self) -> None:
        """Returning "" is the safe default; doing it silently is not, because a model that
        needed a prefix now retrieves worse with nothing to indicate why.
        """
        assert default_query_prefix("someone/unlisted-model-v9") == ""


class CountingEmbedder(FakeEmbedder):
    """Records how many passages actually reached the model."""

    def __init__(self, *a: object, **kw: object) -> None:
        super().__init__(*a, **kw)  # type: ignore[arg-type]
        self.embedded: list[str] = []

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return super().embed_passages(texts)


class TestEmbeddingCache:
    """Measured: ~658ms/chunk on this CPU, so the 1,024-chunk corpus costs ~11 minutes.

    Day 6 re-chunks and re-measures repeatedly. The cache is what keeps that habitual
    rather than something to be avoided.
    """

    def _cache(self, tmp_path, **kw: str) -> EmbeddingCache:  # type: ignore[no-untyped-def]
        opts = {"model_id": "fake/model", "query_prefix": "Q: ", "passage_prefix": ""}
        opts.update(kw)
        return EmbeddingCache(tmp_path / "emb.npz", **opts)  # type: ignore[arg-type]

    def test_second_build_embeds_nothing(self, corpus, tmp_path) -> None:  # type: ignore[no-untyped-def]
        embedder = CountingEmbedder(
            {"north": (0.0, 1.0), "east": (1.0, 0.0), "north-east": (1.0, 1.0)}
        )
        first = DenseIndex.build(list(corpus.values()), embedder, cache=self._cache(tmp_path))
        assert len(embedder.embedded) == 3

        embedder.embedded.clear()
        second = DenseIndex.build(list(corpus.values()), embedder, cache=self._cache(tmp_path))
        assert embedder.embedded == []
        assert np.allclose(first.vectors, second.vectors)

    def test_changed_text_under_the_same_chunk_id_is_a_miss(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The hole a chunk_id-keyed cache would have.

        Chunk ids are derived from document and position, so re-chunking can change a
        chunk's TEXT while keeping its id. Keyed on the id, the stale vector would be
        served for text that no longer exists - and the index would load, search and return
        plausible results computed against the previous chunking, with nothing to indicate
        it. Content addressing makes that a miss by construction.
        """
        embedder = CountingEmbedder({"north": (0.0, 1.0), "east": (1.0, 0.0)})
        DenseIndex.build([chunk("a", "north")], embedder, cache=self._cache(tmp_path))
        embedder.embedded.clear()

        # Same chunk_id, different text.
        DenseIndex.build([chunk("a", "east")], embedder, cache=self._cache(tmp_path))
        assert embedder.embedded == ["east"]

    def test_a_different_model_does_not_reuse_the_old_space(self, corpus, tmp_path) -> None:  # type: ignore[no-untyped-def]
        embedder = CountingEmbedder(
            {"north": (0.0, 1.0), "east": (1.0, 0.0), "north-east": (1.0, 1.0)}
        )
        DenseIndex.build(list(corpus.values()), embedder, cache=self._cache(tmp_path))
        embedder.embedded.clear()
        other = self._cache(tmp_path, model_id="fake/other")
        DenseIndex.build(list(corpus.values()), embedder, cache=other)
        assert len(embedder.embedded) == 3

    def test_a_different_query_prefix_does_not_reuse_the_old_space(self, corpus, tmp_path) -> None:  # type: ignore[no-untyped-def]
        embedder = CountingEmbedder(
            {"north": (0.0, 1.0), "east": (1.0, 0.0), "north-east": (1.0, 1.0)}
        )
        DenseIndex.build(list(corpus.values()), embedder, cache=self._cache(tmp_path))
        embedder.embedded.clear()
        DenseIndex.build(
            list(corpus.values()), embedder, cache=self._cache(tmp_path, query_prefix="")
        )
        assert len(embedder.embedded) == 3

    def test_only_the_changed_chunk_is_re_embedded(self, corpus, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The property that makes day 6 affordable: partial re-embed, not all-or-nothing."""
        embedder = CountingEmbedder(
            {
                "north": (0.0, 1.0),
                "east": (1.0, 0.0),
                "north-east": (1.0, 1.0),
                "south": (0.0, -1.0),
            }
        )
        DenseIndex.build(list(corpus.values()), embedder, cache=self._cache(tmp_path))
        embedder.embedded.clear()
        changed = dict(corpus) | {"b": chunk("b", "south")}
        DenseIndex.build(list(changed.values()), embedder, cache=self._cache(tmp_path))
        assert embedder.embedded == ["south"]
