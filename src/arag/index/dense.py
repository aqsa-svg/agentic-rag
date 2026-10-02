"""Dense vector index and the offline embedder that fills it.

Offline only. This module imports ``sentence_transformers`` and therefore torch, so nothing
in ``arag.api``, ``arag.agent``, ``arag.retrieval`` or ``arag.index.lexical`` may reach it —
enforced over the full import closure by ``tests/test_import_closure.py``. The query-time
counterpart is an ONNX embedder that satisfies the same ``Embedder`` protocol; the retriever
is written against the protocol so swapping them changes no retrieval code.

## Exact search, not HNSW, and why that is not a shortcut

DESIGN §5.4 specifies pgvector with an HNSW index for production. This implementation does
an exact matrix multiply instead, deliberately:

* The corpus is ~10^3 chunks. A 1024x768 float32 matrix is 3MB and a full scan costs under
  a millisecond — HNSW would be optimising something that is not slow.
* HNSW is **approximate**. It introduces a recall loss that depends on ``ef_search``, and
  v2's whole purpose is to measure what dense retrieval contributes over the BM25 baseline.
  Mixing in an unmeasured approximation error would make that delta unattributable — the
  same reasoning that made v1 BM25-only.

So exact search here is the control. The HNSW recall loss becomes its own measurement later,
against these numbers, which is the only way to state it as a number rather than a hope.

## The failure this module is built to refuse

An index embedded with one model and queried with another produces a complete, plausible,
confidently-ordered ranking of nonsense. Every vector is the right shape, every score is a
valid float, nothing raises. Same for a query embedded without its model's instruction
prefix: the ranking is merely *worse*, silently.

Both are prevented structurally rather than by care: the index records ``model_id``,
``dim`` and ``query_prefix``, and ``DenseRetriever`` refuses to be constructed with an
embedder that disagrees.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from arag.obs import get_logger, span
from arag.retrieval.embedding import BGE_QUERY_PREFIX

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arag.ingest.chunk_types import Chunk

log = get_logger(__name__)

# Query instructions are per-model and not discoverable from the weights. Wrong prefix or
# no prefix costs recall without erroring, so the known ones are listed rather than
# guessed, and an unlisted model gets "" with a warning rather than a silent default.
QUERY_PREFIXES: dict[str, str] = {
    "BAAI/bge-base-en-v1.5": BGE_QUERY_PREFIX,
    "BAAI/bge-small-en-v1.5": BGE_QUERY_PREFIX,
    "BAAI/bge-large-en-v1.5": BGE_QUERY_PREFIX,
    "intfloat/multilingual-e5-small": "query: ",
    "sentence-transformers/all-MiniLM-L6-v2": "",
}

# e5 models are asymmetric on the passage side too, unlike bge.
PASSAGE_PREFIXES: dict[str, str] = {"intfloat/multilingual-e5-small": "passage: "}


def default_query_prefix(model_id: str) -> str:
    prefix = QUERY_PREFIXES.get(model_id)
    if prefix is None:
        log.warning(
            "unknown_query_prefix",
            model_id=model_id,
            detail="no instruction prefix will be applied; if this model expects one, "
            "retrieval quality drops with no error anywhere. Add it to QUERY_PREFIXES.",
        )
        return ""
    return prefix


class EmbeddingCache:
    """Content-addressed vector cache, so re-measuring does not re-embed the corpus.

    ## Why this is not an optimisation

    Measured on this machine (CPU, no CUDA): bge-base-en-v1.5 embeds a policy-wording chunk
    in ~658ms, so the 1,024-chunk corpus costs ~11 minutes, plus ~130s to load the model.
    Day 6 changes chunking and re-measures repeatedly. An 11-minute penalty on every run
    does not slow that work down so much as **discourage it** — and a chunking change whose
    effect nobody re-measured is precisely the kind of unverified claim this project exists
    to avoid. The cache exists to keep re-measurement cheap enough to stay habitual.

    ## The key, and why each part of it is load-bearing

    ``sha256(model_id | query_prefix | passage_prefix | chunk text)``.

    * **model_id** — vectors from two models are not comparable. Omit it and a model swap
      silently reuses the old space, which produces a confident ranking of nonsense.
    * **prefixes** — part of the model contract. The same text embedded with and without an
      instruction lands in different places.
    * **chunk text, not chunk_id** — the one that actually bites. Chunk ids are derived from
      document and position, so a chunking change can alter a chunk's *text* while keeping
      its id. Keying on the id would then serve a stale vector for text that no longer
      exists, with nothing to indicate it: the index would load, search, and return
      plausible results computed against the previous chunking. Content addressing makes a
      changed chunk a cache miss by construction, and re-embeds only what changed.

    An earlier version of the measurement script cached the whole index keyed on the set of
    chunk ids. That had exactly this hole.
    """

    def __init__(
        self, path: Path, *, model_id: str, query_prefix: str, passage_prefix: str
    ) -> None:
        self.path = path
        self._model_id = model_id
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._vectors: dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                keys = [str(k) for k in data["keys"]]
                matrix = np.asarray(data["vectors"], dtype=np.float32)
            self._vectors = {k: matrix[i] for i, k in enumerate(keys)}
            log.info("embedding_cache_loaded", path=str(path), n_vectors=len(self._vectors))

    def key(self, text: str) -> str:
        parts = (self._model_id, self._query_prefix, self._passage_prefix, text)
        return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()

    def get(self, text: str) -> np.ndarray | None:
        vector = self._vectors.get(self.key(text))
        if vector is None:
            self.misses += 1
        else:
            self.hits += 1
        return vector

    def put(self, text: str, vector: np.ndarray) -> None:
        self._vectors[self.key(text)] = np.asarray(vector, dtype=np.float32)

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self._vectors)
        np.savez_compressed(
            self.path,
            keys=np.array(keys),
            vectors=np.stack([self._vectors[k] for k in keys]) if keys else np.zeros((0, 0)),
        )
        log.info("embedding_cache_saved", path=str(self.path), n_vectors=len(keys))
        return self.path


class SentenceTransformerEmbedder:
    """Offline embedder. Satisfies ``arag.retrieval.embedding.Embedder``.

    Vectors are L2-normalised at source so a dot product is cosine similarity. Doing it
    here rather than in the index means there is exactly one place it can be forgotten.
    """

    def __init__(self, model_id: str = "BAAI/bge-base-en-v1.5", *, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self._model = SentenceTransformer(model_id, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension() or 0)
        self.query_prefix = default_query_prefix(model_id)
        self.passage_prefix = PASSAGE_PREFIXES.get(model_id, "")
        log.info(
            "embedder_loaded",
            model_id=model_id,
            dim=self.dim,
            device=device,
            query_prefix=repr(self.query_prefix),
        )

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            [self.passage_prefix + t for t in texts],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return [v.tolist() for v in np.asarray(vectors, dtype=np.float32)]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            self.query_prefix + text, normalize_embeddings=True, show_progress_bar=False
        )
        return list(np.asarray(vector, dtype=np.float32).tolist())


@dataclass
class DenseIndex:
    """Chunk ids and their unit-norm vectors, searched exactly."""

    model_id: str
    dim: int
    query_prefix: str
    chunk_ids: list[str]
    vectors: np.ndarray  # (n, dim) float32, L2-normalised

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2 or self.vectors.shape[0] != len(self.chunk_ids):
            raise ValueError(
                f"vectors shape {self.vectors.shape} does not match {len(self.chunk_ids)} chunk ids"
            )
        if self.vectors.shape[1] != self.dim:
            raise ValueError(f"vectors are {self.vectors.shape[1]}-d, index declares {self.dim}")
        if len(set(self.chunk_ids)) != len(self.chunk_ids):
            raise ValueError("duplicate chunk_id in the dense index")

    @property
    def n_chunks(self) -> int:
        return len(self.chunk_ids)

    @classmethod
    def build(
        cls, chunks: Sequence[Chunk], embedder: Any, *, cache: EmbeddingCache | None = None
    ) -> DenseIndex:
        """Embed every chunk, reusing cached vectors for text that has not changed.

        Only cache misses reach the model, so a chunking change that touches 40 chunks
        costs 40 embeds rather than 1,024. See ``EmbeddingCache`` for why the key is the
        chunk *text* and not its id.
        """
        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]

        with span("index.dense.build"):
            if cache is None:
                vectors = np.asarray(embedder.embed_passages(texts), dtype=np.float32)
            else:
                cached = [cache.get(t) for t in texts]
                todo = [i for i, v in enumerate(cached) if v is None]
                if todo:
                    fresh = np.asarray(
                        embedder.embed_passages([texts[i] for i in todo]), dtype=np.float32
                    )
                    for slot, i in enumerate(todo):
                        cache.put(texts[i], fresh[slot])
                        cached[i] = fresh[slot]
                    cache.save()
                log.info(
                    "dense_embed_cache",
                    hits=cache.hits,
                    misses=cache.misses,
                    embedded=len(todo),
                    reused=len(texts) - len(todo),
                )
                vectors = (
                    np.stack([np.asarray(v, dtype=np.float32) for v in cached])
                    if cached
                    else np.zeros((0, embedder.dim), dtype=np.float32)
                )

        norms = np.linalg.norm(vectors, axis=1)
        # Not a formality. An embedder that forgets to normalise still returns a valid
        # matrix, and every later cosine score becomes a dot product scaled by chunk
        # length - which ranks long chunks higher for reasons that have nothing to do with
        # the query, and reports no error at any point.
        if vectors.size and not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(
                f"embedder {embedder.model_id!r} returned un-normalised vectors "
                f"(norms {norms.min():.4f}..{norms.max():.4f}); cosine similarity would "
                "silently become a length-weighted dot product"
            )
        log.info("dense_index_built", n_chunks=len(ids), dim=int(vectors.shape[1]) if ids else 0)
        return cls(
            model_id=embedder.model_id,
            dim=embedder.dim,
            query_prefix=getattr(embedder, "query_prefix", ""),
            chunk_ids=ids,
            vectors=vectors,
        )

    def search(
        self, query_vector: Sequence[float], top_k: int, *, allow: frozenset[str] | None = None
    ) -> list[tuple[str, float]]:
        """Top-k (chunk_id, cosine) pairs, best first.

        ``allow`` narrows *before* scoring, matching the lexical retriever: post-filtering
        would return a short list for a document-scoped query because the global top-k came
        from excluded documents, which reads as "no such clause".
        """
        if not self.chunk_ids or top_k <= 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32)
        if query.shape != (self.dim,):
            raise ValueError(f"query vector is {query.shape}, index expects ({self.dim},)")

        if allow is None:
            rows = np.arange(len(self.chunk_ids))
        else:
            rows = np.array(
                [i for i, cid in enumerate(self.chunk_ids) if cid in allow], dtype=np.int64
            )
            if rows.size == 0:
                return []

        scores = self.vectors[rows] @ query
        k = min(top_k, rows.size)
        # argpartition then sort the k survivors: O(n) rather than O(n log n) on the full
        # corpus. Ties break on chunk_id so the ranking cannot move with ingest order -
        # the same rule the BM25 index uses, because a metric that shifts when unrelated
        # documents are re-ingested is not a measurement.
        top = np.argpartition(-scores, k - 1)[:k]
        ordered = sorted(top, key=lambda i: (-float(scores[i]), self.chunk_ids[rows[i]]))
        return [(self.chunk_ids[rows[i]], float(scores[i])) for i in ordered]

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vectors=self.vectors,
            meta=np.array(
                json.dumps(
                    {
                        "model_id": self.model_id,
                        "dim": self.dim,
                        "query_prefix": self.query_prefix,
                        "chunk_ids": self.chunk_ids,
                    }
                )
            ),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> DenseIndex:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            return cls(
                model_id=meta["model_id"],
                dim=meta["dim"],
                query_prefix=meta["query_prefix"],
                chunk_ids=list(meta["chunk_ids"]),
                vectors=np.asarray(data["vectors"], dtype=np.float32),
            )

    def size_bytes(self) -> int:
        """In-memory footprint of the vectors, for the cost table."""
        return int(self.vectors.nbytes)
