"""BM25 implemented in this repository, so that the claim "true BM25 with tunable k1/b"
survives being checked.

## Why the scoring is not delegated

Two shortcuts were available and both would have made the claim false:

* **Postgres full-text search.** ``ts_rank_cd`` is a cover-density score. It has no
  term-frequency saturation and no document-length normalisation, so there is no ``k1``
  and no ``b`` to tune - not badly named ones, absent ones. Calling it BM25 would be a
  false statement about the system, and the k1/b tuning experiment would have nothing to
  turn.
* **A BM25 library** (``rank_bm25``, Whoosh, Tantivy). Any of these scores correctly, but
  the formula then lives outside the repository and "why did chunk A out-rank chunk B?"
  becomes answerable only by reading someone else's source. The arithmetic below is about
  eighty lines and every one of them is hand-checkable, which is what
  ``tests/test_lexical.py`` does.

The variant used is the Robertson/Lucene one with ``+1`` inside the logarithm:

    idf(t) = ln( 1 + (N - n(t) + 0.5) / (n(t) + 0.5) )

That form is chosen because it is strictly positive for every ``n(t) <= N``. The classic
form without the ``+1`` goes negative once a term appears in more than half the corpus,
which on a six-document corpus happens for words as ordinary as "policy" - and the usual
remedy, clamping negatives to zero, is an undocumented ad-hoc step that changes the
ranking. Choosing the non-negative variant removes the need for the clamp instead of
hiding it.

## What this index deliberately does not do

**No stemming.** "exclusion" and "exclusions" are different terms here and will not match
each other. That is a real recall cost, accepted on purpose: a stemmer is a
retrieval-quality intervention, and every intervention in this project has to arrive with
a measured effect. Shipping v1 unstemmed makes "add a Porter stemmer" an A/B with a number
attached; shipping it stemmed would bake the decision into the baseline where its effect
can never be recovered.

**No positions, so no phrase or proximity queries.** A posting stores a term frequency,
not a list of offsets. Phrase evidence is what the cross-encoder reranker contributes, and
storing offsets would multiply the index size for a capability nothing downstream asks for
yet.

**No text.** The index maps terms to chunk ids and nothing else, so it cannot be the
source of a citation - the chunk store held by ``LexicalRetriever`` is. Keeping text out
means an index file can never disagree with the corpus about what a chunk *says*; it can
only disagree about which chunks exist, and that mismatch is checked when the retriever is
constructed.

## The invariant that makes index and query agree

Both sides tokenise through ``tokenise`` below, which calls ``normalise_query`` from
``arag.ingest.normalise``. Spike S5 measured 269 ligature codepoints in this corpus: if
the index folded U+FB01 and the query path did not, a search for "benefit" would score
zero against every chunk containing the ligature spelling, and the failure would look like
a relevance problem rather than a tokenisation bug. One function, called from both sides,
is the only structural defence against that drift.
"""

from __future__ import annotations

import math
import re
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import Any

from arag.ingest.chunk_types import Chunk
from arag.ingest.normalise import normalise_query
from arag.obs import get_logger

log = get_logger(__name__)

FORMAT_VERSION = 1

# Minimum token length. A length floor rather than a stop-word list: a stop-word list is a
# per-language asset that has to be maintained and defended, a floor is one number. The
# cost is explicit rather than hidden - a bare single digit is not indexable, so "Plan A"
# indexes as {plan} and "5,00,000" as {00, 000}. Numeric answers in this corpus come from
# table chunks and the reranker, never from BM25 term overlap alone.
MIN_TOKEN_LEN = 2

# Alphanumeric runs, Unicode-aware. ``[^\W_]`` is "word character except underscore":
# underscore is excluded so an identifier-shaped string splits instead of glueing two
# terms into one. Unicode-aware rather than ``[a-z0-9]`` so a non-Latin term becomes a
# token instead of vanishing silently, which is the failure class this project keeps
# finding in this corpus.
TOKEN_RE = re.compile(r"[^\W_]+")

# Bytes charged per posting by ``stats["bytes_estimate"]``: an 8-byte reference to the
# chunk id plus a 4-byte frequency. See ``stats`` for what that estimate is and is not.
BYTES_PER_POSTING = 12


def tokenise(text: str) -> list[str]:
    """The one tokeniser: chunk text at index time, query text at search time.

    Lower-cased, NFKC-folded via ``normalise_query``, split on non-alphanumerics, tokens
    shorter than ``MIN_TOKEN_LEN`` dropped, and **no stemming** - deliberately omitted at
    v1 so that adding one later is a measurable change rather than an invisible part of
    the baseline.
    """
    if not text:
        return []
    # normalise_query rather than normalise_text directly: it is the wrapper that passes
    # strict_numerics=False. Index-time text has already cleared the ingest numeric guard,
    # and re-running the strict check here would let a chunk ingest already accepted raise
    # during indexing - a failure at the wrong layer, with no remedy available to it.
    folded = normalise_query(text).lower()
    return [token for token in TOKEN_RE.findall(folded) if len(token) >= MIN_TOKEN_LEN]


@dataclass
class Bm25Index:
    """An inverted index with BM25 scoring.

    ``k1`` and ``b`` are the only constructor arguments because they are the two knobs the
    tuning experiment turns; everything else is derived from the chunks that were added.
    """

    k1: float = 1.5
    b: float = 0.75

    # term -> chunk_id -> term frequency. Nested dicts rather than the parallel packed
    # arrays a web-scale engine uses: this corpus is ~10^3 chunks, so that layout would
    # buy nothing measurable and cost the readability the module exists for.
    _postings: dict[str, dict[str, int]] = field(init=False, default_factory=dict, repr=False)
    _doc_len: dict[str, int] = field(init=False, default_factory=dict, repr=False)
    _idf: dict[str, float] = field(init=False, default_factory=dict, repr=False)
    _avgdl: float = field(init=False, default=0.0)
    _finalised: bool = field(init=False, default=False)

    # --- construction ------------------------------------------------------------------

    def add(self, chunk: Chunk) -> None:
        """Index one chunk. Raises on a repeated ``chunk_id``.

        Re-adding an id would double its term frequencies and inflate its score, which in
        a ranking is indistinguishable from a relevance win. An exception turns a
        double-ingest into a build failure instead of a quiet quality claim.
        """
        if chunk.chunk_id in self._doc_len:
            raise ValueError(
                f"chunk {chunk.chunk_id!r} is already indexed; re-adding it would double "
                "its term frequencies and inflate its BM25 score"
            )
        tokens = tokenise(chunk.text)
        self._doc_len[chunk.chunk_id] = len(tokens)
        for token in tokens:
            postings = self._postings.setdefault(token, {})
            postings[chunk.chunk_id] = postings.get(chunk.chunk_id, 0) + 1
        if not tokens:
            # Not an error: a chunk reading "5%" or a single glyph is legitimate content.
            # Logged because such a chunk is lexically unreachable for ever, which is a
            # retrieval hole that is otherwise invisible. It still counts towards N and
            # avgdl - it is a document in the corpus, and dropping it would misstate the
            # corpus statistics to make one number look tidier.
            log.warning(
                "chunk_has_no_lexical_tokens",
                chunk_id=chunk.chunk_id,
                chars=chunk.char_len,
            )
        # Any addition invalidates idf and avgdl. Marking the index stale rather than
        # recomputing here keeps `add` O(tokens) across a full-corpus build.
        self._finalised = False

    def finalise(self) -> None:
        """Compute the idf table and the average document length. Idempotent."""
        n_docs = len(self._doc_len)
        self._avgdl = sum(self._doc_len.values()) / n_docs if n_docs else 0.0
        self._idf = {
            term: self._idf_for_df(len(postings), n_docs)
            for term, postings in self._postings.items()
        }
        self._finalised = True
        log.info(
            "bm25_finalised",
            n_chunks=n_docs,
            n_terms=len(self._postings),
            avgdl=round(self._avgdl, 3),
            k1=self.k1,
            b=self.b,
        )

    # --- corpus statistics -------------------------------------------------------------

    @staticmethod
    def _idf_for_df(df: int, n_docs: int) -> float:
        return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def doc_frequency(self, term: str) -> int:
        """Number of chunks containing ``term``, which must already be tokenised."""
        return len(self._postings.get(term, ()))

    def idf(self, term: str) -> float:
        """Inverse document frequency of an already-tokenised term.

        Public because the eval harness has to be able to explain a rank, and "that term
        is in five of the six documents, so it contributed almost nothing" is the
        explanation. Derived from the live document frequency rather than read out of the
        finalised table so that it also answers for a term absent from the corpus - such a
        term has no postings, so it can never reach a score whatever its idf says.
        """
        return self._idf_for_df(self.doc_frequency(term), len(self._doc_len))

    @property
    def chunk_ids(self) -> frozenset[str]:
        return frozenset(self._doc_len)

    @property
    def stats(self) -> dict[str, Any]:
        """Index size, for the ingest report and for the deploy decision.

        ``bytes_estimate`` charges the term strings, the chunk id strings and
        ``BYTES_PER_POSTING`` per posting. It is a deliberately crude linear model, and two
        things it is *not* are worth stating so nobody quotes it as either:

        * It is not the in-memory footprint. Python's dict and str overhead is several
          times this figure, and reporting it as memory would put a confidently wrong
          number into a capacity plan.
        * It is not the authority on serialised size. It is a floor, cheap enough to log
          during a build; ``arag.index.build.index_size_bytes`` serialises the index and
          measures the real length, and that is the number a report should publish. Two
          numbers for one quantity is tolerable only while it is written down which one
          answers which question.
        """
        n_postings = sum(len(postings) for postings in self._postings.values())
        term_bytes = sum(len(term) for term in self._postings)
        id_bytes = sum(len(chunk_id) for chunk_id in self._doc_len)
        return {
            "n_chunks": len(self._doc_len),
            "n_terms": len(self._postings),
            "avgdl": self._avgdl,
            "postings": n_postings,
            "bytes_estimate": term_bytes + id_bytes + BYTES_PER_POSTING * n_postings,
        }

    # --- search ------------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 30,
        *,
        allow: AbstractSet[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Score ``query`` and return ``(chunk_id, score)`` pairs, best-first.

        ``allow`` restricts which chunks are *candidates*; it does not touch the idf table.
        That distinction is why filtering belongs here rather than in the caller.
        Restricting candidates keeps scores comparable across queries, because idf stays a
        property of the corpus; recomputing idf over the filtered subset would make the
        same chunk score differently depending on which filter the user happened to send.
        Filtering *after* the ``top_k`` cut would be worse still - a document-scoped query
        could come back empty while matching chunks existed, because the global top 30 all
        came from other documents.
        """
        if not self._finalised:
            raise RuntimeError(
                "Bm25Index.search() called before finalise(); idf and avgdl are not "
                "computed, so any score returned would be silently wrong"
            )
        # An index with no chunks, or one whose every chunk tokenised to nothing, has no
        # candidates at all. Returning early also makes the length normalisation below
        # division-safe without a second guard inside the hot loop.
        if top_k <= 0 or not self._doc_len or self._avgdl <= 0.0:
            return []

        # dict.fromkeys deduplicates while preserving order. Deduplication matters: the
        # scoring formula carries no query-term-frequency factor, so counting a repeated
        # term twice would make "benefit benefit" score double "benefit" for no defensible
        # reason. The alternative - adding the classic k3/qtf term - is extra tuning
        # surface nothing has asked for.
        terms = list(dict.fromkeys(tokenise(query)))
        if not terms:
            return []

        scores: dict[str, float] = {}
        for term in terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for chunk_id, freq in postings.items():
                if allow is not None and chunk_id not in allow:
                    continue
                length_norm = 1.0 - self.b + self.b * (self._doc_len[chunk_id] / self._avgdl)
                contribution = idf * (freq * (self.k1 + 1.0)) / (freq + self.k1 * length_norm)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + contribution

        # Tie-break on chunk_id so a ranking is reproducible run to run. Without it,
        # equal-scoring chunks order by dict insertion and an eval number moves when
        # ingest order changes - a metric that drifts for a non-reason is worse than no
        # metric at all.
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:top_k]

    # --- persistence -------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form. Stores integer counts only; ``from_dict`` recomputes the floats.

        Serialising idf and avgdl instead would round-trip them through decimal text and,
        worse, would let a stale idf table load beside counts that no longer justify it.
        Storing only the counts makes the round trip exact by construction rather than by
        luck of float formatting - which is why the round-trip test can assert score
        equality with ``==`` rather than ``approx``.
        """
        return {
            "version": FORMAT_VERSION,
            "k1": self.k1,
            "b": self.b,
            "doc_len": dict(self._doc_len),
            "postings": {term: dict(postings) for term, postings in self._postings.items()},
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Bm25Index:
        """Rebuild an index from ``to_dict`` output, finalised and ready to search."""
        version = raw.get("version")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"unsupported bm25 index format {version!r}; this build reads "
                f"{FORMAT_VERSION}. An index written by another version must be rebuilt, "
                "not guessed at: a changed tokeniser silently changes every score."
            )
        index = cls(k1=float(raw["k1"]), b=float(raw["b"]))
        index._doc_len = {str(key): int(value) for key, value in raw["doc_len"].items()}
        index._postings = {
            str(term): {str(chunk_id): int(freq) for chunk_id, freq in postings.items()}
            for term, postings in raw["postings"].items()
        }
        index.finalise()
        return index
