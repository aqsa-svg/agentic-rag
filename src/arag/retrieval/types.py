"""Retrieval value types.

``DocSpan`` is the project's unit of provenance and it appears in three places that must
agree exactly: the chunk metadata written at ingest, the citations an answer carries, and
the ``ground_truth_spans`` in the golden set. Defining it once, here, is what makes
"did retrieval find the passage a human labelled?" a computable question rather than a
string-matching guess.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class DocSpan(BaseModel):
    """A citable location in the corpus."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    # Pages are 1-indexed. ge=1 exists so the labelling template's `"page": 0`
    # placeholder is a load error rather than a span that matches nothing.
    page: int | None = Field(default=None, ge=1)
    clause_id: str | None = None

    def key(self) -> str:
        page = self.page if self.page is not None else ""
        return f"{self.document_id}#{self.clause_id or ''}@{page}"

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.key()


class ChunkMeta(BaseModel):
    """Metadata carried on every chunk. The temporal fields exist because supersession is
    a correctness requirement in this domain, not a nice-to-have: a clause that was true
    in 2021 and was replaced in 2024 must be filterable *before* it reaches the reranker.
    """

    model_config = ConfigDict(frozen=True)

    insurer: str | None = None
    product: str | None = None
    section_path: tuple[str, ...] = ()
    effective_from: date | None = None
    effective_to: date | None = None
    superseded_by: str | None = None
    is_table: bool = False
    # Set at ingest by the injection classifier; retrieval never trusts chunk text.
    injection_suspected: bool = False

    # EVERY citable identifier the chunk's text carries, not just the one that opened it.
    #
    # A clause has more than one name. `star-comprehensive-2025` p32 opens a block with the
    # list number "3." and the same block carries "Code Excl 03" - the IRDAI code, which is
    # the PORTABLE key (DESIGN, "the corpus has exactly two portable join keys") and the one
    # a labeller is told to prefer. Carrying only the heading id meant a label written on
    # `excl.03` could not match the chunk that contains excl.03's text, so a correct
    # retrieval scored zero and the metric reported a retrieval failure.
    #
    # Measured before the fix: 0 of 1,024 chunks carried an `excl.NN` id, while 3 of the 7
    # golden items keyed spans on one. Silent-wrongness instance 9.
    clause_ids: tuple[str, ...] = ()


class RetrievedChunk(BaseModel):
    """One candidate passage, with every score that contributed to its rank.

    Scores are kept separately rather than collapsed into one number so that the eval
    harness can attribute a failure to the stage that caused it: a chunk with a high
    ``lexical_score`` and a low ``rerank_score`` is a reranker decision, not a retrieval
    miss, and those two failures need different fixes.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    span: DocSpan
    text: str
    meta: ChunkMeta = Field(default_factory=ChunkMeta)

    dense_score: float | None = None
    lexical_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None

    @property
    def confidence(self) -> float:
        """The score the abstain gate reads. Rerank if present, else fusion, else 0."""
        for candidate in (self.rerank_score, self.fused_score, self.dense_score):
            if candidate is not None:
                return candidate
        return 0.0


class RetrievalFilters(BaseModel):
    """Pre-retrieval narrowing. ``as_of`` is the supersession control."""

    model_config = ConfigDict(frozen=True)

    insurer: str | None = None
    product: str | None = None
    document_ids: tuple[str, ...] = ()
    as_of: date | None = None
    tables_only: bool = False
