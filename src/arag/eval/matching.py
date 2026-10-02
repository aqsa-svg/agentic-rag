"""Span matching — the definition of "retrieval found the right passage".

Every retrieval metric in this project reduces to one boolean: does retrieved chunk *c*
cover ground-truth span *g*? That predicate is therefore the most load-bearing twenty
lines in the harness, and getting it subtly wrong would inflate or deflate every number
downstream without any test failing. It lives alone in this module, with its policy stated
explicitly rather than buried in a metric function.

Policy (``MatchMode.CLAUSE_PREFIX``, the default):

* Documents must match exactly. Cross-document credit is never given.
* If both sides carry a ``clause_id``, they match when one is a **dot-boundary prefix** of
  the other. A chunk covering clause ``4.2`` does cover ground truth ``4.2.b``; a chunk
  covering ``4.21`` does not. Plain ``str.startswith`` would wrongly credit the second,
  which is why the boundary check exists.
* If either side lacks a ``clause_id`` (clause parsing failed, or the label was recorded
  by page), fall back to page equality.

``page_tolerance`` defaults to **0**. Widening it makes numbers look better by crediting a
chunk that landed on the neighbouring page, which is precisely the chunking defect the
metric should expose. Raise it only with a written reason.
"""

from __future__ import annotations

from enum import StrEnum

from arag.agent.types import Citation
from arag.retrieval.types import DocSpan, RetrievedChunk


class MatchMode(StrEnum):
    EXACT = "exact"
    CLAUSE_PREFIX = "clause_prefix"
    PAGE = "page"


def _is_clause_prefix(outer: str, inner: str) -> bool:
    """True if ``outer`` is ``inner`` or a dot-boundary ancestor of it."""
    if outer == inner:
        return True
    return inner.startswith(outer + ".")


def _norm_clause(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().strip(".").lower()
    return cleaned or None


class SpanMatcher:
    def __init__(self, mode: MatchMode = MatchMode.CLAUSE_PREFIX, page_tolerance: int = 0) -> None:
        if page_tolerance < 0:
            raise ValueError("page_tolerance must be >= 0")
        self.mode = mode
        self.page_tolerance = page_tolerance

    def matches(self, retrieved: DocSpan, truth: DocSpan) -> bool:
        if retrieved.document_id != truth.document_id:
            return False

        r_clause, t_clause = _norm_clause(retrieved.clause_id), _norm_clause(truth.clause_id)

        if self.mode is MatchMode.PAGE:
            return self._pages_match(retrieved.page, truth.page)

        if self.mode is MatchMode.EXACT:
            if r_clause is None or t_clause is None:
                return self._pages_match(retrieved.page, truth.page)
            return r_clause == t_clause

        # CLAUSE_PREFIX
        if r_clause is not None and t_clause is not None:
            return _is_clause_prefix(r_clause, t_clause) or _is_clause_prefix(t_clause, r_clause)
        return self._pages_match(retrieved.page, truth.page)

    def _pages_match(self, a: int | None, b: int | None) -> bool:
        if a is None or b is None:
            # Neither clause nor page available on one side: the label is unusable for
            # scoring. Refuse to guess — returning True here would manufacture recall.
            return False
        return abs(a - b) <= self.page_tolerance

    # --- convenience wrappers over collections ---

    def spans_of(self, chunk: RetrievedChunk) -> list[DocSpan]:
        """Every span this chunk can legitimately be cited as.

        A chunk answers to more than one identifier. The block opening ``3.`` on
        ``star-comprehensive-2025`` p32 also carries ``Code Excl 03``; a label keyed on
        either name is pointing at this chunk, and the matcher has to know that.

        Matching only ``chunk.span`` is silent-wrongness instance 9. Before the chunker
        carried ``meta.clause_ids``, a span labelled ``excl.03`` could not match the chunk
        containing excl.03's text — so a correct retrieval scored zero, and the failure was
        reported against the *retriever*. Three of seven golden items were unscoreable that
        way, and the conclusions drawn from them were about the wrong component.

        The primary span comes first and is unchanged; the alternates differ only in
        ``clause_id``, because an alternate name does not move a chunk to another page.
        """
        primary = chunk.span
        alternates = [
            primary.model_copy(update={"clause_id": cid})
            for cid in chunk.meta.clause_ids
            if cid != primary.clause_id
        ]
        return [primary, *alternates]

    def matches_chunk(self, chunk: RetrievedChunk, truth: DocSpan) -> bool:
        """Whether any of the chunk's identities matches the labelled span."""
        return any(self.matches(s, truth) for s in self.spans_of(chunk))

    def relevance_vector(
        self, chunks: list[RetrievedChunk], truths: tuple[DocSpan, ...]
    ) -> list[int]:
        """Binary relevance, one entry per retrieved chunk, in rank order."""
        return [1 if any(self.matches_chunk(c, t) for t in truths) else 0 for c in chunks]

    def hit_truths(self, chunks: list[RetrievedChunk], truths: tuple[DocSpan, ...]) -> set[int]:
        """Indices of ground-truth spans covered by at least one retrieved chunk.

        Recall counts *distinct truths found*, not relevant chunks returned. Two chunks
        covering the same clause is one hit, not two — otherwise a redundant retriever
        would score higher than a precise one.
        """
        return {ti for ti, t in enumerate(truths) if any(self.matches_chunk(c, t) for c in chunks)}


def violates_must_not_cite(
    citations: tuple[Citation, ...], forbidden: tuple[str, ...]
) -> list[str]:
    """Return the forbidden patterns that were actually cited.

    A pattern is ``doc_id`` (any clause in that document is forbidden) or
    ``doc_id#clause_id`` (that clause and its descendants). Used for the supersession
    trap: citing the 2021 version of a clause replaced in 2024 is a hard failure
    regardless of how good the prose is.
    """
    hits: list[str] = []
    for pattern in forbidden:
        doc, _, clause = pattern.partition("#")
        want_clause = _norm_clause(clause) if clause else None
        for cite in citations:
            if cite.span.document_id != doc:
                continue
            if want_clause is None:
                hits.append(pattern)
                break
            got = _norm_clause(cite.span.clause_id)
            if got is not None and _is_clause_prefix(want_clause, got):
                hits.append(pattern)
                break
    return hits
