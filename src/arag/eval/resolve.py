"""Resolve labelled spans against the real corpus.

## The gap this closes

Schema validation confirms a span is *well-formed*. It cannot confirm the span is *true*.
``{"document_id": "star-comprehensive-2025", "page": 34, "clause_id": "15"}`` is perfectly
well-formed and, if clause 15 is actually on page 14, it is also wrong — and wrong in the
worst available way: the item loads, appears in the golden set, is counted in the
denominator, and scores zero forever because ``SpanMatcher`` can never match it. No error
is raised anywhere. The stratum simply looks harder than it is.

## Three outcomes, not two

The obvious check — "is this clause on this page?" — is **insufficient for this corpus**,
and that was found by testing rather than assumed. Measured: clause ``15`` in
``star-comprehensive-2025`` resolves to pages ``[14, 34, 36, 42]``, because any line
beginning ``15.`` is indistinguishable from an ordinary numbered list item, and the
qualified form ``II.15`` is not even extractable (the document writes ``Section II`` and
``15.`` on separate lines). So a wrong page passes the clause check.

That means a label can be in one of three states, and collapsing them loses the one that
matters:

* **ERROR** — demonstrably false. The document does not exist, the page does not exist,
  the clause is absent, or the concept's anchors are present in the document but *not* on
  the labelled page. Blocking.
* **UNVERIFIABLE** — not shown false, but *nothing can check it*. An ambiguous clause id
  with no working concept cross-check. Also blocking, and reported in its own section,
  because it is indistinguishable from a wrong label and must not be skimmed past in a
  long run.
* **WARN** — checkable, checked, fine, but worth knowing (an ambiguous id that the concept
  check *did* verify; a span with no clause id at all).

Ambiguity alone is deliberately **not** blocking. It fires on plenty of correct labels —
including the verified bariatric-surgery span — so erroring on it would block good work and
teach the labeller to bypass the tool. What blocks is ambiguity *plus* the absence of any
independent check, which is the actual hole.

## Why it reads a JSON index rather than the PDFs

``arag.eval`` must not depend on the corpus being present. The PDFs are copyrighted and
never committed, so CI has none — validation that required them would run only on the
labeller's machine, which is precisely where a mistake is least likely to be caught and
most likely to be trusted.

``data/manifest/clause_index.json`` is produced by ``arag-ingest clause-index``, holds
identifiers and page numbers but no document text, and commits cleanly. This module parses
that JSON into its own small view rather than importing ``arag.ingest.clause_index``, which
would pull PyMuPDF into the eval package and break the import boundary that makes "harness
before pipeline" true.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from arag.eval.concepts import Concept, anchors_for
from arag.eval.schema import GoldenItem, GoldenSet
from arag.retrieval.types import DocSpan

# A clause id appearing on more than this many pages does not identify a clause.
#
# Measured: in star-comprehensive-2025 the id "15" appears on pages [14, 34, 36, 42].
# The IRDAI exclusion codes (excl.02) are the corpus's only unambiguous clause identifier,
# which is why they are the preferred key for supersession items.
AMBIGUOUS_CLAUSE_PAGES = 3

# The two clause-id forms imposed by the regulator rather than chosen by an insurer: IRDAI
# standardised exclusion codes and defined terms. Both are portable across document
# versions, which is why they are the preferred join keys - and why their absence from the
# index is ambiguous between a bad label and an incomplete index rather than proof of a bad
# label. See the branch in _resolve_span that uses this.
PORTABLE_CLAUSE_FORM = re.compile(r"^(?:excl\.\d{2}|def\.[a-z][a-z0-9_]*)$")


class Severity(StrEnum):
    ERROR = "error"
    UNVERIFIABLE = "unverifiable"
    WARN = "warn"

    @property
    def blocking(self) -> bool:
        """UNVERIFIABLE blocks as hard as ERROR.

        A label nothing can check is not safer than a label shown to be wrong — it is the
        same risk with less information. Treating it as a warning would leave the only
        guard against a silently-wrong golden item as a line in a long log.
        """
        return self in (Severity.ERROR, Severity.UNVERIFIABLE)


@dataclass(frozen=True)
class Finding:
    item_id: str
    severity: Severity
    message: str

    def __str__(self) -> str:
        return f"{self.item_id}: {self.message}"


@dataclass
class CorpusIndex:
    """Read-only view over ``clause_index.json``."""

    generated_at: str = ""
    documents: dict[str, dict[str, Any]] = field(default_factory=dict)
    source: Path | None = None

    @classmethod
    def load(cls, path: Path) -> CorpusIndex:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            generated_at=str(raw.get("generated_at", "")),
            documents=dict(raw.get("documents", {})),
            source=path,
        )

    @property
    def document_ids(self) -> set[str]:
        return set(self.documents)

    def pages(self, document_id: str) -> int:
        return int(self.documents.get(document_id, {}).get("pages", 0))

    def clause_pages(self, document_id: str, clause_id: str) -> list[int]:
        clauses = self.documents.get(document_id, {}).get("clauses", {})
        return [int(p) for p in clauses.get(clause_id, [])]

    def has_clause(self, document_id: str, clause_id: str) -> bool:
        return clause_id in self.documents.get(document_id, {}).get("clauses", {})

    def concept_pages(self, document_id: str, concept: Concept) -> list[int]:
        entry = self.documents.get(document_id, {}).get("concepts", {}).get(concept.value)
        return [int(p) for p in entry["pages"]] if entry else []


def _fmt_anchors(concept: Concept) -> str:
    """Render a concept's anchor phrases for an error message.

    Included in every concept rejection because of two real bugs found while writing the
    worked examples: WAIT_INITIAL listed "30 days waiting" while the document says
    "30-day waiting period", and vaccination had no concept at all. Both produced a
    rejection whose cause was invisible without opening the module - which is exactly the
    wrong place to send someone mid-labelling.
    """
    phrases = anchors_for(concept)
    shown = ", ".join(repr(p) for p in phrases[:6])
    more = f" (+{len(phrases) - 6} more)" if len(phrases) > 6 else ""
    return f"[{shown}]{more}"


def _resolve_span(
    span: DocSpan,
    concept: Concept | None,
    index: CorpusIndex,
) -> list[tuple[Severity, str]]:
    """Everything checkable about one span, as (severity, message) pairs."""
    out: list[tuple[Severity, str]] = []
    doc = span.document_id

    if doc not in index.document_ids:
        return [
            (
                Severity.ERROR,
                f"ground_truth_span names unknown document {doc!r}. "
                f"Known: {sorted(index.document_ids)}",
            )
        ]

    total = index.pages(doc)
    if span.page is not None and span.page > total:
        out.append(
            (Severity.ERROR, f"page {span.page} does not exist in {doc} (it has {total} pages)")
        )

    # --- concept cross-check, computed first because the clause verdict depends on it ---
    concept_pages: list[int] = []
    concept_verified: bool | None = None  # None = no check possible
    if concept is not None and span.page is not None:
        concept_pages = index.concept_pages(doc, concept)
        if concept_pages:
            concept_verified = span.page in concept_pages
            if not concept_verified:
                out.append(
                    (
                        Severity.ERROR,
                        f"concept {concept.value!r} does not appear on page {span.page} of "
                        f"{doc} - its anchors matched pages {concept_pages[:8]}. The span "
                        "is almost certainly on the wrong page. "
                        f"Anchors tried: {_fmt_anchors(concept)}. If the clause genuinely "
                        "discusses this concept in different words, extend "
                        "arag.eval.concepts.CONCEPT_ANCHORS rather than moving the span.",
                    )
                )
        else:
            out.append(
                (
                    Severity.WARN,
                    f"concept {concept.value!r} has no anchor match anywhere in {doc}. "
                    f"Anchors tried: {_fmt_anchors(concept)}. Either the concept is wrong "
                    "for this document, or its phrases need extending in "
                    "arag.eval.concepts.CONCEPT_ANCHORS.",
                )
            )

    if span.clause_id is None:
        out.append(
            (
                Severity.WARN,
                f"span in {doc} has no clause_id, so it can only be matched by page. Page "
                "numbers are not comparable across document versions (S5: the two Star "
                "wordings differ 2.7x in pagination), so prefer a clause.",
            )
        )
        return out

    if not index.has_clause(doc, span.clause_id):
        # A clause id in one of the two PORTABLE forms is a different situation from a
        # made-up one, and must not be reported in the same words.
        #
        # `excl.NN` and `def.<term>` are regulator-imposed: the labeller did not invent
        # them, they either exist in the document or they do not. So when one is absent
        # from the index there are two candidate explanations - a wrong label, or an
        # incomplete index - and the harness cannot tell which. Saying "not found
        # anywhere, use locate to find the real clause" asserts the first, and that
        # message cost real work: instance 8 in docs/SILENT_WRONGNESS.md was a reader bug
        # that dropped excl.06/07/08/16 from star-comprehensive-2025, and the only symptom
        # was this line telling a labeller their correct span was wrong.
        #
        # UNVERIFIABLE, not ERROR, and deliberately: it blocks just as hard (Severity.
        # blocks) but says the truthful thing, which is that nothing here can be checked.
        if PORTABLE_CLAUSE_FORM.match(span.clause_id):
            kind = "code" if span.clause_id.startswith("excl.") else "defined term"
            out.append(
                (
                    Severity.UNVERIFIABLE,
                    f"{kind} {span.clause_id!r} is not in clause_index.json for {doc} - "
                    "the index may be incomplete; verify against the PDF before moving "
                    "your span. This form is regulator-imposed, not insurer numbering, so "
                    "its absence is as likely to be a gap in the index as an error in the "
                    "label. Check with `arag-ingest locate`, and if the text is there, "
                    "rebuild with `arag-ingest clause-index` and re-run this.",
                )
            )
            return out
        out.append(
            (
                Severity.ERROR,
                f"clause {span.clause_id!r} was not found anywhere in {doc}. This item "
                "would load, count in the denominator, and score zero forever. Use "
                "`arag-ingest locate` to find the real clause.",
            )
        )
        return out

    found_on = index.clause_pages(doc, span.clause_id)
    if span.page is not None and span.page not in found_on:
        out.append(
            (
                Severity.ERROR,
                f"clause {span.clause_id!r} exists in {doc} but NOT on page {span.page} "
                f"(found on pages {found_on}). SpanMatcher would never match this span.",
            )
        )
        return out

    if len(found_on) > AMBIGUOUS_CLAUSE_PAGES and concept_verified is not False:
        # concept_verified is False means the span already has a hard ERROR for being on
        # the wrong page. Adding "and also unverifiable" tells the labeller nothing new and
        # dilutes the UNVERIFIABLE section, whose whole value is that it stays short.
        if concept_verified is True:
            out.append(
                (
                    Severity.WARN,
                    f"clause {span.clause_id!r} is ambiguous in {doc} - it appears on "
                    f"{len(found_on)} pages {found_on[:6]}, most likely matching ordinary "
                    f"numbered list items. Page {span.page} was confirmed independently by "
                    f"the {concept.value!r} anchors, so the span stands.",  # type: ignore[union-attr]
                )
            )
        else:
            reason = (
                "the item has no concept_id"
                if concept is None
                else f"concept {concept.value!r} has no anchors in this document"
                if not concept_pages
                else "the span has no page to check"
            )
            out.append(
                (
                    Severity.UNVERIFIABLE,
                    f"clause {span.clause_id!r} appears on {len(found_on)} pages of {doc} "
                    f"{found_on[:6]}, so the clause id alone cannot locate it - and {reason}, "
                    "so nothing can confirm the page either. This label cannot be checked "
                    "by anything, which is indistinguishable from it being wrong. Fix it by "
                    "using an unambiguous clause id (an IRDAI 'excl.NN' code where the "
                    "clause is a standardised exclusion) or by setting a concept_id whose "
                    "anchors appear on the page.",
                )
            )
    return out


def resolve_item(item: GoldenItem, index: CorpusIndex) -> list[Finding]:
    """Check one item's spans against the corpus."""
    out: list[Finding] = []

    for span in item.ground_truth_spans:
        for severity, message in _resolve_span(span, item.concept_id, index):
            out.append(Finding(item.id, severity, message))

    for pattern in item.must_not_cite:
        doc = pattern.split("#", 1)[0]
        if doc not in index.document_ids:
            out.append(
                Finding(
                    item.id,
                    Severity.ERROR,
                    f"must_not_cite names unknown document {doc!r}, so the supersession "
                    "trap can never fire and the item silently tests nothing",
                )
            )
    return out


@dataclass
class ResolveReport:
    findings: list[Finding] = field(default_factory=list)
    items_checked: int = 0
    index_generated_at: str = ""

    @property
    def hard_errors(self) -> list[Finding]:
        """Demonstrably false labels."""
        return [f for f in self.findings if f.severity is Severity.ERROR]

    @property
    def unverifiable(self) -> list[Finding]:
        """Labels nothing can check. Reported separately so they cannot be skimmed."""
        return [f for f in self.findings if f.severity is Severity.UNVERIFIABLE]

    @property
    def errors(self) -> list[Finding]:
        """Everything blocking, both kinds."""
        return [f for f in self.findings if f.severity.blocking]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARN]

    @property
    def unusable_item_ids(self) -> set[str]:
        return {f.item_id for f in self.errors}

    @property
    def ok(self) -> bool:
        return not self.errors


def resolve_golden_set(golden: GoldenSet, index: CorpusIndex) -> ResolveReport:
    report = ResolveReport(items_checked=len(golden.items), index_generated_at=index.generated_at)
    for item in golden.items:
        report.findings.extend(resolve_item(item, index))
    return report
