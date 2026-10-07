"""Clause index: what clause identifiers exist, on which page, in which document.

## Why this artefact exists

Two problems it solves, both of which were blocking:

**1. Labelled spans could not be checked against reality.** Schema validation confirms a
``clause_id`` is *well-formed* (``II.15`` yes, ``Room Rent`` no). It cannot confirm the
clause actually exists on the page claimed. A well-formed span pointing at the wrong page
passes validation and then silently fails at ``SpanMatcher`` time — the item is present in
the file, absent from every metric, and nothing errors. That is the single worst outcome
available to a hand-labelled dataset.

**2. The cross-version join needed evidence, not assertion.** Spike S5 established that
``clause_id`` cannot join two versions of the same product. ``concept_id`` was proposed as
the alternative. This index is what turns that proposal into a demonstrated mapping: for
each concept anchor, it records which pages of which documents contain it, so the mapping
from ``star-2021`` clause to ``star-2025`` clause can be *printed and inspected* rather
than trusted.

## Why a committed JSON artefact rather than reading PDFs at validation time

``arag.eval`` must not depend on the corpus being present. CI has no PDFs — they are
copyrighted and never committed — so validation that required them would be un-runnable in
CI and the golden set would be checked only on the labeller's machine. The index is small
(identifiers and page numbers, no document text), derived, and reproducible from the
manifest, so it commits cleanly and validation works everywhere.

## Extraction is per-document, by necessity

S5 measured that heading grammar differs *between documents of the same product*:
``star-comprehensive-2025`` is dominated by ``numbered_clause``, ``star-comprehensive-2021``
by ``all_caps`` with ``Section N`` identifiers. So the extractor applies the union of all
known patterns and records which pattern produced each identifier. Recording the pattern
matters: it is what lets a reviewer see *why* an identifier was extracted, instead of
inheriting a list of numbers with no provenance.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pymupdf

from arag.eval.concepts import Concept, anchors_for, slug_term, validate_clause_id
from arag.ingest.manifest import Manifest
from arag.ingest.normalise import normalise_text
from arag.obs import get_logger, span

log = get_logger(__name__)

# Identifier patterns, each named so extracted ids carry their provenance.
# Order matters only for reporting; every pattern is applied to every line.
ID_PATTERNS: dict[str, re.Pattern[str]] = {
    # "Section 6", "Section II" - star-comprehensive-2021's scheme
    "section": re.compile(r"^\s{0,8}Section\s+([IVXLCDM]+|\d{1,2})\b", re.IGNORECASE),
    # "15." / "4.2" / "1.2.14" - star-comprehensive-2025 and the IRDAI documents
    "numbered": re.compile(r"^\s{0,8}(\d{1,2}(?:\.\d{1,3}){0,3})[\.\)]\s"),
    # "II - Section 6 m." - the compound form seen in star-2021's summary table
    "compound": re.compile(
        r"\b([IVXLCDM]+)\s*[-\u2013]\s*Section\s+(\d{1,2})\s*([a-z])?\.", re.IGNORECASE
    ),
    # "Hospital: A Hospital means..." - a defined term in the Definitions section, which
    # carries no clause numbering of its own. Anchored to the line start because a
    # mid-sentence "Word:" is usually a list label, not a definition.
    "definition": re.compile(r"^\s{0,6}([A-Z][A-Za-z][A-Za-z /&'-]{1,38}):\s"),
    # "Code Excl 02" - the IRDAI standardised exclusion codes. Indexed because they are
    # the only clause identifier in this corpus that is BOTH unambiguous and stable across
    # document versions, which makes them the preferred key for supersession items.
    # Searched rather than line-anchored: the code trails the clause title on the same line.
    "excl_code": re.compile(r"\bCode\s+Excl\s*(\d{2})\b", re.IGNORECASE),
}

# A running page number ("12 / 47", "3 of 18") looks exactly like a bare clause number.
# Every page in this corpus has one, so without this exclusion the index would claim every
# document contains a clause named after each of its own page numbers.
PAGE_NUMBER = re.compile(r"^\s*\d{1,3}\s*(?:/|of)\s*\d{1,3}\s*$", re.IGNORECASE)


@dataclass
class DocumentClauseIndex:
    source_id: str
    pages: int = 0
    # canonical clause id -> sorted pages it appears on
    clauses: dict[str, list[int]] = field(default_factory=dict)
    # page -> sorted canonical clause ids found on it
    page_clauses: dict[int, list[str]] = field(default_factory=dict)
    # which pattern produced each id, for reviewability
    id_pattern: dict[str, str] = field(default_factory=dict)
    # concept value -> {"pages": [...], "clauses": [...]}
    concepts: dict[str, dict[str, list[Any]]] = field(default_factory=dict)

    def add(self, clause_id: str, page: int, pattern: str) -> None:
        self.clauses.setdefault(clause_id, [])
        if page not in self.clauses[clause_id]:
            self.clauses[clause_id].append(page)
        self.page_clauses.setdefault(page, [])
        if clause_id not in self.page_clauses[page]:
            self.page_clauses[page].append(clause_id)
        self.id_pattern.setdefault(clause_id, pattern)

    def has_clause(self, clause_id: str, page: int | None = None) -> bool:
        if clause_id not in self.clauses:
            return False
        if page is None:
            return True
        return page in self.clauses[clause_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pages": self.pages,
            "clauses": {k: sorted(v) for k, v in sorted(self.clauses.items())},
            "page_clauses": {str(k): sorted(v) for k, v in sorted(self.page_clauses.items())},
            "id_pattern": dict(sorted(self.id_pattern.items())),
            "concepts": dict(sorted(self.concepts.items())),
        }


@dataclass
class ClauseIndex:
    documents: dict[str, DocumentClauseIndex] = field(default_factory=dict)
    generated_at: str = ""

    def document(self, source_id: str) -> DocumentClauseIndex | None:
        return self.documents.get(source_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "note": (
                "Derived from the corpus by `arag-ingest clause-index`. Contains "
                "identifiers and page numbers only, never document text, so it commits "
                "without redistributing copyrighted material."
            ),
            "documents": {k: v.to_dict() for k, v in sorted(self.documents.items())},
        }

    @classmethod
    def load(cls, path: Path) -> ClauseIndex:
        raw = json.loads(path.read_text(encoding="utf-8"))
        index = cls(generated_at=raw.get("generated_at", ""))
        for source_id, body in raw.get("documents", {}).items():
            doc = DocumentClauseIndex(source_id=source_id, pages=int(body.get("pages", 0)))
            doc.clauses = {k: list(v) for k, v in body.get("clauses", {}).items()}
            doc.page_clauses = {int(k): list(v) for k, v in body.get("page_clauses", {}).items()}
            doc.id_pattern = dict(body.get("id_pattern", {}))
            doc.concepts = dict(body.get("concepts", {}))
            index.documents[source_id] = doc
        return index

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


# Which patterns anchor to the start of a line, and which are searched anywhere.
#
# Not a style choice - this split is the fix for a measured defect. The two groups must be
# applied to different units of text:
#
#   line-anchored -> one line at a time. "^Section 6" means nothing matched mid-line.
#   page-searched -> the WHOLE page. "Code Excl 08" trails a clause title on the same
#                    visual line, but PyMuPDF emits a newline wherever the PDF wraps, and
#                    in star-comprehensive-2025's two-column exclusions layout it wraps
#                    INSIDE the token: 'Code \nExcl 08', 'Code Excl \n06'.
#
# Applying the page-searched group per line dropped 5 (code, page) pairs from
# star-comprehensive-2025 - excl.06/07/08 on p33, excl.16 on p34, and p44 from excl.03's
# page list. star-comprehensive-2021 is single-column and lost nothing, so the gap looked
# like a real difference between the two wordings rather than a defect in the reader.
#
# Silent-wrongness instance 8. Its only symptom was `arag-eval validate` reporting
# UNVERIFIABLE for a correct span, which reads as "your label is wrong".
LINE_ANCHORED = ("section", "numbered", "definition")
PAGE_SEARCHED = ("compound", "excl_code")


def _canonical(name: str, match: re.Match[str]) -> str | None:
    """Canonical clause id for a pattern match, or None if it will not canonicalise.

    An identifier that will not canonicalise is not usable as a label, so it is not
    indexed - indexing it would let a labeller reference something SpanMatcher cannot
    match.
    """
    if name == "compound":
        roman, number, letter = match.group(1), match.group(2), match.group(3)
        raw = f"{roman.upper()}.{number}" + (f".{letter.lower()}" if letter else "")
    elif name == "excl_code":
        raw = f"excl.{match.group(1)}"
    elif name == "definition":
        raw = f"def.{slug_term(match.group(1))}"
    else:
        raw = match.group(1)
    try:
        return validate_clause_id(raw)
    except ValueError:
        return None


def _ids_on_line(line: str) -> list[tuple[str, str]]:
    """Canonical (clause_id, pattern_name) pairs for the LINE_ANCHORED patterns only.

    Callers must also run ``_ids_on_page`` over the page text. See LINE_ANCHORED above for
    why splitting the two is load-bearing rather than tidy.
    """
    if PAGE_NUMBER.match(line):
        return []
    out: list[tuple[str, str]] = []
    for name in LINE_ANCHORED:
        match = ID_PATTERNS[name].match(line)
        if match:
            clause_id = _canonical(name, match)
            if clause_id is not None:
                out.append((clause_id, name))
    return out


def ids_on_page(text: str) -> list[tuple[str, str]]:
    """Canonical (clause_id, pattern_name) pairs for the PAGE_SEARCHED patterns.

    Matched against the whole page rather than line by line, so a PDF line wrap inside the
    identifier cannot hide it: ``\\s`` spans the newline, and 'Code \\nExcl 08' matches
    exactly as 'Code Excl 08' does.
    """
    out: list[tuple[str, str]] = []
    for name in PAGE_SEARCHED:
        for match in ID_PATTERNS[name].finditer(text):
            clause_id = _canonical(name, match)
            if clause_id is not None:
                out.append((clause_id, name))
    return out


def build_clause_index(
    manifest: Manifest,
    raw_dir: Path,
    *,
    max_pages: int | None = None,
) -> ClauseIndex:
    index = ClauseIndex(generated_at=datetime.now(UTC).isoformat(timespec="seconds"))

    with span("ingest.clause_index"):
        # production_sources, not sources: the clause index is the cross-document join key
        # for real wordings, and an adversarial fixture's payload clauses must not pollute
        # it any more than they may reach retrieval.
        for source in manifest.production_sources:
            path = raw_dir / f"{source.id}.pdf"
            if not path.exists():
                log.warning("clause_index_skipped", source_id=source.id, reason="not fetched")
                continue

            doc_index = DocumentClauseIndex(source_id=source.id)
            wanted = {c: anchors_for(c) for c in Concept}

            pdf = pymupdf.open(path)
            with pdf:
                doc_index.pages = pdf.page_count
                limit = pdf.page_count if max_pages is None else min(pdf.page_count, max_pages)
                for i in range(limit):
                    raw = pdf[i].get_text("text") or ""
                    # Normalised so anchors match: "beneﬁt" would not match "benefit".
                    text, _ = normalise_text(
                        raw, context=f"{source.id} p{i + 1}", strict_numerics=False
                    )
                    page_no = i + 1

                    for line in text.splitlines():
                        for clause_id, pattern_name in _ids_on_line(line):
                            doc_index.add(clause_id, page_no, pattern_name)
                    # Whole-page pass, so an identifier broken across a PDF line wrap is
                    # still found. Must not be folded into the loop above.
                    for clause_id, pattern_name in ids_on_page(text):
                        doc_index.add(clause_id, page_no, pattern_name)

                    lowered = text.lower()
                    for concept, phrases in wanted.items():
                        if any(p in lowered for p in phrases):
                            entry = doc_index.concepts.setdefault(
                                concept.value, {"pages": [], "clauses": []}
                            )
                            if page_no not in entry["pages"]:
                                entry["pages"].append(page_no)

            # Attach the clause ids present on each concept's pages. This is the join
            # evidence: for a given concept it shows the identifier used by each document.
            for entry in doc_index.concepts.values():
                clauses: list[str] = []
                for page_no in entry["pages"]:
                    clauses.extend(doc_index.page_clauses.get(page_no, []))
                entry["clauses"] = sorted(set(clauses))

            index.documents[source.id] = doc_index
            log.info(
                "clause_index_built",
                source_id=source.id,
                pages=doc_index.pages,
                clause_ids=len(doc_index.clauses),
                concepts=len(doc_index.concepts),
            )
    return index


def concept_mapping(
    index: ClauseIndex, left: str, right: str
) -> list[tuple[str, list[int], list[str], list[int], list[str]]]:
    """The cross-version join, rendered for inspection.

    Returns one row per concept present in BOTH documents:
    ``(concept, left_pages, left_clauses, right_pages, right_clauses)``.

    This function is the answer to "show me the mapping mechanism". The join is on
    ``concept``; the clause columns are what differ, and seeing them side by side is what
    makes the impossibility of a clause-level join concrete rather than asserted.
    """
    a, b = index.document(left), index.document(right)
    if a is None or b is None:
        return []
    rows = []
    for concept in sorted(set(a.concepts) & set(b.concepts)):
        rows.append(
            (
                concept,
                list(a.concepts[concept]["pages"]),
                list(a.concepts[concept]["clauses"]),
                list(b.concepts[concept]["pages"]),
                list(b.concepts[concept]["clauses"]),
            )
        )
    return rows
