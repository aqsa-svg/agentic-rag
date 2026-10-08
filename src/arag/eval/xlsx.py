"""Spreadsheet round-trip for the golden set.

Labelling happens in a spreadsheet because that is where reading six PDFs and typing
findings actually goes. JSONL is the storage format; this module is the bridge, and it is
deliberately lossy in one direction only — every field the schema validates survives, and
nothing is invented on the way back.

## The multi-span convention: ONE ROW PER ITEM, pipe-delimited spans

The alternative is one row per span with a repeated id. It was rejected:

* **A repeated-id sheet corrupts the moment someone sorts it.** Sort by strata to group
  your work and the continuation rows detach from their parents, silently, with no error
  until load. In a spreadsheet, sorting and filtering are not edge cases; they are how the
  tool is used.
* **It makes `question` and `reference_answer` ambiguous.** Repeat them and two copies can
  drift; blank them on continuation rows and a filtered view shows an item with no
  question.

So the row *is* the item. `document_id`, `page` and `clause_id` each hold a `|`-delimited
list, and the three must have the **same number of parts** — that constraint is the price
of the convention, and it is mechanically checkable, which is why it is the acceptable
price. A mismatch is a loud error naming the row, not a span quietly dropped.

``must_not_cite`` is `|`-delimited too, for consistency. Empty cells mean empty, never zero.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from arag.eval.schema import GoldenItem, GoldenSet
from arag.obs import get_logger

if TYPE_CHECKING:
    from pathlib import Path

log = get_logger(__name__)

COLUMNS = (
    "id",
    "strata",
    "expected_behaviour",
    "question",
    "reference_answer",
    "concept_id",
    "document_id",
    "page",
    "clause_id",
    "must_not_cite",
    "expected_to_fail",
    "expected_failure_reason",
    "injection_canary",
    "notes",
    "authored_by",
    "as_of_date",
)

SEP = "|"


class XlsxRowError(ValueError):
    """A row that cannot become a GoldenItem, named by its sheet row number."""


def _split(value: Any) -> list[str]:
    """Split a list cell, DISCARDING empty parts. For must_not_cite, where empties are noise."""
    if value is None:
        return []
    text = str(value).strip()
    return [p.strip() for p in text.split(SEP) if p.strip()] if text else []


def _split_positional(value: Any) -> list[str]:
    """Split a span column, PRESERVING empty parts, because position carries meaning.

    The three span columns are read by index, so an empty slot is information: h-02 has a
    clause on its first span and none on its second, which must survive as
    ``"excl.02|"`` -> ``["excl.02", ""]`` and not collapse to ``["excl.02"]``.

    Found by this module's own alignment check rejecting a sheet this module had just
    written - the writer emitted the trailing separator correctly and the reader threw the
    slot away. A round-trip test is the only thing that catches a reader and a writer
    disagreeing, because each is self-consistent.
    """
    if value is None:
        return []
    text = str(value).strip()
    return [p.strip() for p in text.split(SEP)] if text else []


def _truthy(value: Any) -> bool:
    """A spreadsheet boolean, which arrives in whatever form the labeller typed.

    openpyxl returns a real `bool` for a cell Excel stored as one, and a string for a cell
    someone typed "TRUE" or "yes" into. Anything unrecognised is FALSE rather than an
    error, deliberately: `expected_to_fail` only ever relaxes a gate, so a cell nobody
    filled in must not quietly mark an item as a known failure. The schema still refuses
    a true value with no `expected_failure_reason`.
    """
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower() if value is not None else ""
    return text in {"true", "yes", "y", "1"}


def _cell(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def row_to_item(row: dict[str, Any], *, row_number: int) -> GoldenItem | None:
    """One sheet row to a ``GoldenItem``. Returns None for a row that is not yet started.

    Validation is NOT duplicated here. The row is assembled into the same dict the JSONL
    loader builds and handed to ``GoldenItem``, so a malformed row fails through exactly
    the schema that would have rejected it in the file - same rules, same messages, one
    implementation. The only checks added are the ones the spreadsheet format introduces
    and the schema cannot see: span-column alignment.
    """
    item_id = _cell(row.get("id"))
    if not item_id:
        return None

    # NOT STARTED is not MALFORMED, and conflating them would make the importer unusable.
    #
    # The sheet ships with 33 pre-filled empty rows - id, strata, expected_behaviour and
    # any settled spans, with `question` and `reference_answer` deliberately blank because
    # those are the labeller's to write. Every one of those rows fails schema validation,
    # correctly: an `answer` item with no question and no reference answer is not a valid
    # item.
    #
    # Treating that as an error would mean nothing imports until all 40 are finished, which
    # is the opposite of what a labelling workflow needs - the whole point is importing
    # after every few rows and letting validation catch mistakes early.
    #
    # `question` is the marker of intent: a row with one has been STARTED, and anything
    # wrong with it from there is a real error the labeller wants to see.
    if not _cell(row.get("question")):
        return None

    docs = _split_positional(row.get("document_id"))
    pages = _split_positional(row.get("page"))
    clauses = _split_positional(row.get("clause_id"))

    # A clause_id is optional per span, so an EMPTY clause column is allowed - but a
    # partially filled one is not, because position is what binds the three columns
    # together and a short list silently re-pairs every span after the gap.
    if clauses and any(clauses) and len(clauses) != len(docs):
        raise XlsxRowError(
            f"row {row_number} ({item_id}): {len(docs)} document_id(s) but "
            f"{len(clauses)} clause_id(s). The three span columns are positional - "
            f"use '{SEP}{SEP}' to leave one span's clause blank rather than omitting it."
        )
    if docs and len(pages) != len(docs):
        raise XlsxRowError(
            f"row {row_number} ({item_id}): {len(docs)} document_id(s) but "
            f"{len(pages)} page(s). They are positional and must correspond one to one."
        )

    spans = []
    for i, doc in enumerate(docs):
        page_text = pages[i] if i < len(pages) else ""
        try:
            page = int(page_text) if page_text else None
        except ValueError as exc:
            raise XlsxRowError(
                f"row {row_number} ({item_id}): page {page_text!r} is not a number"
            ) from exc
        clause = clauses[i] if i < len(clauses) else None
        spans.append({"document_id": doc, "page": page, "clause_id": clause or None})

    payload: dict[str, Any] = {
        "id": item_id,
        "strata": _cell(row.get("strata")),
        "expected_behaviour": _cell(row.get("expected_behaviour")),
        "question": _cell(row.get("question")) or "",
        "authored_by": _cell(row.get("authored_by")) or "human",
    }
    for key, value in (
        ("reference_answer", _cell(row.get("reference_answer"))),
        ("concept_id", _cell(row.get("concept_id"))),
        ("notes", _cell(row.get("notes"))),
        ("as_of_date", _cell(row.get("as_of_date"))),
    ):
        if value is not None:
            payload[key] = value
    if spans:
        payload["ground_truth_spans"] = spans
    forbidden = _split(row.get("must_not_cite"))
    if forbidden:
        payload["must_not_cite"] = forbidden
    if _truthy(row.get("expected_to_fail")):
        payload["expected_to_fail"] = True
    reason = _cell(row.get("expected_failure_reason"))
    if reason:
        payload["expected_failure_reason"] = reason
    canary = _cell(row.get("injection_canary"))
    if canary:
        payload["injection_canary"] = canary

    try:
        return GoldenItem(**payload)
    except Exception as exc:
        raise XlsxRowError(f"row {row_number} ({item_id}): {exc}") from exc


def item_to_row(item: GoldenItem) -> dict[str, Any]:
    """A ``GoldenItem`` back to a sheet row. The inverse of ``row_to_item``."""
    return {
        "id": item.id,
        "strata": item.strata.value,
        "expected_behaviour": item.expected_behaviour.value,
        "question": item.question,
        "reference_answer": item.reference_answer or "",
        "concept_id": item.concept_id.value if item.concept_id else "",
        "document_id": SEP.join(s.document_id for s in item.ground_truth_spans),
        "page": SEP.join(str(s.page or "") for s in item.ground_truth_spans),
        "clause_id": SEP.join(s.clause_id or "" for s in item.ground_truth_spans),
        "must_not_cite": SEP.join(item.must_not_cite),
        "expected_to_fail": "TRUE" if item.expected_to_fail else "",
        "expected_failure_reason": item.expected_failure_reason or "",
        "injection_canary": item.injection_canary or "",
        "notes": item.notes or "",
        "authored_by": item.authored_by.value,
        "as_of_date": item.as_of_date.isoformat() if item.as_of_date else "",
    }


def read_xlsx(path: Path) -> tuple[list[GoldenItem], list[str]]:
    """Every STARTED row as a ``GoldenItem``, plus one message per rejected row.

    A row with an id and no question is not yet started and is skipped silently; it is the
    expected state of the 33 pre-filled rows the workbook ships with.

    Rows are collected rather than failing on the first error: a labeller who made the same
    mistake in eight rows should see eight messages, not one per run.
    """
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet = workbook.active
    rows = sheet.iter_rows(values_only=True)
    header = [str(c).strip() if c else "" for c in next(rows)]

    missing = [c for c in COLUMNS if c not in header]
    if missing:
        raise XlsxRowError(f"{path.name} is missing required column(s): {missing}")

    items: list[GoldenItem] = []
    problems: list[str] = []
    not_started = 0
    for number, raw in enumerate(rows, start=2):
        record = dict(zip(header, raw, strict=False))
        try:
            item = row_to_item(record, row_number=number)
        except XlsxRowError as exc:
            problems.append(str(exc))
            continue
        if item is not None:
            items.append(item)
        elif _cell(record.get("id")):
            not_started += 1
    workbook.close()
    log.info(
        "xlsx_read",
        path=str(path),
        items=len(items),
        not_started=not_started,
        problems=len(problems),
    )
    return items, problems


def write_xlsx(path: Path, items: list[GoldenItem], pending: list[dict[str, Any]]) -> Path:
    """Write the labelling workbook: completed items first, then pre-filled empty rows."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "golden"
    sheet.append(list(COLUMNS))

    header_font = Font(bold=True)
    done_fill = PatternFill("solid", fgColor="E8F5E9")
    todo_fill = PatternFill("solid", fgColor="FFF8E1")
    for cell in sheet[1]:
        cell.font = header_font
        cell.alignment = Alignment(vertical="top")

    for item in items:
        sheet.append([item_to_row(item).get(c, "") for c in COLUMNS])
    for row in pending:
        sheet.append([row.get(c, "") for c in COLUMNS])

    for index, row in enumerate(sheet.iter_rows(min_row=2), start=2):
        fill = done_fill if index - 1 <= len(items) else todo_fill
        for cell in row:
            cell.fill = fill
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    widths = {
        "id": 9,
        "strata": 22,
        "expected_behaviour": 18,
        "question": 52,
        "reference_answer": 52,
        "concept_id": 26,
        "document_id": 26,
        "page": 10,
        "clause_id": 18,
        "must_not_cite": 34,
        "expected_to_fail": 14,
        "expected_failure_reason": 44,
        "injection_canary": 28,
        "notes": 44,
        "authored_by": 14,
        "as_of_date": 12,
    }
    for index, column in enumerate(COLUMNS, start=1):
        sheet.column_dimensions[sheet.cell(row=1, column=index).column_letter].width = widths.get(
            column, 16
        )
    sheet.freeze_panes = "D2"

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)
    return path


def write_jsonl(path: Path, items: list[GoldenItem], header: str) -> Path:
    """Write items back to JSONL, preserving the file's comment header."""
    lines = [header.rstrip("\n")] if header else []
    lines += [
        # exclude_defaults as well as exclude_none: a no-op import must produce a no-op
        # DIFF. Without it, re-importing an unchanged sheet rewrites every row with
        # `expected_to_fail: false`, `must_not_cite: []`, `tags: []` - semantically
        # identical, and enough churn that a real change becomes hard to see in review.
        json.dumps(
            item.model_dump(mode="json", exclude_none=True, exclude_defaults=True),
            ensure_ascii=False,
        )
        for item in items
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    GoldenSet.load(path)  # round-trip proof: what we wrote must load as what we meant
    return path
