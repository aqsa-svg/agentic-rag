"""Every 'Code Excl NN' in the extracted text must reach clause_index.json.

## Why this test exists

`clause_index.json` is the only artefact standing between a labeller and a wrong span, and
it is trusted absolutely: `arag-eval validate` reports a clause it cannot find as a problem
with the *label*. So an incomplete index does not degrade gracefully — it actively
misdirects, telling someone their correct span is wrong.

That happened. The index builder matched its identifier patterns line by line, but PyMuPDF
emits a newline wherever the PDF wraps, and star-comprehensive-2025's two-column exclusions
layout wraps *inside the token*: `'Code \\nExcl 08'`, `'Code Excl \\n06'`. Five (code, page)
pairs were dropped — excl.06/07/08 on p33, excl.16 on p34, and p44 from excl.03's page list
— plus eleven compound ids in star-comprehensive-2021. star-2021's exclusions are
single-column and lost nothing, so the gap read as a genuine difference between the two
wordings. Instance 8 in `docs/SILENT_WRONGNESS.md`.

## Why it asserts against the PDF and not against a stored number

A test pinning "star-2025 has 35 exclusion codes" would have passed just as happily with
the bug present: the wrong number would simply have been the expected one. The assertion
has to be a *property* — everything in the text is in the index — measured against the
source document each time. That is why this test needs the corpus and is skipped without
it, rather than being made to run in CI on a fixture that cannot catch the failure.

Marked `slow` and `needs_corpus`: it opens every PDF. The default suite excludes it; it is
the gate to run after any change to the extraction or indexing path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.needs_corpus]

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest" / "sources.jsonl"
INDEX = ROOT / "data" / "manifest" / "clause_index.json"
RAW = ROOT / "data" / "raw"

# Deliberately NOT imported from arag.ingest.clause_index. A test that reuses the code
# under test's own pattern cannot detect that the pattern is applied to the wrong unit of
# text, which is exactly the defect this file exists to catch. `\s` spans the newline, so
# this matches the wrapped forms the builder's per-line pass could not see.
CODE_EXCL = re.compile(r"\bCode\s+Excl\s*(\d{2})\b", re.IGNORECASE)


def _corpus_pdfs() -> list[Path]:
    if not MANIFEST.exists() or not INDEX.exists():
        return []
    ids = [
        json.loads(line)["id"]
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return [RAW / f"{i}.pdf" for i in ids if (RAW / f"{i}.pdf").exists()]


PDFS = _corpus_pdfs()


@pytest.mark.skipif(not PDFS, reason="corpus PDFs not fetched; run `arag-ingest fetch`")
class TestExclusionCodeCompleteness:
    @pytest.mark.parametrize("pdf", PDFS, ids=lambda p: p.stem)
    def test_every_code_excl_in_the_text_is_in_the_index(self, pdf: Path) -> None:
        import pymupdf

        from arag.ingest.normalise import normalise_text

        index = json.loads(INDEX.read_text(encoding="utf-8"))["documents"]
        clauses: dict[str, list[int]] = index.get(pdf.stem, {}).get("clauses", {})

        in_text: set[tuple[str, int]] = set()
        doc = pymupdf.open(pdf)
        try:
            for i in range(doc.page_count):
                text, _ = normalise_text(doc[i].get_text("text") or "")
                for match in CODE_EXCL.finditer(text):
                    in_text.add((f"excl.{match.group(1)}", i + 1))
        finally:
            doc.close()

        if not in_text:
            pytest.skip(f"{pdf.stem} contains no IRDAI exclusion codes")

        missing = sorted(
            (code, page) for code, page in in_text if page not in clauses.get(code, [])
        )
        assert not missing, (
            f"{pdf.stem}: {len(missing)} (code, page) pair(s) are in the document text but "
            f"not in clause_index.json: {missing}. A labeller writing one of these spans "
            f"gets told their label is wrong when the index is what is incomplete. Rebuild "
            f"with `arag-ingest clause-index`; if that does not fix it, the identifier "
            f"pattern is being applied to the wrong unit of text (see LINE_ANCHORED vs "
            f"PAGE_SEARCHED in arag/ingest/clause_index.py)."
        )

    def test_the_probe_can_see_a_wrapped_code(self) -> None:
        """Guard against the guard: this test is worthless if its own regex is line-bound.

        The defect it exists to catch is precisely an identifier broken across a newline,
        so if `CODE_EXCL` stopped spanning one, every assertion above would pass while the
        index silently lost entries again.
        """
        for wrapped in ("Code \nExcl 08", "Code Excl \n06", "Code\nExcl\n16"):
            assert CODE_EXCL.search(wrapped), (
                f"CODE_EXCL no longer matches {wrapped!r}. It must span a newline, or this "
                "file cannot detect the failure it was written for."
            )
