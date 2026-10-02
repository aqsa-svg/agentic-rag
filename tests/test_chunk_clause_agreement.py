"""The chunker and the clause index must agree on what a clause is called.

## The agreement the whole eval rests on, which had no test

`clause_index.json` tells a labeller which identifiers exist and where. The chunker decides
which identifiers a retrieved chunk carries. `SpanMatcher` compares the two. If they use
different schemes for the same clause, a **correct retrieval scores zero** and every metric
attributes the failure to the retriever.

That is not hypothetical. Measured on 2026-09-16, before the fix:

* `clause_index.json` held 35 `excl.NN` ids for `star-comprehensive-2025`.
* **0 of 1,024 indexed chunks carried a single one of them.** The chunker assigned the list
  number (`3`) where the index and the labels used the IRDAI code (`excl.03`).
* 3 of the 7 golden items keyed spans on `excl.NN`, so all three were **unscoreable** — and
  h-02's correct chunk was retrieved at rank #10 by the dense retriever while the item was
  reported as recall 0.000.

Silent-wrongness instance 9. The portable-key principle (DESIGN, "the corpus has exactly two
portable join keys") had been applied to the labels and to `clause_index.json` and never to
the chunker — the one component that has to agree with the labels for any number to mean
anything.

## Why the assertion is stated this way

"For every `(clause_id, page)` in `clause_index.json`, at least one chunk on that page
carries that `clause_id`." Directional, and deliberately so: the index is the labeller's
menu, so anything on the menu must be reachable. The reverse is not required — a chunk may
carry an id the index missed, which is a gap in the index, not in the chunker.

Marked `slow` + `needs_corpus`: it builds the full corpus. This is the gate to run after any
change to chunking or identifier extraction.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.needs_corpus]

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest" / "sources.jsonl"
INDEX = ROOT / "data" / "manifest" / "clause_index.json"
RAW = ROOT / "data" / "raw"

pytest.importorskip("pymupdf")


def _have_corpus() -> bool:
    return MANIFEST.exists() and INDEX.exists() and any(RAW.glob("*.pdf"))


@pytest.mark.skipif(not _have_corpus(), reason="corpus not fetched; run `arag-ingest fetch`")
class TestChunkerAgreesWithClauseIndex:
    @pytest.fixture(scope="class")
    def built(self):  # type: ignore[no-untyped-def]
        from arag.index.build import build_corpus

        return build_corpus(MANIFEST, RAW, serialisation="markdown")

    def test_every_indexed_clause_id_is_carried_by_a_chunk_on_its_page(self, built) -> None:  # type: ignore[no-untyped-def]
        """Scoped, and the scope is measured rather than assumed.

        The assertion as first written - "every (clause_id, page) in the index is carried by
        a chunk" - FAILS, on 624 of 1,238 pairs. Not from instance 9: measured breakdown
        after the fix,

            family          in index   unreachable   % missing
            excl.NN               83             0          0%     <- instance 9, fixed
            bare numeric         678           271         40%
            def.<term>           449           351         78%
            other                 28             2          7%

        the two large gaps are the clause index being deliberately MORE permissive than the
        chunker, and predate this work:

        * `def.<term>` - the index applies the definition pattern on every page. The chunker
          emits definitions only inside a detected definitions section, on purpose: without
          that gate, "Note:" and "Important:" open a `def.*` chunk on every page of every
          document (see DEFINITIONS_HEADING in chunk.py). The index's extra entries are
          mostly that noise.
        * `bare numeric` - the index records every numbered line. The chunker deliberately
          does NOT open a chunk per sub-clause: R2 forbids splitting a clause from its
          limbs, so `A.`/`B.`/`C.` live inside their parent's chunk and never become chunk
          ids of their own.

        So the counts are pinned as a regression baseline instead of asserted to zero. What
        is asserted to zero is the family that must never diverge - the portable keys the
        golden set is told to prefer - plus every span actually labelled. If the numeric or
        definition gap GROWS, this fails and someone looks; it cannot quietly widen.
        """
        index = json.loads(INDEX.read_text(encoding="utf-8"))["documents"]

        carried: dict[tuple[str, int], set[str]] = defaultdict(set)
        for chunk in built.chunks.values():
            for page in chunk.pages:
                carried[(chunk.span.document_id, page)].update(chunk.meta.clause_ids)

        def family(cid: str) -> str:
            if cid.startswith("excl."):
                return "excl.NN"
            if cid.startswith("def."):
                return "def.<term>"
            return "bare numeric" if re.fullmatch(r"[\d.]+", cid) else "other"

        missing: dict[str, list[str]] = defaultdict(list)
        total: dict[str, int] = defaultdict(int)
        for doc, body in index.items():
            for clause_id, pages in body.get("clauses", {}).items():
                for page in pages:
                    total[family(clause_id)] += 1
                    if clause_id not in carried.get((doc, int(page)), set()):
                        missing[family(clause_id)].append(f"{doc} p{page} {clause_id}")

        assert sum(total.values()) > 0, "no (clause_id, page) pairs to check"

        # The portable keys must be perfectly reachable. This is the instance 9 assertion.
        assert not missing["excl.NN"], (
            f"{len(missing['excl.NN'])} IRDAI exclusion codes in clause_index.json are "
            f"carried by no chunk on their page. A label using one can never match a "
            f"retrieved chunk and scores zero however good retrieval is - silent-wrongness "
            f"instance 9, returning. {missing['excl.NN'][:10]}"
        )

        # Measured 2026-09-16, immediately after the instance 9 fix. A ratchet, not a target.
        CEILINGS = {"bare numeric": 271, "def.<term>": 351, "other": 2}
        worse = {
            fam: (len(missing[fam]), ceiling)
            for fam, ceiling in CEILINGS.items()
            if len(missing[fam]) > ceiling
        }
        assert not worse, (
            f"the chunker/index divergence GREW: {worse} (family -> (now, baseline)). These "
            f"gaps are tolerated as documented design differences at their measured size, "
            f"not as a licence to widen. If the increase is deliberate, re-measure and move "
            f"the ceiling in the same commit as the change that caused it."
        )

    def test_the_exclusion_codes_specifically_are_carried(self, built) -> None:  # type: ignore[no-untyped-def]
        """Guard against the general test passing for the wrong reason.

        `excl.NN` is the corpus's only unambiguous, version-stable clause identifier and the
        one the golden set prefers. It is also the family that was entirely absent before
        instance 9. Asserted on its own so a regression here cannot hide inside an aggregate
        over 800 mostly-numeric ids.
        """
        by_doc: dict[str, set[str]] = defaultdict(set)
        for chunk in built.chunks.values():
            by_doc[chunk.span.document_id].update(
                cid for cid in chunk.meta.clause_ids if cid.startswith("excl.")
            )

        index = json.loads(INDEX.read_text(encoding="utf-8"))["documents"]
        for doc in ("star-comprehensive-2025", "star-comprehensive-2021"):
            expected = {c for c in index[doc]["clauses"] if c.startswith("excl.")}
            if not expected:
                continue
            got = by_doc[doc]
            assert got >= expected, (
                f"{doc}: chunks carry {len(got)} exclusion codes, the index has "
                f"{len(expected)}. Missing: {sorted(expected - got)[:10]}"
            )

    def test_golden_set_spans_are_all_reachable(self, built) -> None:  # type: ignore[no-untyped-def]
        """The end of the chain: every labelled span must be matchable by some chunk.

        The general test above checks the index; this checks the labels actually written.
        An unmatchable span makes its item score zero forever and reports the failure
        against the retriever - the exact shape of instance 9.
        """
        from arag.eval.matching import SpanMatcher
        from arag.eval.schema import GoldenSet
        from arag.retrieval.types import RetrievedChunk

        golden = GoldenSet.load(ROOT / "data" / "golden" / "v1.jsonl")
        matcher = SpanMatcher()
        as_retrieved = [
            RetrievedChunk(chunk_id=c.chunk_id, span=c.span, text=c.text, meta=c.meta)
            for c in built.chunks.values()
        ]

        unreachable: list[str] = []
        for item in golden.items:
            for span_ in item.ground_truth_spans:
                if not any(matcher.matches_chunk(c, span_) for c in as_retrieved):
                    unreachable.append(
                        f"{item.id}: {span_.document_id} p{span_.page} {span_.clause_id}"
                    )
        assert not unreachable, (
            "these labelled spans match NO chunk in the corpus, so their items score zero "
            "regardless of retrieval quality and the failure is reported against the "
            f"retriever: {unreachable}"
        )
