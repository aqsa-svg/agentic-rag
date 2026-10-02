"""Ratchet on signals that are produced and never consumed.

## The class of defect this guards

Instances 1-9 in ``docs/SILENT_WRONGNESS.md`` are code that was wrong. This guards the
second pattern: code that is **right and unread**. A field populated correctly at the
producing site, covered by a test asserting it is populated correctly, and read by no
branch anywhere — so every test passes and the signal does nothing.

Measured cost, one session, three signals: ``LLMError.retryable`` (no retries happened at
all), ``LLMError.retry_after_s`` (the server's own delay parsed and discarded), and
``LLMResponse.truncated``. The third failed 5 of 7 questions in the first live generation
run, burned two requests per item against a 20/day quota by reprompting a truncation with
the same budget, and presented as "the model cannot produce JSON" — sending every
diagnosis at the prompt layer while the answer sat in the response object.

## Why the producer tests could not catch it

``tests/test_generate.py`` asserts every ``LLMErrorKind`` maps to its own
``AbstainReason`` and that ``retryable`` is correct per kind. Those check the PRODUCER, and
a signal nothing consumes has a perfect producer. Coverage was no help either: the line
setting ``truncated=True`` was covered, by a test asserting it gets set. The distinction is
between "is the value right?" and "does the value change what happens?" — only the second
is a behaviour.

## Why the allowed set is exact rather than a maximum

Both directions have to be deliberate:

* A NEW unread signal fails the build. That is the ratchet.
* A signal that BECOMES consumed also fails, until it is deleted from the list below.
  That is not pedantry — it forces the person who wired it up to state that they did, and
  it stops the list decaying into a stale inventory nobody trusts. The same reasoning as
  the measured ceilings in ``tests/test_chunk_clause_agreement.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_audit() -> Any:
    """Import tools/audit_signals.py by path.

    ``tools/`` is deliberately not a package: it holds development instruments, not
    product code, and putting it on the import path would let a tool be imported from
    ``src/arag`` by accident.
    """
    spec = importlib.util.spec_from_file_location(
        "arag_audit_signals", ROOT / "tools" / "audit_signals.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------
# The allowed set, audited 2026-09-21. Split by RISK, because someone reading the type
# cannot tell the two groups apart and they are not equally acceptable.
# ---------------------------------------------------------------------------------------

# Reads as an implemented defence. Is not one. Every entry here is a claim the type system
# makes that the code does not keep.
FALSE_ASSURANCE = {
    # The supersession abstention this project is largely ABOUT. Nothing produces it.
    "AbstainReason.ONLY_SUPERSEDED_EVIDENCE",
    "AbstainReason.CONTRADICTORY_UNRESOLVABLE",
    # The injection defence is currently a field name.
    "ChunkMeta.injection_suspected",
    # Only GENERATION is ever set, so `degraded` cannot distinguish a missing retrieval
    # leg from a missing generator - which is the entire reason it is a list.
    "DegradedComponent.DENSE_RETRIEVAL",
    "DegradedComponent.LEXICAL_RETRIEVAL",
    "DegradedComponent.RERANKER",
    "DegradedComponent.CHECKPOINTER",
}

# A slice not yet built, written down as such. A placeholder someone has recorded is a
# plan; an unread field nobody has noticed is a false assurance. That is the whole
# difference between this list and the one above.
PLANNED_PLACEHOLDER = {
    "RetrievedChunk.fused_score",  # set by the fuser; no consumer until the reranker lands
    "RetrievedChunk.rerank_score",  # v2 reranker, not built
    "AnswerResult.tool_calls",  # v3 agent, not built
    "AnswerResult.cold_start",  # reported once a cold path exists to report
    "LLMResponse.raw",  # deliberate: debugging payload, never branched on
}

ALLOWED_UNREAD = FALSE_ASSURANCE | PLANNED_PLACEHOLDER


class TestNoNewUnreadSignals:
    def test_the_unread_set_is_exactly_the_allowed_set(self) -> None:
        unread = _load_audit().unread_signals()

        new = sorted(unread - ALLOWED_UNREAD)
        assert not new, (
            f"{len(new)} signal(s) are produced and consumed by nothing: {new}. "
            "A field added to a failure taxonomy is not done until something BRANCHES on "
            "it and a test asserts the branch - see 'A SECOND pattern' in "
            "docs/SILENT_WRONGNESS.md. Either wire it up, or add it to "
            "PLANNED_PLACEHOLDER with the slice it is waiting for."
        )

        fixed = sorted(ALLOWED_UNREAD - unread)
        assert not fixed, (
            f"{len(fixed)} signal(s) now HAVE a production consumer: {fixed}. Good - "
            "delete them from ALLOWED_UNREAD in this file. The list is exact on purpose: "
            "an inventory that silently goes stale is one nobody trusts."
        )

    def test_the_two_risk_groups_do_not_overlap(self) -> None:
        """A signal is either a false assurance or a recorded plan, never filed as both."""
        assert not (FALSE_ASSURANCE & PLANNED_PLACEHOLDER)

    def test_the_audit_can_actually_see_a_consumed_signal(self) -> None:
        """Guard against the guard.

        If the AST walk stopped detecting reads, every signal would look unread, the
        first assertion would fail loudly - but if it stopped detecting *anything* and
        returned an empty set, the ratchet would pass while checking nothing. So assert a
        signal known to be heavily consumed is NOT in the unread set.
        """
        unread = _load_audit().unread_signals()
        for consumed in ("LLMError.kind", "LLMResponse.truncated", "AnswerResult.abstained"):
            assert consumed not in unread, (
                f"{consumed} is read in production code, so the audit is not detecting "
                "reads and this ratchet is vacuous"
            )


class TestSupersessionDefenceIsHonestlyRecorded:
    """The supersession defence has THREE independent holes. This pins the third.

    1. No version filter is wired - ``RetrievalFilters.as_of`` is honoured by both
       retrievers but no caller sets it.
    2. h-01 carries no ``must_not_cite``, so ``max_supersession_violations: 0`` is vacuous.
    3. Nothing produces ``AbstainReason.ONLY_SUPERSEDED_EVIDENCE``.

    Dense retrieval happens to rank the current 2025 definition above the superseded 2021
    one for h-01. That is incidental - an artefact of what the embedder preferred - and it
    sits on top of all three holes. The true claim is "we can see supersession failing and
    have not fixed it". Anything stronger is not supported.
    """

    def test_nothing_produces_the_supersession_abstention(self) -> None:
        unread = _load_audit().unread_signals()
        assert "AbstainReason.ONLY_SUPERSEDED_EVIDENCE" in unread, (
            "something now produces ONLY_SUPERSEDED_EVIDENCE. If the supersession defence "
            "was built, remove this test and the entry in FALSE_ASSURANCE, and update "
            "docs/BASELINE_V1R.md which currently records the hole."
        )

    def test_no_label_guards_against_citing_the_SUPERSEDED_wording(self) -> None:
        """Hole 2, stated precisely rather than approximately.

        An earlier version of this test asserted "no item carries must_not_cite" and was
        wrong: h-23 and h-24 carry seven entries each. Those are ABSTENTION DECOYS - the
        day-care and OPD clauses a retriever is expected to surface for an unanswerable
        question - and they guard a different failure entirely.

        What no label does is name a SUPERSEDED DOCUMENT. So
        ``max_supersession_violations: 0`` passes for h-01 not because the system avoided
        the 2021 wording, but because nothing asked it to: h-01 retrieves
        ``star-comprehensive-2021 p3 def.hospital`` at rank 1 under BM25 and the gate is
        silent. The metric runs; the measurement is empty.
        """
        from arag.eval.schema import GoldenSet

        superseded = {"star-comprehensive-2021"}
        golden = GoldenSet.load(ROOT / "data" / "golden" / "v1.jsonl")
        guarded = [
            item.id
            for item in golden.items
            if any(entry.split("#")[0] in superseded for entry in item.must_not_cite)
        ]
        if guarded:
            pytest.fail(
                f"items {guarded} now guard against citing a superseded wording, so the "
                "supersession metric is no longer vacuous. Delete this test and update "
                "docs/BASELINE_V1R.md, which currently records the hole."
            )
