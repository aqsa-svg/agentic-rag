"""Node 5(b): the grounded-and-answers judge.

## Why this exists, and why it is independent of the generator

The self-report measurement (DESIGN §5.7, `data/selfreport.json`) asked whether the generator
can report its own abstention. It cannot: across 30 labelled items it answered three of the
five unanswerable questions at confidence 1.0, and no threshold on its self-reported confidence
beat the binary `can_answer` flag. A model that is maximally confident on exactly the questions
it should refuse cannot be its own verifier.

So this judge is a **separate** call that never sees the generator's confidence. It is given
only the question, the answer, and the passages the answer cited, and decides two things using
*only* those passages:

* **support** — for each cited passage, does it actually contain the specific claim the answer
  makes? Topicality is not support: a passage about the same subject that does not state the
  claim counts as unsupported. This is the distinction a bi-encoder cannot draw (it scores
  aboutness) and the reason a lexical overlap check was rejected (the unanswerable decoys share
  surface vocabulary with the question — DESIGN §5.7).
* **responsiveness** — does the answer actually address the question that was asked?

## The membership check (5a) runs first, elsewhere

This judge presupposes that every citation already resolved to a retrieved chunk — that is the
cheap mechanical membership check in `GeneratingEngine._verify`. The judge is invoked only on
answers that passed it, so it is never asked to reason about a fabricated citation; it reasons
about whether a *real, retrieved* passage supports the claim.

## It is a test instrument until it clears a measured bar

Whether this judge is allowed to GATE an abstention is decided by data, not by its existence.
It is scored on the same instrument as self-report — abstains caught out of the 5 unanswerable
items, answerables wrongly declined out of the 25 — and it gates only if it **strictly beats**
self-report's 2/5 caught and 2/25 wrongly declined on both numbers. Until then its verdict is
advisory and logged (see DESIGN §5.7, "node 5(b)"). Cohen's κ is not the gate: n=30 with 5
positives makes it unstable, and 0.60 is not a threshold this project derived.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from arag.agent.llm import LLMRequest
from arag.obs import get_logger

if TYPE_CHECKING:
    from arag.agent.llm import LLMProvider
    from arag.retrieval.types import RetrievedChunk

log = get_logger(__name__)

JUDGE_SYSTEM = """You verify answers about Indian health insurance policy wordings. You are \
given a QUESTION, an ANSWER, and the numbered PASSAGES the answer cited. Judge only from the \
passages; you have no other knowledge.

Decide two things independently:
1. "support": a list with one true/false per cited passage, in order. true ONLY if that \
passage actually STATES the specific claim the answer makes. A passage that is about the same \
topic but does not state the claim is false. Being about the subject is not support.
2. "responsive": true if the answer actually addresses the question that was asked, false if \
it answers a different question or does not answer it.

Reply with JSON only, no markdown fence:
{"support": [<true|false per passage>], "responsive": <true|false>, "reason": "<one sentence>"}
"""

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)
_OBJ = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class JudgeVerdict:
    """The judge's two sub-verdicts, plus whether it could be parsed.

    ``abstains`` is the decision rule, pre-registered: an answer is refused if it is not
    responsive, OR if no cited passage supports the claim. "No cited passage supports" rather
    than "not every passage supports" because a model that over-cites one irrelevant passage
    alongside a supporting one has still grounded its claim; the failure this catches is the
    answer that NONE of its passages support - the confident ungrounded answer. The stricter
    all()-support reading is reported separately as a diagnostic, not used to gate.
    """

    support: tuple[bool, ...]
    responsive: bool
    reason: str
    parsed: bool

    def abstains(self) -> bool:
        if not self.parsed:
            # A judge that cannot be parsed cannot clear an answer. Failing closed here would
            # turn every judge malformation into an abstention and let the judge's own
            # reliability masquerade as the system refusing well; failing OPEN (do not abstain
            # on a parse failure) keeps the judge's defects out of the abstention metric, and
            # the parse-failure rate is logged and reported separately.
            return False
        return (not self.responsive) or (not any(self.support))

    def abstains_strict(self) -> bool:
        """Diagnostic variant: refuse unless responsive AND every cited passage supports."""
        if not self.parsed:
            return False
        return (not self.responsive) or (not all(self.support))


class GroundedAnswersJudge:
    """One independent verification call per answer, at temperature 0."""

    def __init__(self, provider: LLMProvider, *, model: str | None = None) -> None:
        self._provider = provider
        self._model = model or provider.model

    def _build_user(self, question: str, answer: str, cited: list[RetrievedChunk]) -> str:
        blocks = []
        for i, chunk in enumerate(cited, start=1):
            where = f"{chunk.span.document_id}, page {chunk.span.page}"
            if chunk.span.clause_id:
                where += f", clause {chunk.span.clause_id}"
            blocks.append(f"[{i}] ({where})\n{chunk.text}")
        passages = "\n\n".join(blocks)
        return f"PASSAGES:\n\n{passages}\n\nQUESTION: {question}\n\nANSWER: {answer}"

    async def judge(self, question: str, answer: str, cited: list[RetrievedChunk]) -> JudgeVerdict:
        req = LLMRequest(
            model=self._model,
            system=JUDGE_SYSTEM,
            user=self._build_user(question, answer, cited),
            temperature=0.0,
            max_output_tokens=256,
        )
        resp = await self._provider.complete(req)
        return self._parse(resp.text, n_citations=len(cited))

    @staticmethod
    def _parse(text: str, *, n_citations: int) -> JudgeVerdict:
        raw = text.strip()
        m = _FENCE.match(raw)
        if m:
            raw = m.group(1).strip()
        obj = None
        try:
            obj = json.loads(raw)
        except Exception:
            mo = _OBJ.search(raw)
            if mo:
                try:
                    obj = json.loads(mo.group(0))
                except Exception:
                    obj = None
        if not isinstance(obj, dict):
            return JudgeVerdict(support=(), responsive=False, reason="", parsed=False)

        def _b(v: object) -> bool:
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.strip().casefold() in {"true", "yes", "1"}
            return bool(v)

        support_raw = obj.get("support")
        if isinstance(support_raw, list):
            support = tuple(_b(x) for x in support_raw)
        elif support_raw is not None:
            # a model sometimes collapses the list to a single bool; broadcast it
            support = tuple([_b(support_raw)] * max(n_citations, 1))
        else:
            support = ()
        responsive = _b(obj.get("responsive"))
        reason = str(obj.get("reason", ""))[:300]
        return JudgeVerdict(support=support, responsive=responsive, reason=reason, parsed=True)
