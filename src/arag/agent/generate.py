"""Retrieve → generate → **verify** → answer or abstain.

## The verify step is the reason this file is not thin

A RAG generator that returns whatever the model said is a demo. The failure this domain
punishes is not a crash, it is a fluent, confident, well-cited answer about a clause the
retriever never returned — and in insurance that answer has a rupee value attached.

So generation here is a three-stage pipeline and the middle stage is adversarial toward the
model's own output:

1. **Retrieve** — candidates, with their scores kept per-stage.
2. **Generate** — JSON, temperature 0, cite by the chunk numbers shown in the prompt.
3. **Verify** — every citation must resolve to a chunk that was actually retrieved. An
   answer whose citations do not survive verification is discarded and becomes an
   abstention, not a warning.

The verification is cheap and mechanical, and it is the only part that makes
``max_ungrounded_citation_rate`` in `thresholds.yaml` a real number rather than a field.

## Why the model cites indexes, not chunk ids

The prompt numbers the passages `[1]`, `[2]`, … and asks for those integers back.
Chunk ids look like `star-comprehensive-2025:p32:s146`, and a model asked to reproduce one
will occasionally produce a *plausible* variant — a different page, a different sequence
number — which is a hallucinated citation that passes a regex and points at a real, wrong
chunk. An integer in `1..len(candidates)` cannot be plausibly wrong: it is either in range
and refers to a passage the engine itself chose, or it is out of range and rejected.

## Abstention is a decision, not a fallback

Five distinct paths reach an abstention and each records a different ``AbstainReason``, so
v4's false-abstention work can decompose the number instead of staring at one rate:
no candidates, weak candidates below the threshold, provider unavailable, unparseable
output after one repair attempt, and — the interesting one — an answer that was produced
but failed verification.
"""

from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING, Any

from arag.agent.llm import LLMError, LLMErrorKind, LLMRequest
from arag.agent.types import (
    AbstainReason,
    AnswerResult,
    Citation,
    DegradedComponent,
    StageLatency,
)
from arag.obs import get_logger, span
from arag.obs.cost import Usage
from arag.retrieval.types import RetrievalFilters

if TYPE_CHECKING:
    from datetime import date

    from arag.agent.llm import LLMProvider
    from arag.retrieval.protocols import Retriever
    from arag.retrieval.types import RetrievedChunk

log = get_logger(__name__)

SYSTEM_PROMPT = """You answer questions about Indian health insurance policy wordings.

Rules, in priority order:
1. Answer ONLY from the numbered passages provided. You have no other knowledge of these
   policies. If the passages do not contain the answer, say so.
2. Cite the passage number for every factual claim. A claim with no citation is a defect.
3. If the passages conflict, or a figure is conditional on something the question does not
   state, describe the conflict or the condition. Do not choose a branch on the reader's
   behalf.
4. Quote figures, durations and code numbers exactly as written. Never round or convert.
5. If you cannot answer from the passages, set "answer" to null and explain in "reasoning".

Reply with JSON only, no markdown fence:
{"answer": "<answer or null>", "citations": [<passage numbers>], "reasoning": "<one sentence>"}
"""

# One repair attempt, not a loop. A model that emits unparseable JSON twice at temperature 0
# will emit it a third time - the input has not changed - so further retries buy nothing and
# spend the free tier's request budget, which is the scarce resource here.
MAX_REPAIR_ATTEMPTS = 1

_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _strip_fence(text: str) -> str:
    """Models emit fenced JSON despite being told not to. Tolerated, not relied upon."""
    match = _JSON_FENCE.match(text)
    return match.group(1) if match else text


def build_prompt(question: str, candidates: list[RetrievedChunk]) -> str:
    """Number the passages and show their provenance.

    The document id and page are shown because the answer needs to be able to say "the 2025
    wording" — but the citation the model returns is the integer, never the id. See the
    module docstring.
    """
    blocks = []
    for i, chunk in enumerate(candidates, start=1):
        where = f"{chunk.span.document_id}, page {chunk.span.page}"
        if chunk.span.clause_id:
            where += f", clause {chunk.span.clause_id}"
        blocks.append(f"[{i}] ({where})\n{chunk.text}")
    passages = "\n\n".join(blocks)
    return f"PASSAGES:\n\n{passages}\n\nQUESTION: {question}"


def _costs(usage: Usage) -> dict[str, float]:
    """Cost in USD, or zeros with a warning when the model has no price entry.

    ``Usage.marginal_usd`` raises ``KeyError`` for an unpriced model, and that is correct
    *there*: a cost table that silently invents a zero for an unknown model would make the
    cost claims in the README unfalsifiable, which is the whole reason that module refuses.

    It is the wrong behaviour *here*. An unpriced model is a bookkeeping gap, and letting it
    turn a correct, grounded, citable answer into an exception would fail the user for a
    reason that has nothing to do with the answer. So the cost is reported as unknown and
    logged loudly, while the answer survives. The tokens are still recorded, so the gap is
    reconstructible once the price lands.
    """
    if not usage.input_tokens:
        return {"marginal_usd": 0.0, "shadow_usd": 0.0}
    try:
        return {
            "marginal_usd": float(usage.marginal_usd()),
            "shadow_usd": float(usage.shadow_usd()),
        }
    except KeyError as exc:
        log.warning(
            "cost_unknown_for_model",
            detail=str(exc),
            fix="add the model to arag.obs.cost.PRICES, or pass price_key to the engine",
        )
        return {"marginal_usd": 0.0, "shadow_usd": 0.0}


# Strings a model emits when it means "no answer" but the JSON field is still a string.
#
# The prompt says: set "answer" to null. Models write `"answer": "null"` instead - the word,
# in quotes - and a four-character string is perfectly truthy. Measured on the first live
# smoke test: h-24 ("how much premium would a 70-year-old pay?", structurally unanswerable
# because premium tables are not in policy wordings at all) was CORRECTLY declined by the
# model and recorded by this engine as an answer of "null".
#
# That is the worst direction for the error to run. A refusal misread as an answer is a
# false confident response - the exact failure the whole verify-or-abstain design exists to
# prevent - and it was introduced by the layer meant to prevent it.
_DECLINED = frozenset({"null", "none", "n/a", "na", "nil", "unknown", "not applicable", "-"})


def _is_declined(answer: Any) -> bool:
    """Whether the model's `answer` field means "I could not answer"."""
    if answer is None:
        return True
    text = str(answer).strip()
    if not text:
        return True
    return text.casefold().strip(".") in _DECLINED


class GeneratingEngine:
    """The first engine in this project that actually answers."""

    name = "generate"

    def __init__(
        self,
        retriever: Retriever,
        provider: LLMProvider,
        *,
        top_k: int = 10,
        abstain_threshold: float = 0.0,
        price_key: str | None = None,
        name: str | None = None,
    ) -> None:
        self._retriever = retriever
        self._provider = provider
        self._top_k = top_k
        # 0.0 by default and deliberately so: the threshold is CALIBRATED in v4 against the
        # measured cost ratio, and picking a value now - before any false-abstention data
        # exists - would bake a guess into the baseline the calibration is measured against.
        self._abstain_threshold = abstain_threshold
        self._price_key = price_key
        if name:
            self.name = name

    async def answer(self, question: str, *, as_of: date | None = None) -> AnswerResult:
        started = time.perf_counter()
        usage = Usage()

        def finish(
            result: AnswerResult, retrieve_ms: float, generate_ms: float | None
        ) -> AnswerResult:
            return result.model_copy(
                update={
                    "latency": StageLatency(
                        retrieve_ms=retrieve_ms,
                        generate_ms=generate_ms,
                        total_ms=(time.perf_counter() - started) * 1000,
                    ),
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    **_costs(usage),
                    "engine": self.name,
                    "as_of": as_of,
                }
            )

        with span("agent.generate"):
            t0 = time.perf_counter()
            filters = RetrievalFilters(as_of=as_of) if as_of else None
            candidates = await self._retriever.retrieve(
                question, filters=filters, top_k=self._top_k
            )
            retrieve_ms = (time.perf_counter() - t0) * 1000

            base = AnswerResult(question=question, retrieved=tuple(candidates))
            confidence = max((c.confidence for c in candidates), default=0.0)

            if not candidates:
                return finish(
                    base.model_copy(
                        update={
                            "abstained": True,
                            "abstain_reason": AbstainReason.NO_CANDIDATES,
                            "confidence": 0.0,
                        }
                    ),
                    retrieve_ms,
                    None,
                )

            if confidence < self._abstain_threshold:
                # Refusing BEFORE generating, not after. The point of a retrieval-confidence
                # threshold is to avoid spending a request on context that cannot support an
                # answer; generating first and discarding would cost the same as answering.
                log.info(
                    "abstain_low_confidence",
                    confidence=confidence,
                    threshold=self._abstain_threshold,
                )
                return finish(
                    base.model_copy(
                        update={
                            "abstained": True,
                            "abstain_reason": AbstainReason.LOW_CONFIDENCE,
                            "confidence": confidence,
                        }
                    ),
                    retrieve_ms,
                    None,
                )

            t1 = time.perf_counter()
            try:
                parsed = await self._generate(question, candidates, usage)
            except LLMError as exc:
                generate_ms = (time.perf_counter() - t1) * 1000
                log.warning("generation_failed", kind=exc.kind.value, detail=str(exc)[:200])
                return finish(
                    base.model_copy(
                        update={
                            "abstained": True,
                            "abstain_reason": _ABSTAIN_FOR.get(
                                exc.kind, AbstainReason.GENERATION_UNAVAILABLE
                            ),
                            "confidence": confidence,
                            "degraded": (DegradedComponent.GENERATION,),
                        }
                    ),
                    retrieve_ms,
                    generate_ms,
                )
            generate_ms = (time.perf_counter() - t1) * 1000

            answer_text = parsed.get("answer")
            citations, ungrounded = self._verify(parsed.get("citations"), candidates)

            if ungrounded:
                # An answer citing passages that were never retrieved is the failure this
                # whole layer exists to catch. It is DISCARDED, not annotated: shipping it
                # with a warning field would put a fluent wrong answer in front of someone
                # asking whether their surgery is covered.
                log.warning(
                    "ungrounded_citations_discarded",
                    ungrounded=ungrounded,
                    n_candidates=len(candidates),
                )
                return finish(
                    base.model_copy(
                        update={
                            "abstained": True,
                            "abstain_reason": AbstainReason.MALFORMED_GENERATION,
                            "confidence": confidence,
                        }
                    ),
                    retrieve_ms,
                    generate_ms,
                )

            if _is_declined(answer_text):
                # The model followed rule 5 and said it could not answer. That is a correct
                # abstention on its part, and is recorded as LOW_CONFIDENCE rather than as a
                # malformed response - the distinction v4 needs to separate "the system
                # refused well" from "the system broke".
                return finish(
                    base.model_copy(
                        update={
                            "abstained": True,
                            "abstain_reason": AbstainReason.LOW_CONFIDENCE,
                            "confidence": confidence,
                        }
                    ),
                    retrieve_ms,
                    generate_ms,
                )

            return finish(
                base.model_copy(
                    update={
                        "answer": str(answer_text).strip(),
                        "citations": citations,
                        "confidence": confidence,
                    }
                ),
                retrieve_ms,
                generate_ms,
            )

    async def _generate(
        self, question: str, candidates: list[RetrievedChunk], usage: Usage
    ) -> dict[str, Any]:
        """Call the model and parse JSON, repairing at most once."""
        request = LLMRequest(
            model=self._provider.model,
            system=SYSTEM_PROMPT,
            user=build_prompt(question, candidates),
            response_mime_type="application/json",
        )
        last_text = ""
        for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
            response = await self._provider.complete(request)
            usage.record(
                self._price_key or response.model, response.input_tokens, response.output_tokens
            )
            last_text = response.text
            if response.truncated:
                # Reprompting cannot fix this. The output budget ran out, and asking again
                # with the same budget produces the same truncation - which is exactly what
                # the first live run did, burning a second request per item to learn
                # nothing. The flag existed from the start and this branch is the thing
                # that was missing.
                raise LLMError(
                    LLMErrorKind.MALFORMED,
                    f"output truncated at {response.output_tokens} tokens "
                    f"(max_output_tokens={request.max_output_tokens}). On a thinking model "
                    f"the reasoning is billed against the same budget, so raise the budget "
                    f"or set thinking_budget=0 - do not reprompt.",
                )
            try:
                parsed = json.loads(_strip_fence(response.text))
            except json.JSONDecodeError:
                if attempt >= MAX_REPAIR_ATTEMPTS:
                    break
                log.info("repairing_malformed_json", attempt=attempt + 1)
                request = LLMRequest(
                    model=request.model,
                    system=SYSTEM_PROMPT,
                    user=(
                        f"{request.user}\n\nYour previous reply was not valid JSON:\n"
                        f"{response.text[:500]}\n\nReply with JSON only."
                    ),
                    response_mime_type=request.response_mime_type,
                )
                continue
            if not isinstance(parsed, dict):
                break
            return parsed
        raise LLMError(
            LLMErrorKind.MALFORMED,
            f"model output was not a JSON object after {MAX_REPAIR_ATTEMPTS + 1} attempts: "
            f"{last_text[:200]}",
        )

    def _verify(
        self, raw_citations: Any, candidates: list[RetrievedChunk]
    ) -> tuple[tuple[Citation, ...], list[Any]]:
        """Resolve cited passage numbers to retrieved chunks.

        Returns the grounded citations and whatever could not be grounded. Anything not an
        integer in ``1..len(candidates)`` is ungrounded — including a chunk id the model
        decided to quote back, which is exactly the plausible-looking fabrication the
        integer scheme exists to make impossible.
        """
        grounded: list[Citation] = []
        ungrounded: list[Any] = []
        seen: set[str] = set()

        for entry in raw_citations or []:
            try:
                index = int(entry)
            except (TypeError, ValueError):
                ungrounded.append(entry)
                continue
            if not 1 <= index <= len(candidates):
                ungrounded.append(entry)
                continue
            chunk = candidates[index - 1]
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            grounded.append(Citation(span=chunk.span, chunk_id=chunk.chunk_id))
        return tuple(grounded), ungrounded


_ABSTAIN_FOR: dict[LLMErrorKind, AbstainReason] = {
    LLMErrorKind.RATE_LIMITED: AbstainReason.GENERATION_UNAVAILABLE,
    LLMErrorKind.TRANSPORT: AbstainReason.SERVICE_DEGRADED,
    LLMErrorKind.SAFETY_BLOCKED: AbstainReason.GENERATION_BLOCKED,
    LLMErrorKind.MALFORMED: AbstainReason.MALFORMED_GENERATION,
    LLMErrorKind.NOT_CONFIGURED: AbstainReason.GENERATION_NOT_CONFIGURED,
    LLMErrorKind.PROVIDER_SUSPENDED: AbstainReason.GENERATION_FORBIDDEN,
    LLMErrorKind.UNKNOWN: AbstainReason.GENERATION_UNAVAILABLE,
}
