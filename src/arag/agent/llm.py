"""The LLM provider seam, and the failure taxonomy generation is allowed to have.

## Why a protocol with its own error type

``QueryEngine`` says implementations must not raise for expected failure modes — a rate
limit is an ``AnswerResult`` with ``abstained`` and ``degraded`` set, because the harness
has to *score* the degraded behaviour rather than see a stack trace. That contract is only
keepable if the layer underneath distinguishes its failures. ``LLMError`` carries a typed
``kind`` so the engine can map rate-limiting to one ``AbstainReason`` and a safety block to
another, which is what makes false-abstention rate decomposable in v4.

A bare ``except Exception`` at the engine boundary would collapse F4 (rate limit), F5
(malformed output) and F6 (safety block) into one number, and those three need different
fixes.

## Why the cassette wraps THIS layer and not the engine

DESIGN §5.9: cassettes record at the **provider boundary**. Recording at the engine
boundary would be easier and would destroy the harness — a change to the prompt, the
retrieval depth, or the abstain threshold would replay the identical stored answer and the
eval numbers would not move. Metrics that cannot move are not metrics. Recording the raw
request/response pair means the engine is re-executed on every replay and only the network
call is stubbed.

The key is the full request payload, so a changed prompt is a cassette *miss* rather than a
silently stale hit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable


class LLMErrorKind(StrEnum):
    """Why generation failed. Each maps to a different AbstainReason and a different fix."""

    RATE_LIMITED = "rate_limited"  # F4 - retry, then degrade
    SAFETY_BLOCKED = "safety_blocked"  # F6 - the provider refused; never retry
    MALFORMED = "malformed"  # F5 - unparseable output, repair once
    TRANSPORT = "transport"  # network, DNS, timeout
    NOT_CONFIGURED = "not_configured"  # no API key - a deployment fault, not a model one
    # 403 CONSUMER_SUSPENDED. An ACCOUNT state: permanent until a human acts in a billing
    # console, and distinct from both a quota (waits out) and a missing key (fix the env).
    # Collapsing it into either sends the reader to the wrong console.
    PROVIDER_SUSPENDED = "provider_suspended"
    UNKNOWN = "unknown"


class LLMError(RuntimeError):
    def __init__(
        self,
        kind: LLMErrorKind,
        message: str,
        *,
        retryable: bool | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        # What the SERVER said to wait, when it said anything. Preferred over any schedule
        # the client invents: a quota window is the server's fact, not ours to estimate.
        self.retry_after_s = retry_after_s
        # Retryability is a property of the failure, decided here rather than by the caller
        # re-deriving it from a string. SAFETY_BLOCKED is never retryable: retrying a
        # refusal burns the rate-limit budget that a genuinely transient failure needs.
        self.retryable = (
            retryable
            if retryable is not None
            else kind in {LLMErrorKind.RATE_LIMITED, LLMErrorKind.TRANSPORT}
        )


@dataclass(frozen=True)
class LLMRequest:
    """One generation request. Also the cassette key, so every field must be deterministic.

    ``temperature`` is 0.0 by default and that is load-bearing for the eval, not a style
    choice: a stochastic generator makes a regression indistinguishable from a resample,
    and the gate would flake. v4's judge-agreement work needs a fixed seed for the same
    reason.
    """

    model: str
    system: str
    user: str
    temperature: float = 0.0
    # 4096, not 1024, and the reason is measured rather than cautious. On a THINKING model
    # (gemini-3.x) internal reasoning is billed against this same budget, so a 1024 ceiling
    # left ~39 tokens of visible output and every JSON reply was truncated mid-string:
    # `{"answer": "Pre-existing Diseases (PED) are excluded for 36 months of continuous`
    # and then nothing. Valid-looking, unparseable, and identical in shape to a model that
    # simply cannot produce JSON.
    max_output_tokens: int = 4096
    # None leaves the provider default. 0 disables thinking where supported, which is the
    # cheaper fix for a structured-output task that needs no deliberation - but it is left
    # unset so the choice is made per call site rather than assumed here.
    thinking_budget: int | None = None
    response_mime_type: str | None = None

    def cassette_payload(self) -> dict[str, object]:
        """Everything that can change the response. A changed prompt must MISS, not hit."""
        return {
            "model": self.model,
            "system": self.system,
            "user": self.user,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "thinking_budget": self.thinking_budget,
            "response_mime_type": self.response_mime_type,
        }


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = "stop"
    # True when the provider truncated at max_output_tokens. Kept separate from the text
    # because a truncated JSON answer is a MALFORMED failure with a known cause, and
    # telling the two apart decides whether the fix is a bigger budget or a better prompt.
    truncated: bool = False
    raw: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    """A text-in, text-out model call. No retrieval, no parsing, no business logic.

    Deliberately this thin. Everything the engine does — prompt construction, citation
    grounding, the abstain decision — stays testable without a network call, and swapping
    Gemini for a local model is one class rather than a refactor.
    """

    name: str
    model: str

    async def complete(self, request: LLMRequest) -> LLMResponse: ...
