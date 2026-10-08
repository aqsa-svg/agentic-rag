"""Concrete ``LLMProvider`` implementations, and the cassette wrapper that stubs them.

## Why raw HTTP rather than the google-generativeai SDK

The SDK pulls grpc, protobuf and a transitive dependency tree into a package that DESIGN §9
caps at Vercel's limit and that `tests/test_import_closure.py` polices. The Gemini REST API
is one POST with a JSON body — the SDK would be ~40MB of bundle to avoid writing twenty
lines, and it would put a vendor's release cadence inside the online path.

``httpx`` is already a dependency (FastAPI's test client uses it) and is the only import
here.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any, ClassVar

from arag.agent.llm import LLMError, LLMErrorKind, LLMProvider, LLMRequest, LLMResponse
from arag.obs import get_logger, span

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arag.eval.cassettes import CassetteStore

log = get_logger(__name__)

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiProvider:
    """Google Gemini over REST.

    The API key is read from the environment and **never** logged, defaulted or embedded.
    A missing key raises ``NOT_CONFIGURED`` at call time rather than sending an empty
    header, because an empty header produces a 401 whose message points at authentication
    rather than at the deployment mistake that actually caused it.
    """

    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-3.6-flash",
        *,
        api_key: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("ARAG_GEMINI_API_KEY") or None
        self._timeout_s = timeout_s

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if not self._api_key:
            raise LLMError(
                LLMErrorKind.NOT_CONFIGURED,
                "ARAG_GEMINI_API_KEY is not set. Generation is unavailable; this is a "
                "deployment fault, not a model failure, and is reported as such so it "
                "cannot be mistaken for a quality result.",
            )

        import httpx

        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": request.user}]}],
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_output_tokens,
            },
        }
        if request.system:
            body["systemInstruction"] = {"parts": [{"text": request.system}]}
        if request.response_mime_type:
            body["generationConfig"]["responseMimeType"] = request.response_mime_type
        if request.thinking_budget is not None:
            body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": request.thinking_budget}

        url = GEMINI_ENDPOINT.format(model=request.model or self.model)
        with span("llm.gemini"):
            try:
                async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                    response = await client.post(
                        url, json=body, headers={"x-goog-api-key": self._api_key}
                    )
            except Exception as exc:
                raise LLMError(LLMErrorKind.TRANSPORT, f"gemini transport failure: {exc}") from exc

        if response.status_code == 429:
            # The server usually says how long to wait, either in the header or in
            # RetryInfo inside the error body. Guessing when we have been told is how a
            # retry schedule ends up shorter than the quota window it is waiting on.
            # A per-DAY quota is not retryable in any useful sense. The server still
            # returns a retryDelay (25s, observed) and honouring it would burn three more
            # requests against a quota that resets at midnight Pacific - the retry cannot
            # succeed, and each attempt makes tomorrow's budget smaller by nothing and
            # today's wait longer by 25 seconds.
            daily = _is_daily_quota(response)
            raise LLMError(
                LLMErrorKind.RATE_LIMITED,
                f"gemini rate limit (429){' - DAILY quota exhausted' if daily else ''}: "
                f"{redact(response.text)[:200]}",
                retryable=not daily,
                retry_after_s=None if daily else _retry_after(response),
            )
        if response.status_code == 403:
            # Never retryable. A suspended project does not recover on its own, and the
            # message names the console rather than leaving the reader to infer it - the
            # whole cost of this failure was two sessions of diagnosing it as something
            # else.
            raise LLMError(
                LLMErrorKind.PROVIDER_SUSPENDED,
                f"gemini HTTP 403 - the API key's project is suspended or the API is not "
                f"enabled for it. This is an ACCOUNT state, not a quota: it will not clear "
                f"by waiting or retrying. Check console.cloud.google.com for the project. "
                f"{redact(response.text)[:200]}",
                retryable=False,
            )
        if response.status_code >= 400:
            # 5xx is upstream trouble and retryable - 503 "high demand" is the one this
            # corpus actually hits, and the next identical call usually succeeds. 4xx other
            # than 429 is a request the caller got wrong; retrying it is pointless and
            # spends the free tier's budget.
            raise LLMError(
                LLMErrorKind.TRANSPORT if response.status_code >= 500 else LLMErrorKind.UNKNOWN,
                f"gemini HTTP {response.status_code}: {redact(response.text)[:300]}",
            )
        return _parse_gemini(response.json(), fallback_model=request.model or self.model)


# Google's own error bodies echo the API key back. A 403 reads:
#   "Permission denied: Consumer 'api_key:AIzaSy...' has been suspended."
# and this module interpolates `response.text` into LLMError messages, which are logged and
# surfaced in eval output. That put a live credential into logs - from code whose only
# mistake was quoting the upstream error faithfully.
#
# Redacting at the boundary rather than at every call site: there is one place a provider
# response becomes a string, and it is here.
_SECRET_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),  # Google AI Studio keys
    re.compile(r"sk-(?:ant-|proj-)?[0-9A-Za-z_\-]{20,}"),  # OpenAI / Anthropic
    re.compile(r"AQ\.[0-9A-Za-z_\-]{20,}"),  # newer Google key shape
)


def redact(text: str) -> str:
    """Mask anything credential-shaped before it reaches a log or an exception message."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda m: f"{m.group(0)[:6]}...REDACTED", text)
    return text


def _is_daily_quota(response: Any) -> bool:
    """Whether the 429 names a per-day quota rather than a per-minute one.

    The distinction decides whether retrying is recovery or waste. Measured on the free
    tier: ``GenerateRequestsPerDayPerProjectPerModel-FreeTier`` with a value of **20** -
    twenty generation requests per day, per model. Backoff cannot outlast that, and the
    eval harness will exhaust it in one run.
    """
    try:
        details = (response.json().get("error") or {}).get("details") or []
    except Exception:
        return False
    for detail in details:
        if not isinstance(detail, dict) or not str(detail.get("@type", "")).endswith(
            "QuotaFailure"
        ):
            continue
        for violation in detail.get("violations") or []:
            if "PerDay" in str(violation.get("quotaId", "")):
                return True
    return False


def _retry_after(response: Any) -> float | None:
    """Seconds the server asked us to wait, from the header or Google's RetryInfo detail."""
    header = response.headers.get("retry-after") if hasattr(response, "headers") else None
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    try:
        details = (response.json().get("error") or {}).get("details") or []
    except Exception:
        return None
    for detail in details:
        delay = detail.get("retryDelay") if isinstance(detail, dict) else None
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                continue
    return None


def _parse_gemini(payload: dict[str, Any], *, fallback_model: str) -> LLMResponse:
    """Turn a Gemini response body into ``LLMResponse``, or a typed error.

    Split out from the HTTP call so the response shapes that matter — a safety block, a
    truncation, an empty candidate list — are testable without a network or a key. Those
    three are the ones that produce a *plausible empty answer* if handled carelessly, which
    is the failure shape this project keeps finding.
    """
    candidates: Sequence[dict[str, Any]] = payload.get("candidates") or []
    feedback = payload.get("promptFeedback") or {}

    if not candidates:
        reason = feedback.get("blockReason")
        if reason:
            raise LLMError(LLMErrorKind.SAFETY_BLOCKED, f"gemini blocked the prompt: {reason}")
        raise LLMError(LLMErrorKind.MALFORMED, "gemini returned no candidates")

    candidate = candidates[0]
    finish = str(candidate.get("finishReason") or "STOP")
    if finish == "SAFETY":
        raise LLMError(LLMErrorKind.SAFETY_BLOCKED, "gemini blocked the response (SAFETY)")

    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(str(p.get("text", "")) for p in parts)

    usage = payload.get("usageMetadata") or {}
    truncated = finish == "MAX_TOKENS"
    if truncated and not text.strip():
        raise LLMError(LLMErrorKind.MALFORMED, "gemini truncated before emitting any text")

    return LLMResponse(
        text=text,
        model=str(payload.get("modelVersion") or fallback_model),
        input_tokens=int(usage.get("promptTokenCount") or 0),
        output_tokens=int(usage.get("candidatesTokenCount") or 0),
        finish_reason=finish,
        truncated=truncated,
        raw=payload,
    )


class CassettedProvider:
    """Wraps any provider so the network call — and only the network call — is stubbed.

    This is the boundary DESIGN §5.9 specifies, and the placement is the whole point. Wrap
    the *engine* instead and a prompt change, a retrieval-depth change or an abstain-
    threshold change would all replay the identical stored answer: the eval numbers would
    not move, and a metric that cannot move is not a metric. Wrapping here means the engine
    re-executes on every replay and only the HTTP round trip is served from disk.

    The cassette key is the full request payload, so an edited prompt is a MISS. In REPLAY
    mode a miss raises rather than falling through to the network — a test that silently
    started making live calls would be slow, costly and non-deterministic without saying so.
    """

    def __init__(self, inner: Any, store: CassetteStore) -> None:
        self._inner = inner
        self._store = store
        self.name = f"cassette({getattr(inner, 'name', 'unknown')})"
        self.model = getattr(inner, "model", "unknown")

    async def complete(self, request: LLMRequest) -> LLMResponse:
        async def call() -> dict[str, Any]:
            response = await self._inner.complete(request)
            return {
                "text": response.text,
                "model": response.model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "finish_reason": response.finish_reason,
                "truncated": response.truncated,
            }

        stored = await self._store.play_or_record("llm", request.cassette_payload(), call)
        return LLMResponse(
            text=str(stored["text"]),
            model=str(stored["model"]),
            input_tokens=int(stored["input_tokens"]),
            output_tokens=int(stored["output_tokens"]),
            finish_reason=str(stored["finish_reason"]),
            truncated=bool(stored["truncated"]),
        )


class RetryingProvider:
    """Retries the retryable failures, with jittered backoff. DESIGN F4.

    ``LLMError.retryable`` existed from the start and nothing consumed it, so the first
    live run reported 6 of 7 questions as ABSTAINED — which reads as a quality result and
    was not one. The actual cause was HTTP 503 "this model is currently experiencing high
    demand": transient upstream overload, where the very next identical call succeeded.

    Retrying only ``retryable`` failures is the point of the taxonomy. A SAFETY_BLOCKED
    response is a decision, not a hiccup, and retrying it burns the free tier's request
    budget that a genuinely transient failure needs. NOT_CONFIGURED will never fix itself
    either.

    ## A quota is not a blip, and one schedule cannot serve both

    DESIGN F4 specifies "2 retries (250ms, 750ms jittered)". Measured against the real free
    tier, that is right for one failure and actively harmful for the other:

    * **503 "high demand"** - a genuine blip. The next call usually succeeds and sub-second
      backoff is correct.
    * **429 quota** - a per-MINUTE window. No sub-two-second schedule can outlast it, so
      every retry is guaranteed to fail AND spends another request against the very quota
      being waited on. Retrying fast makes it strictly worse.

    First live run, seven questions: three retries each at 0.27s / 0.66s / 1.45s, every one
    a 429, six of seven abstaining. The retries rescued nothing and burned 18 extra
    requests doing it.

    So the schedule is per-kind, and the server's own ``retryDelay`` beats both when
    supplied - a quota window is the server's fact, not ours to estimate.

    ## Why jitter, on a single-client workload

    Not for thundering-herd control — there is one client. It is because the eval harness
    fires seven questions in a tight loop, and a fixed backoff makes every retry from that
    loop land in the same narrow window as the burst that caused the overload. Jitter
    spreads them.

    ## Where this sits relative to the cassette

    ``CassettedProvider(RetryingProvider(GeminiProvider()))`` — cassette OUTERMOST. The
    cassette then stores the response that eventually succeeded, and replay is one
    deterministic read with no sleeping. Inverted, a replay miss would be retried against
    a provider the test never intended to call.
    """

    # Per-kind base delay. TRANSPORT keeps DESIGN F4's figure; RATE_LIMITED gets a
    # delay on the order of the quota window it is actually waiting for.
    BASE_DELAY_S: ClassVar[dict[LLMErrorKind, float]] = {
        LLMErrorKind.TRANSPORT: 0.25,
        LLMErrorKind.RATE_LIMITED: 20.0,
    }

    def __init__(
        self,
        inner: LLMProvider,
        *,
        max_attempts: int = 3,
        base_delay_s: float | None = None,
        jitter: float = 0.5,
        max_delay_s: float = 90.0,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._inner = inner
        self._max_attempts = max_attempts
        self._base_delay_s = base_delay_s
        self._jitter = jitter
        self._max_delay_s = max_delay_s
        self.name = f"retry({getattr(inner, 'name', 'unknown')})"
        self.model = getattr(inner, "model", "unknown")

    async def complete(self, request: LLMRequest) -> LLMResponse:
        import asyncio
        import random

        last: LLMError | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return await self._inner.complete(request)
            except LLMError as exc:
                last = exc
                if not exc.retryable or attempt == self._max_attempts:
                    # Re-raised, not swallowed. Exhausted retries are still a failure and
                    # the engine must turn them into a typed abstention rather than this
                    # layer inventing an empty answer.
                    raise
                if exc.retry_after_s is not None:
                    # The server told us how long. Believe it, plus a margin so we do not
                    # land exactly on the boundary of the window.
                    delay = min(exc.retry_after_s + 1.0, self._max_delay_s)
                    source = "server"
                else:
                    base = self._base_delay_s or self.BASE_DELAY_S.get(exc.kind, 0.25)
                    delay = min(base * (2 ** (attempt - 1)), self._max_delay_s)
                    delay *= 1 + random.random() * self._jitter
                    source = "backoff"
                log.warning(
                    "llm_retry",
                    kind=exc.kind.value,
                    attempt=attempt,
                    of=self._max_attempts,
                    delay_s=round(delay, 2),
                    source=source,
                )
                await asyncio.sleep(delay)
        raise last if last else RuntimeError("unreachable")


class ScriptedProvider:
    """A deterministic provider for tests and for measuring the engine without a key.

    Not a mock in the usual sense: it returns text the caller supplies, so a test asserts on
    *the engine's handling* of a given model output rather than on the model. That is the
    only way to test the paths that matter — malformed JSON, a citation to a chunk that was
    never retrieved, an answer with no citations at all — since a real model produces those
    only by accident and never on demand.
    """

    name = "scripted"

    def __init__(
        self,
        responses: Sequence[str] | str,
        *,
        model: str = "scripted-1",
        error: LLMError | None = None,
        input_tokens: int = 100,
        output_tokens: int = 50,
    ) -> None:
        self.model = model
        self._responses = [responses] if isinstance(responses, str) else list(responses)
        self._error = error
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self.calls: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if self._error is not None:
            raise self._error
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        return LLMResponse(
            text=self._responses[index],
            model=self.model,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )
