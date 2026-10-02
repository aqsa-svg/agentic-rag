"""The generating engine, tested on the outputs a real model produces only by accident.

A live model at temperature 0 will almost always return well-formed JSON citing real
passages. That is exactly why it is useless for testing this layer: the paths that matter —
a citation to a passage that was never retrieved, unparseable JSON, a safety block, a rate
limit — are the ones a real call will not reproduce on demand. ``ScriptedProvider`` returns
text the test chooses, so every assertion is about *the engine's handling* of a given model
output rather than about the model.

The ungrounded-citation test is the important one. It is the difference between a demo and
a system: without it, a fluent answer citing a clause the retriever never returned reaches
someone asking whether their surgery is covered.
"""

from __future__ import annotations

import json

import pytest

from arag.agent.generate import GeneratingEngine, build_prompt
from arag.agent.llm import LLMError, LLMErrorKind, LLMRequest
from arag.agent.providers import CassettedProvider, ScriptedProvider, _parse_gemini
from arag.agent.types import AbstainReason, DegradedComponent
from arag.eval.cassettes import CassetteMiss, CassetteMode, CassetteStore
from arag.retrieval.types import DocSpan, RetrievalFilters, RetrievedChunk


def chunk(chunk_id: str, text: str, page: int = 1, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        span=DocSpan(document_id="star-comprehensive-2025", page=page, clause_id=chunk_id),
        text=text,
        dense_score=score,
    )


class StubRetriever:
    name = "stub"

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.calls: list[tuple[str, RetrievalFilters | None, int]] = []

    async def retrieve(
        self, query: str, *, filters: RetrievalFilters | None = None, top_k: int = 30
    ) -> list[RetrievedChunk]:
        self.calls.append((query, filters, top_k))
        return self._chunks[:top_k]


@pytest.fixture
def candidates() -> list[RetrievedChunk]:
    return [
        chunk("c1", "30-day waiting period - Code Excl 03.", page=32),
        chunk("c2", "Specified disease waiting period - Code Excl 02.", page=31, score=0.7),
    ]


def reply(answer: object, citations: list[object]) -> str:
    return json.dumps({"answer": answer, "citations": citations, "reasoning": "because"})


class TestHappyPath:
    async def test_answers_and_grounds_its_citations(self, candidates) -> None:  # type: ignore[no-untyped-def]
        engine = GeneratingEngine(
            StubRetriever(candidates), ScriptedProvider(reply("30 days.", [1]))
        )
        result = await engine.answer("when am I covered?")

        assert not result.abstained
        assert result.answer == "30 days."
        assert [c.chunk_id for c in result.citations] == ["c1"]
        assert result.citations[0].span.page == 32
        assert result.confidence == pytest.approx(0.9)
        assert result.retrieved == tuple(candidates)

    async def test_usage_and_cost_are_recorded(self, candidates) -> None:  # type: ignore[no-untyped-def]
        engine = GeneratingEngine(
            StubRetriever(candidates),
            ScriptedProvider(reply("30 days.", [1]), input_tokens=800, output_tokens=40),
            price_key="gemini-flash-free",
        )
        result = await engine.answer("when am I covered?")
        assert result.input_tokens == 800
        assert result.output_tokens == 40
        # Free tier: marginal cost is zero, shadow cost is not. Reporting only the marginal
        # number would make the system look free at any scale.
        assert result.marginal_usd == 0.0
        assert result.shadow_usd > 0.0

    async def test_duplicate_citations_collapse(self, candidates) -> None:  # type: ignore[no-untyped-def]
        engine = GeneratingEngine(
            StubRetriever(candidates), ScriptedProvider(reply("30 days.", [1, 1, 1]))
        )
        result = await engine.answer("q?")
        assert len(result.citations) == 1

    async def test_as_of_becomes_a_retrieval_filter(self, candidates) -> None:  # type: ignore[no-untyped-def]
        """The supersession control has to reach the retriever, or it does nothing."""
        from datetime import date

        retriever = StubRetriever(candidates)
        engine = GeneratingEngine(retriever, ScriptedProvider(reply("x", [1])))
        await engine.answer("q?", as_of=date(2026, 1, 1))
        _query, filters, _k = retriever.calls[0]
        assert filters is not None and filters.as_of == date(2026, 1, 1)


class TestUngroundedCitationsAreDiscarded:
    """An answer citing passages that were never retrieved is discarded, not annotated.

    This is the layer's reason to exist. Shipping such an answer with a warning field would
    put a fluent, confident, wrong answer in front of someone asking whether their surgery
    is covered - and it would be indistinguishable from a good answer to everyone except
    the person who checks the citation.
    """

    @pytest.mark.parametrize(
        ("label", "citations"),
        [
            ("out of range high", [7]),
            ("zero is not a passage", [0]),
            ("negative", [-1]),
            ("a chunk id the model invented", ["star-comprehensive-2025:p32:s146"]),
            ("a plausible but fabricated id", ["c3"]),
            ("nonsense", [{"page": 32}]),
        ],
    )
    async def test_the_whole_answer_is_refused(self, candidates, label, citations) -> None:  # type: ignore[no-untyped-def]
        engine = GeneratingEngine(
            StubRetriever(candidates), ScriptedProvider(reply("Confidently wrong.", citations))
        )
        result = await engine.answer("q?")
        assert result.abstained, label
        assert result.abstain_reason is AbstainReason.MALFORMED_GENERATION
        assert result.answer is None, "the ungrounded answer must not survive"

    async def test_one_bad_citation_poisons_the_answer(self, candidates) -> None:  # type: ignore[no-untyped-def]
        """Partial grounding is not grounding.

        Keeping the valid citation and dropping the invalid one would emit an answer whose
        remaining citation looks like it supports the whole claim. The claim that rested on
        the fabricated passage would still be in the text, now apparently sourced.
        """
        engine = GeneratingEngine(
            StubRetriever(candidates), ScriptedProvider(reply("Two claims.", [1, 99]))
        )
        result = await engine.answer("q?")
        assert result.abstained
        assert result.answer is None


class TestMalformedOutput:
    async def test_repairs_once_then_succeeds(self, candidates) -> None:  # type: ignore[no-untyped-def]
        provider = ScriptedProvider(["this is not json", reply("30 days.", [1])])
        engine = GeneratingEngine(StubRetriever(candidates), provider)
        result = await engine.answer("q?")
        assert not result.abstained
        assert len(provider.calls) == 2
        assert "not valid JSON" in provider.calls[1].user

    async def test_gives_up_after_one_repair(self, candidates) -> None:  # type: ignore[no-untyped-def]
        """Not a retry loop. At temperature 0 with unchanged input, a third call returns the
        same unparseable text and spends the free tier's scarce request budget.
        """
        provider = ScriptedProvider(["nope", "still nope", "would have worked"])
        engine = GeneratingEngine(StubRetriever(candidates), provider)
        result = await engine.answer("q?")
        assert result.abstained
        assert result.abstain_reason is AbstainReason.MALFORMED_GENERATION
        assert len(provider.calls) == 2

    async def test_truncation_is_not_reprompted(self, candidates) -> None:  # type: ignore[no-untyped-def]
        """Measured on the first live run: 5 of 7 questions truncated, none recoverable.

        gemini-3.6-flash is a thinking model, so internal reasoning is billed against
        `maxOutputTokens`. A 1024 budget left ~39 visible tokens and every JSON reply was
        cut mid-string. The engine reported generic `malformed` and reprompted with the
        SAME budget, which cannot succeed and spent a second request per item.

        `LLMResponse.truncated` was being set correctly the whole time and nothing read it.
        This test is the thing that stops that happening again.
        """
        from arag.agent.llm import LLMResponse

        class TruncatingProvider:
            name, model = "truncating", "thinky-1"

            def __init__(self) -> None:
                self.calls = 0

            async def complete(self, request):  # type: ignore[no-untyped-def]
                self.calls += 1
                return LLMResponse(
                    text='{"answer": "Pre-existing Diseases are excluded for 36 months of',
                    model=self.model,
                    output_tokens=39,
                    finish_reason="MAX_TOKENS",
                    truncated=True,
                )

        provider = TruncatingProvider()
        engine = GeneratingEngine(StubRetriever(candidates), provider)
        result = await engine.answer("q?")
        assert result.abstained
        assert result.abstain_reason is AbstainReason.MALFORMED_GENERATION
        assert provider.calls == 1, "a truncated reply must not be reprompted"

    def test_the_default_output_budget_accounts_for_thinking_tokens(self) -> None:
        """1024 was not enough on a thinking model. Pinned so it is not quietly lowered."""
        assert LLMRequest(model="m", system="s", user="u").max_output_tokens >= 4096

    async def test_a_fenced_json_reply_is_tolerated(self, candidates) -> None:  # type: ignore[no-untyped-def]
        fenced = "```json\n" + reply("30 days.", [1]) + "\n```"
        engine = GeneratingEngine(StubRetriever(candidates), ScriptedProvider(fenced))
        assert not (await engine.answer("q?")).abstained

    async def test_a_json_array_is_not_an_answer(self, candidates) -> None:  # type: ignore[no-untyped-def]
        """Valid JSON of the wrong shape must not be treated as a parsed answer."""
        engine = GeneratingEngine(StubRetriever(candidates), ScriptedProvider('["30 days"]'))
        result = await engine.answer("q?")
        assert result.abstained
        assert result.abstain_reason is AbstainReason.MALFORMED_GENERATION


class TestAbstentionIsDecomposable:
    """Each failure records a DIFFERENT reason, so v4 can decompose false-abstention rate.

    Collapsing a rate limit, a safety block and an empty corpus into one number would make
    the metric unactionable: three causes with three different fixes, one statistic.
    """

    async def test_no_candidates(self) -> None:
        engine = GeneratingEngine(StubRetriever([]), ScriptedProvider(reply("x", [])))
        result = await engine.answer("q?")
        assert result.abstain_reason is AbstainReason.NO_CANDIDATES
        assert result.confidence == 0.0

    async def test_below_threshold_does_not_call_the_model(self, candidates) -> None:  # type: ignore[no-untyped-def]
        """Refusing before generating. Generating and then discarding costs the same as
        answering, which defeats the purpose of a retrieval-confidence threshold.
        """
        provider = ScriptedProvider(reply("x", [1]))
        engine = GeneratingEngine(StubRetriever(candidates), provider, abstain_threshold=0.95)
        result = await engine.answer("q?")
        assert result.abstain_reason is AbstainReason.LOW_CONFIDENCE
        assert provider.calls == [], "the model must not be called below the threshold"

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            (LLMErrorKind.RATE_LIMITED, AbstainReason.GENERATION_UNAVAILABLE),
            (LLMErrorKind.SAFETY_BLOCKED, AbstainReason.GENERATION_BLOCKED),
            (LLMErrorKind.TRANSPORT, AbstainReason.SERVICE_DEGRADED),
            # These two used to BOTH map to GENERATION_UNAVAILABLE alongside RATE_LIMITED.
            # Two live runs then reported "7 abstained (generation_unavailable)" for an
            # un-prefixed env var (zero network calls) and for a suspended project - tables
            # identical, causes unrelated, consoles different.
            (LLMErrorKind.NOT_CONFIGURED, AbstainReason.GENERATION_NOT_CONFIGURED),
            (LLMErrorKind.PROVIDER_SUSPENDED, AbstainReason.GENERATION_FORBIDDEN),
        ],
    )
    async def test_provider_failures_map_to_distinct_reasons(
        self,
        candidates,  # type: ignore[no-untyped-def]
        kind: LLMErrorKind,
        expected: AbstainReason,
    ) -> None:
        provider = ScriptedProvider("", error=LLMError(kind, "boom"))
        engine = GeneratingEngine(StubRetriever(candidates), provider)
        result = await engine.answer("q?")
        assert result.abstained
        assert result.abstain_reason is expected
        assert DegradedComponent.GENERATION in result.degraded

    def test_every_account_side_failure_has_its_own_reason(self) -> None:
        """No two account-side causes may share an abstain reason.

        The point of the taxonomy is that a reader of the results table can tell what to
        DO. A missing credential is fixed in `.env`; a suspension is fixed in a billing
        console; a quota is fixed by waiting or paying. One statistic covering all three
        tells you none of them.
        """
        from arag.agent.generate import _ABSTAIN_FOR

        account_side = (
            LLMErrorKind.NOT_CONFIGURED,
            LLMErrorKind.PROVIDER_SUSPENDED,
            LLMErrorKind.RATE_LIMITED,
        )
        reasons = [_ABSTAIN_FOR[k] for k in account_side]
        assert len(set(reasons)) == len(reasons), f"collapsed: {reasons}"

    def test_a_suspension_is_never_retried(self) -> None:
        """A suspended project does not recover by waiting, and each retry costs a request."""
        err = LLMError(LLMErrorKind.PROVIDER_SUSPENDED, "403")
        assert not err.retryable

    async def test_the_engine_never_raises_for_an_expected_failure(self, candidates) -> None:  # type: ignore[no-untyped-def]
        """QueryEngine's contract. A raise here would be an unscoreable item rather than a
        scoreable degradation, and the harness would lose the failure entirely.
        """
        provider = ScriptedProvider("", error=LLMError(LLMErrorKind.RATE_LIMITED, "429"))
        engine = GeneratingEngine(StubRetriever(candidates), provider)
        assert (await engine.answer("q?")).abstained

    async def test_a_model_declining_to_answer_is_low_confidence_not_malformed(
        self,
        candidates,  # type: ignore[no-untyped-def]
    ) -> None:
        """Rule 5 of the prompt. The model refusing well is not the system breaking, and
        v4 needs to tell those apart.
        """
        engine = GeneratingEngine(StubRetriever(candidates), ScriptedProvider(reply(None, [])))
        result = await engine.answer("q?")
        assert result.abstain_reason is AbstainReason.LOW_CONFIDENCE


class TestPrompt:
    def test_passages_are_numbered_from_one_with_provenance(self, candidates) -> None:  # type: ignore[no-untyped-def]
        prompt = build_prompt("when?", candidates)
        assert "[1] (star-comprehensive-2025, page 32, clause c1)" in prompt
        assert "[2] (star-comprehensive-2025, page 31, clause c2)" in prompt
        assert "QUESTION: when?" in prompt

    def test_the_prompt_forbids_outside_knowledge_and_rounding(self) -> None:
        from arag.agent.generate import SYSTEM_PROMPT

        assert "ONLY from the numbered passages" in SYSTEM_PROMPT
        assert "Never round" in SYSTEM_PROMPT


class TestGeminiResponseParsing:
    """The shapes that produce a plausible empty answer if handled carelessly."""

    def test_a_normal_response(self) -> None:
        response = _parse_gemini(
            {
                "candidates": [{"content": {"parts": [{"text": "hello"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3},
                "modelVersion": "gemini-2.0-flash",
            },
            fallback_model="x",
        )
        assert response.text == "hello"
        assert response.input_tokens == 10 and response.output_tokens == 3

    def test_a_blocked_prompt_raises_safety_not_malformed(self) -> None:
        with pytest.raises(LLMError) as exc:
            _parse_gemini({"promptFeedback": {"blockReason": "SAFETY"}}, fallback_model="x")
        assert exc.value.kind is LLMErrorKind.SAFETY_BLOCKED
        assert not exc.value.retryable, "retrying a refusal burns the rate-limit budget"

    def test_a_blocked_candidate_raises_safety(self) -> None:
        with pytest.raises(LLMError) as exc:
            _parse_gemini({"candidates": [{"finishReason": "SAFETY"}]}, fallback_model="x")
        assert exc.value.kind is LLMErrorKind.SAFETY_BLOCKED

    def test_no_candidates_and_no_reason_is_malformed(self) -> None:
        with pytest.raises(LLMError) as exc:
            _parse_gemini({"candidates": []}, fallback_model="x")
        assert exc.value.kind is LLMErrorKind.MALFORMED

    def test_truncation_with_text_is_flagged_not_raised(self) -> None:
        """A truncated answer is still an answer; the flag says the budget was the cause."""
        response = _parse_gemini(
            {
                "candidates": [
                    {"content": {"parts": [{"text": '{"a":'}]}, "finishReason": "MAX_TOKENS"}
                ]
            },
            fallback_model="x",
        )
        assert response.truncated

    def test_truncation_with_no_text_is_malformed(self) -> None:
        with pytest.raises(LLMError):
            _parse_gemini(
                {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]},
                fallback_model="x",
            )


class TestCassetteBoundary:
    """DESIGN §5.9: recording at the PROVIDER boundary, not the engine boundary."""

    async def test_record_then_replay_without_the_provider(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        inner = ScriptedProvider(reply("30 days.", [1]))
        store = CassetteStore(tmp_path, CassetteMode.RECORD)
        await CassettedProvider(inner, store).complete(LLMRequest(model="m", system="s", user="u"))
        assert store.recorded == 1

        dead = ScriptedProvider("", error=LLMError(LLMErrorKind.TRANSPORT, "no network"))
        replay = CassetteStore(tmp_path, CassetteMode.REPLAY)
        response = await CassettedProvider(dead, replay).complete(
            LLMRequest(model="m", system="s", user="u")
        )
        assert json.loads(response.text)["answer"] == "30 days."
        assert replay.hits == 1

    async def test_a_changed_prompt_MISSES_rather_than_replaying_a_stale_answer(
        self,
        tmp_path,  # type: ignore[no-untyped-def]
    ) -> None:
        """The property that makes cassettes safe for an eval harness.

        If an edited prompt replayed the old recording, every prompt change would score
        identically to the prompt it replaced and the harness would report no delta for a
        real change. A metric that cannot move is not a metric.
        """
        store = CassetteStore(tmp_path, CassetteMode.RECORD)
        provider = CassettedProvider(ScriptedProvider(reply("a", [1])), store)
        await provider.complete(LLMRequest(model="m", system="s", user="original"))

        replay = CassetteStore(tmp_path, CassetteMode.REPLAY)
        with pytest.raises(CassetteMiss):
            await CassettedProvider(ScriptedProvider("x"), replay).complete(
                LLMRequest(model="m", system="s", user="EDITED PROMPT")
            )

    async def test_the_engine_still_runs_on_replay(self, candidates, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Only the network call is stubbed: verification must still reject a bad citation
        served from a cassette, exactly as it would from a live call.
        """
        store = CassetteStore(tmp_path, CassetteMode.RECORD)
        provider = CassettedProvider(ScriptedProvider(reply("wrong", [99])), store)
        engine = GeneratingEngine(StubRetriever(candidates), provider)
        assert (await engine.answer("q?")).abstain_reason is AbstainReason.MALFORMED_GENERATION


class TestGeminiProviderConfiguration:
    async def test_a_missing_key_is_a_deployment_fault_not_a_model_failure(self) -> None:
        from arag.agent.providers import GeminiProvider

        provider = GeminiProvider(api_key=None)
        if provider.configured:  # pragma: no cover - only when a real key is exported
            pytest.skip("ARAG_GEMINI_API_KEY is set in this environment")
        with pytest.raises(LLMError) as exc:
            await provider.complete(LLMRequest(model="m", system="s", user="u"))
        assert exc.value.kind is LLMErrorKind.NOT_CONFIGURED


class TestCredentialsNeverReachLogs:
    """Google's own 403 body echoes the API key back:

        "Permission denied: Consumer 'api_key:AIzaSy...' has been suspended."

    and this provider interpolates `response.text` into LLMError messages, which are logged
    and surfaced in eval output. Quoting the upstream error faithfully put a live credential
    into logs. Redaction happens at the one place a provider response becomes a string.
    """

    # Fabricated keys of the right SHAPE. A real credential must never be a test
    # fixture: it would be committed, and a redaction test that leaks the thing it
    # redacts is worse than no test. This file previously held a live key.
    @pytest.mark.parametrize(
        "secret",
        [
            "AIzaSyFAKE0000000000000000000000000000",
            "sk-ant-api03-abcdefghijklmnopqrstuvwxyz012345",
            "AQ.Ab8RN6JmQwErTyUiOpAsDfGhJkLzXcVbNm1234567890",
        ],
    )
    def test_credential_shapes_are_masked(self, secret: str) -> None:
        from arag.agent.providers import redact

        masked = redact(f"Permission denied: Consumer 'api_key:{secret}' has been suspended.")
        assert secret not in masked
        assert "REDACTED" in masked
        assert "Permission denied" in masked, "the diagnostic text must survive"

    def test_ordinary_text_is_untouched(self) -> None:
        from arag.agent.providers import redact

        text = "gemini HTTP 503: This model is currently experiencing high demand."
        assert redact(text) == text

    def test_the_patterns_contain_no_control_characters(self) -> None:
        """A `\b` written into this file through a shell heredoc becomes a literal
        backspace byte (0x08), which silently stops the pattern matching anything. That
        happened here, and to the traceback in docs/BUILD_RECORD.md before it. Invisible in
        grep output and in an editor.
        """
        from arag.agent.providers import _SECRET_PATTERNS

        for pattern in _SECRET_PATTERNS:
            assert not any(ord(c) < 32 for c in pattern.pattern), (
                f"control character in {pattern.pattern!r} - it will match nothing"
            )


class TestADeclinedAnswerIsNotAnAnswer:
    """The prompt asks for JSON `null`; models write the STRING "null".

    Found on the first live smoke test, not by any scripted test. h-24 asks what premium a
    70-year-old would pay - structurally unanswerable, premium tables are not in policy
    wordings - and Qwen2.5-1.5B correctly declined by emitting `"answer": "null"`. A
    four-character string is truthy, so the engine recorded a correct refusal AS AN ANSWER.

    That is the worst direction for the error to run: a refusal misread as an answer is a
    false confident response, which is the precise failure the verify-or-abstain design
    exists to prevent, introduced by the layer meant to prevent it.
    """

    @pytest.mark.parametrize(
        "declined",
        [None, "", "   ", "null", "NULL", "Null", "none", "None.", "N/A", "n/a", "nil", "-"],
    )
    async def test_declined_forms_abstain(self, candidates, declined) -> None:  # type: ignore[no-untyped-def]
        engine = GeneratingEngine(StubRetriever(candidates), ScriptedProvider(reply(declined, [1])))
        result = await engine.answer("q?")
        assert result.abstained, f"{declined!r} was treated as an answer"
        assert result.abstain_reason is AbstainReason.LOW_CONFIDENCE
        assert result.answer is None

    @pytest.mark.parametrize(
        "real",
        [
            "No",
            "No.",
            "36 months",
            "Not covered unless necessitated by an accident",
            "None of the listed exclusions apply to this claim",
        ],
    )
    async def test_real_answers_survive(self, candidates, real) -> None:  # type: ignore[no-untyped-def]
        """The guard must not eat genuine short answers.

        "No" is a complete and correct answer to a yes/no coverage question, and "None of
        the listed exclusions apply" opens with a declined-looking word. Matching on the
        WHOLE stripped field rather than a substring is what keeps both.
        """
        engine = GeneratingEngine(StubRetriever(candidates), ScriptedProvider(reply(real, [1])))
        result = await engine.answer("q?")
        assert not result.abstained, f"{real!r} is a real answer and was discarded"
        assert result.answer == real
