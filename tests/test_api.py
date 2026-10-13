"""Service tests.

The public instance serves IRDAI-only and selects its engine by env var; these tests run
against the default ``null`` stub, so the property under test is the **contract**, not answer
quality: the route layer holds no logic (DESIGN §10), swapping the stub for the real engine
changes nothing here, abstention is a first-class 200 response, no endpoint leaks raw passage
text, quotes are capped, and the service rate-limits. The landing page carries the scope
honesty a live URL needs.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from arag.api.app import MAX_QUOTE_CHARS, RATE_LIMIT_REQUESTS, _cap, create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


class TestHealth:
    def test_health_is_cheap_and_dependency_free(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert "uptime_s" in body and "engine" in body


class TestLandingScope:
    def test_landing_declares_irdai_only_scope(self, client: TestClient) -> None:
        html = client.get("/").text
        assert "IRDAI" in html
        # the honesty a live URL needs: insurer wordings and measured results are NOT served
        assert "not served here" in html
        assert "copyrighted" in html

    def test_landing_serves_an_interactive_ui_wired_to_ask(self, client: TestClient) -> None:
        # the root is a usable browser UI, not just docs: an input, a submit button, example
        # buttons, and vanilla JS that POSTs to the existing /ask (no new backend).
        html = client.get("/").text
        assert '<textarea id="q"' in html
        assert 'id="ask"' in html
        assert "fetch('/ask'" in html
        assert 'class="chip"' in html
        # the example questions: ones the IRDAI corpus answers, and one it does not
        assert 'data-q="What is the grace period' in html
        assert "room rent sub-limit under the Star" in html
        # the hero form submits in place (no reload) and the answer area is a live region
        assert '<form id="askform"' in html
        assert 'id="out" aria-live' in html

    def test_landing_carries_the_measured_abstention_caveat(self, client: TestClient) -> None:
        # a public URL must warn, next to the demo box: the measured failure (2/5 unanswerables
        # refused) is stated, the numbers are attributed to the local 1.5B, and the deployed
        # (production) model's behaviour is called unmeasured rather than implied good.
        html = client.get("/").text
        assert "local 1.5B instrument" in html
        assert "2 of 5" in html
        assert "unmeasured" in html
        assert "not advice" in html


class TestAskContract:
    def test_abstain_is_a_200_with_a_typed_reason(self, client: TestClient) -> None:
        r = client.post("/ask", json={"question": "What is the room rent limit?"})
        assert r.status_code == 200
        body = r.json()
        assert body["abstained"] is True
        assert body["abstain_reason"]  # typed, not null

    def test_full_answerresult_contract_is_present(self, client: TestClient) -> None:
        body = client.post("/ask", json={"question": "What is the cataract sub-limit?"}).json()
        for field in (
            "question",
            "as_of",
            "answer",
            "citations",
            "abstained",
            "abstain_reason",
            "surfaced_conflict",
            "confidence",
            "degraded",
            "usage",
            "latency",
            "engine",
            "trace_id",
        ):
            assert field in body, field
        for u in ("input_tokens", "output_tokens", "marginal_usd", "shadow_usd"):
            assert u in body["usage"], u
        assert "total_ms" in body["latency"]

    def test_as_of_is_accepted_and_echoed(self, client: TestClient) -> None:
        body = client.post(
            "/ask",
            json={"question": "grace period for instalment premiums?", "as_of": "2026-01-01"},
        ).json()
        assert body["as_of"] == "2026-01-01"

    def test_query_is_normalised_on_the_way_in(self, client: TestClient) -> None:
        # the ligature must be folded before it reaches the engine
        body = client.post(
            "/ask", json={"question": "what is the beneﬁt limit for cataract"}
        ).json()
        assert "beneﬁt" not in body["question"] and "benefit" in body["question"]

    def test_two_requests_get_distinct_trace_ids(self, client: TestClient) -> None:
        a = client.post("/ask", json={"question": "room rent?"}).json()
        b = client.post("/ask", json={"question": "room rent?"}).json()
        assert a["trace_id"] and b["trace_id"] and a["trace_id"] != b["trace_id"]

    @pytest.mark.parametrize("bad", ["", "ab"])
    def test_input_bounds_are_enforced(self, client: TestClient, bad: str) -> None:
        assert client.post("/ask", json={"question": bad}).status_code == 422

    def test_missing_body_is_rejected(self, client: TestClient) -> None:
        assert client.post("/ask", json={}).status_code == 422

    def test_wrong_method_is_rejected(self, client: TestClient) -> None:
        assert client.get("/ask").status_code == 405

    def test_no_raw_passage_text_is_returned(self, client: TestClient) -> None:
        # /ask returns citation METADATA only - document, page, clause - never chunk text.
        body = client.post("/ask", json={"question": "room rent?"}).json()
        for c in body["citations"]:
            assert set(c) <= {"document_id", "page", "clause_id", "quote"}


class TestQuoteCap:
    def test_quote_is_capped(self) -> None:
        assert _cap(None) is None
        short = "x" * 10
        assert _cap(short) == short
        long = "y" * (MAX_QUOTE_CHARS + 50)
        out = _cap(long)
        assert out is not None and len(out) <= MAX_QUOTE_CHARS + 1 and out.endswith("…")


class TestRateLimit:
    def test_over_the_window_returns_429(self, client: TestClient) -> None:
        # the limiter is per-client and in-memory; TestClient is one client
        last = None
        for _ in range(RATE_LIMIT_REQUESTS + 1):
            last = client.post("/ask", json={"question": "room rent?"})
        assert last is not None and last.status_code == 429


class TestEngineSelection:
    def test_engine_comes_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAG_ENGINE", "echo")
        body = TestClient(create_app()).get("/health").json()
        assert body["engine"] == "echo"

    def test_unknown_engine_falls_back_to_null(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAG_ENGINE", "does-not-exist")
        assert TestClient(create_app()).get("/health").json()["engine"] == "null"
