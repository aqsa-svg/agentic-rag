"""Service tests.

The day-1 deploy ships a stub engine on purpose, so the property that matters is not
"does it answer" but **does it refuse to imply that it answers**. A live URL that looks
like a working system is worse than no URL, so `/status` honesty is tested as a contract.

The other property is that the route layer holds no logic (DESIGN §10): swapping the stub
for the real engine must change nothing here.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from arag.api.app import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


class TestHealth:
    def test_health_is_cheap_and_dependency_free(self, client: TestClient) -> None:
        """The container healthcheck hits this, so it must not touch a database, a model
        or the corpus - otherwise a cold dependency marks a live container unhealthy.
        """
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert "uptime_s" in body


class TestStatusHonesty:
    def test_status_declares_the_engine_is_a_stub(self, client: TestClient) -> None:
        body = client.get("/status").json()
        assert body["engine"] == "null"
        assert body["engine_is_stub"] is True

    def test_status_lists_what_is_not_wired(self, client: TestClient) -> None:
        """This list is the difference between an honest day-1 deploy and a demo that
        overstates itself. It must name the missing pieces explicitly.
        """
        missing = client.get("/status").json()["not_wired"]
        for component in ("corpus index", "hybrid retrieval", "reranker"):
            assert component in missing

    def test_status_says_the_eval_gate_has_no_data(self, client: TestClient) -> None:
        assert client.get("/status").json()["eval_gate"] == "no labelled data yet"


class TestAsk:
    def test_stub_abstains_and_says_why(self, client: TestClient) -> None:
        body = client.post(
            "/ask", json={"question": "Is bariatric surgery covered under Star Comprehensive?"}
        ).json()
        assert body["abstained"] is True
        assert body["abstain_reason"] == "not_implemented"
        assert body["engine"] == "null"
        assert body["warning"] and "abstains on every question" in body["warning"]

    def test_response_carries_a_trace_id(self, client: TestClient) -> None:
        """Every request must be traceable from the response alone, or a user report
        cannot be tied to a log line.
        """
        body = client.post("/ask", json={"question": "What is the room rent limit?"}).json()
        assert body["trace_id"]
        assert len(body["trace_id"]) == 16

    def test_two_requests_get_distinct_trace_ids(self, client: TestClient) -> None:
        a = client.post("/ask", json={"question": "What is the room rent limit?"}).json()
        b = client.post("/ask", json={"question": "What is the room rent limit?"}).json()
        assert a["trace_id"] != b["trace_id"]

    def test_query_is_normalised_on_the_way_in(self, client: TestClient) -> None:
        """N1: a user pasting a phrase out of the PDF sends ligatures too. Normalising
        only at index time leaves that query broken in exactly the case where the user is
        quoting the document verbatim.
        """
        body = client.post(
            "/ask", json={"question": "what is the beneﬁt limit for cataract"}
        ).json()
        assert "ﬁ" not in body["question"]
        assert "benefit" in body["question"]

    def test_latency_is_reported(self, client: TestClient) -> None:
        body = client.post("/ask", json={"question": "What is the room rent limit?"}).json()
        assert body["latency_ms"] >= 0

    def test_abstain_is_a_200_not_an_error(self, client: TestClient) -> None:
        """Abstaining is a first-class outcome, not a failure. Returning 4xx/5xx would
        make the abstain-calibration metric unmeasurable through the API and would tell a
        caller to retry something that is working correctly.
        """
        assert (
            client.post("/ask", json={"question": "What is the cataract sub-limit?"}).status_code
            == 200
        )

    @pytest.mark.parametrize("bad", ["", "hi", "x" * 3000])
    def test_input_bounds_are_enforced(self, client: TestClient, bad: str) -> None:
        assert client.post("/ask", json={"question": bad}).status_code == 422

    def test_missing_body_is_rejected(self, client: TestClient) -> None:
        assert client.post("/ask", json={}).status_code == 422

    def test_wrong_method_is_rejected(self, client: TestClient) -> None:
        assert client.get("/ask").status_code == 405


class TestEngineSelection:
    def test_engine_comes_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The engine is swappable by env var so the deployed service can be pointed at a
        real engine without a code change - the route layer holds no logic.
        """
        monkeypatch.setenv("ARAG_ENGINE", "echo")
        client = TestClient(create_app())
        body = client.post("/ask", json={"question": "What is the room rent limit?"}).json()
        assert body["engine"] == "echo"
        assert body["abstained"] is False

    def test_unknown_engine_falls_back_to_the_safest_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A typo in configuration must not start a service that answers confidently.
        Falling back to the abstaining engine fails closed.
        """
        monkeypatch.setenv("ARAG_ENGINE", "does-not-exist")
        client = TestClient(create_app())
        assert client.get("/status").json()["engine"] == "null"
