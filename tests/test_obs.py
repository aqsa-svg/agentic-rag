"""Observability tests.

Structured logging and cost accounting are wired in the first commit rather than bolted on
later, so they get tests in the first commit too. The log-level test below pins a bug that
was actually present: module-level ``log = get_logger(__name__)`` runs at import time, so a
configure-once implementation let the first import fix the level for the whole process and
``ARAG_LOG_LEVEL`` silently did nothing.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest
import structlog

from arag.obs.context import current_span_path, current_trace_id, span, trace
from arag.obs.cost import PRICES, Usage, unverified_models
from arag.obs.logging import configure_logging, get_logger


class TestTraceContext:
    def test_trace_id_is_none_outside_a_scope(self) -> None:
        assert current_trace_id() is None

    def test_scope_sets_and_restores(self) -> None:
        with trace("abc123") as tid:
            assert tid == "abc123"
            assert current_trace_id() == "abc123"
        assert current_trace_id() is None

    def test_nested_scope_restores_the_outer_id(self) -> None:
        with trace("outer"):
            with trace("inner"):
                assert current_trace_id() == "inner"
            assert current_trace_id() == "outer"

    def test_span_path_accumulates_and_unwinds(self) -> None:
        with span("retrieval"):
            with span("dense"):
                assert current_span_path() == ("retrieval", "dense")
            assert current_span_path() == ("retrieval",)
        assert current_span_path() == ()

    async def test_concurrent_tasks_do_not_share_a_trace_id(self) -> None:
        """Fluid Compute reuses one instance across concurrent requests, so contextvars
        isolation is not academic: without it, two users' log lines would interleave under
        one trace id and every trace would be unreadable.
        """
        import asyncio

        seen: list[str | None] = []

        async def worker(tid: str) -> None:
            with trace(tid):
                await asyncio.sleep(0)
                seen.append(current_trace_id())

        await asyncio.gather(worker("a"), worker("b"))
        assert sorted(x for x in seen if x) == ["a", "b"]


class TestLogging:
    def test_emits_json_with_trace_id(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging("INFO", force_json=True)
        log = get_logger("test.module")
        with trace("t-1"), span("stage"):
            log.info("something_happened", answer=42)
        payload = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
        assert payload["event"] == "something_happened"
        assert payload["trace_id"] == "t-1"
        assert payload["span"] == "stage"
        assert payload["answer"] == 42
        assert payload["logger"] == "test.module"
        assert payload["level"] == "info"

    def test_level_is_honoured_after_a_logger_was_already_bound(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The regression test for the import-order bug."""
        configure_logging("INFO", force_json=True)
        log = get_logger("test.module")  # bound while the level is INFO
        configure_logging("WARNING", force_json=True)
        capsys.readouterr()

        log.info("should_be_suppressed")
        log.warning("should_appear")

        err = capsys.readouterr().err
        assert "should_be_suppressed" not in err
        assert "should_appear" in err
        configure_logging("INFO", force_json=True)

    def test_unknown_level_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown log level"):
            configure_logging("CHATTY")
        configure_logging("INFO", force_json=True)

    def test_reconfigure_with_same_level_is_a_noop(self) -> None:
        configure_logging("INFO", force_json=True)
        before = structlog.get_config()["wrapper_class"]
        configure_logging("INFO", force_json=True)
        assert structlog.get_config()["wrapper_class"] is before


class TestCost:
    def test_free_tier_marginal_cost_is_zero(self) -> None:
        usage = Usage()
        usage.record("gemini-flash-free", 10_000, 2_000)
        assert usage.marginal_usd() == Decimal("0")

    def test_shadow_cost_is_computed_at_paid_rates(self) -> None:
        """$0 is true and useless. The shadow number is what a reviewer actually wants.

        10,000 in at $0.10/MTok = $0.001; 2,000 out at $0.40/MTok = $0.0008. Total $0.0018.
        """
        usage = Usage()
        usage.record("gemini-flash-free", 10_000, 2_000)
        assert usage.shadow_usd("gemini-flash-paid") == Decimal("0.0018")

    def test_multi_model_costs_are_attributed_per_model(self) -> None:
        usage = Usage()
        usage.record("gemini-flash-paid", 1_000_000, 0)  # $0.10
        usage.record("claude-haiku-4-5", 1_000_000, 0)  # $1.00
        assert usage.marginal_usd() == Decimal("1.10")
        assert usage.calls == 2

    def test_unknown_model_raises_rather_than_costing_zero(self) -> None:
        """A silent zero here would understate cost every time a model is added and
        forgotten, which is precisely how a published cost-per-query figure becomes wrong.
        """
        usage = Usage()
        usage.record("some-new-model", 1000, 100)
        with pytest.raises(KeyError, match="no price entry"):
            usage.marginal_usd()

    def test_local_models_are_free_and_marked_verified(self) -> None:
        assert PRICES["local-ollama"].verified is True
        assert PRICES["local-ollama"].input_per_mtok == Decimal("0")

    def test_unverified_prices_are_reported_for_the_report_banner(self) -> None:
        """Cost figures must be labelled estimates until the rates are checked against
        provider pricing pages. This list is what stamps that banner on EVAL_LOG.md.
        """
        unverified = unverified_models()
        assert "gemini-flash-paid" in unverified
        assert "local-ollama" not in unverified
