"""Trace context.

A ``trace_id`` is minted once at the outermost boundary (API request, eval item, ingest
document) and read by every log line and span below it without being threaded through
function signatures. ``contextvars`` is the right primitive because it is asyncio-aware:
each task gets its own copy, so concurrent requests inside one Vercel function instance
cannot bleed trace ids into each other.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

_trace_id: ContextVar[str | None] = ContextVar("arag_trace_id", default=None)
_span_path: ContextVar[tuple[str, ...]] = ContextVar("arag_span_path", default=())


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def current_trace_id() -> str | None:
    return _trace_id.get()


def current_span_path() -> tuple[str, ...]:
    return _span_path.get()


@contextmanager
def trace(trace_id: str | None = None) -> Iterator[str]:
    """Open a new trace scope. Nesting is allowed; the inner scope wins until it exits."""
    tid = trace_id or new_trace_id()
    token: Token[str | None] = _trace_id.set(tid)
    try:
        yield tid
    finally:
        _trace_id.reset(token)


@contextmanager
def span(name: str) -> Iterator[None]:
    """Push a named stage onto the span path, e.g. ``retrieval.dense``.

    This is intentionally not a tracing SDK. It records enough structure for log lines to
    be grouped by stage; Langfuse/OTel spans are layered on top of the same names in v3
    rather than replacing them, so logs and traces always agree on stage naming.
    """
    token = _span_path.set((*_span_path.get(), name))
    try:
        yield
    finally:
        _span_path.reset(token)
