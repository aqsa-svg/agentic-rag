"""Structured logging. Configured in the first commit, not bolted on later.

Every log line is a JSON object on stdout carrying ``trace_id`` and ``span`` automatically.
Rationale for the choices here:

* **JSON always in non-TTY, pretty only in a TTY.** Vercel and GitHub Actions capture
  stdout; a human reading a terminal wants colour. Detecting the TTY rather than reading a
  flag means nobody has to remember to set the flag in CI.
* **stdlib logging is routed through structlog**, so a third-party library's ``logging``
  call (httpx, psycopg) lands in the same JSON stream with the same trace id instead of
  printing an unstructured line that breaks log parsing.
* **No file handlers, no rotation.** The runtime is ephemeral; durable records go to
  Postgres (``query_logs``, ``eval_runs``), not to disk.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog

from arag.obs.context import current_span_path, current_trace_id

_configured_level: str | None = None

DEFAULT_LEVEL = "INFO"


def _env_level() -> str:
    return os.environ.get("ARAG_LOG_LEVEL", DEFAULT_LEVEL).upper()


def _rename_logger_name(
    _logger: object, _method: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    """Move ``_logger_name`` to ``logger`` in the output.

    ``structlog.get_logger()`` forwards keyword arguments to ``wrap_logger``, which already
    has a parameter called ``logger`` — so binding that key directly raises a TypeError.
    Binding a private key and renaming it here keeps the lazy proxy (see ``get_logger``)
    while leaving the emitted JSON schema conventional.
    """
    name = event_dict.pop("_logger_name", None)
    if name is not None:
        event_dict.setdefault("logger", name)
    return event_dict


def _inject_trace(
    _logger: object, _method: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    tid = current_trace_id()
    if tid is not None:
        event_dict.setdefault("trace_id", tid)
    path = current_span_path()
    if path:
        event_dict.setdefault("span", ".".join(path))
    return event_dict


def configure_logging(level: str | None = None, *, force_json: bool | None = None) -> None:
    """Configure logging. Safe to call repeatedly.

    The level defaults to ``$ARAG_LOG_LEVEL`` rather than a hardcoded constant, and an
    explicit level always reconfigures. Both details fix a real bug: module-level
    ``log = get_logger(__name__)`` statements run at import time, so with a
    configure-once-and-ignore implementation the first import silently fixed the level for
    the whole process and ``ARAG_LOG_LEVEL=WARNING`` did nothing. Deferring to the
    environment means the setting works no matter which module imports first.
    """
    global _configured_level
    resolved = (level or _env_level()).upper()
    if _configured_level == resolved:
        return

    as_json = force_json if force_json is not None else not sys.stderr.isatty()

    # NOTE: `structlog.stdlib.add_logger_name` is deliberately absent. It reads
    # `logger.name`, which only exists on a stdlib logger, and this configuration uses
    # PrintLoggerFactory for first-party logs to avoid the stdlib round trip on every
    # line. The logger name is bound explicitly in `get_logger` instead.
    shared: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        _rename_logger_name,
        _inject_trace,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if as_json
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    levels = logging.getLevelNamesMapping()
    if resolved not in levels:
        raise ValueError(f"unknown log level {resolved!r}; expected one of {sorted(levels)}")

    structlog.configure(
        processors=[*shared, structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(levels[resolved]),
        # The stream is resolved per call rather than captured here. Binding
        # `file=sys.stderr` once would pin whatever object `sys.stderr` happened to be at
        # configure time, which breaks anything that legitimately swaps the stream later:
        # pytest's capture, `contextlib.redirect_stderr`, and a serverless runtime that
        # rebinds stdio between invocations.
        logger_factory=lambda *_args: structlog.PrintLogger(file=sys.stderr),
        # False so a later reconfigure actually reaches loggers created at import time.
        cache_logger_on_first_use=False,
    )

    # Route stdlib logging (httpx, psycopg, uvicorn) into the same stream. `add_logger_name`
    # is safe *here* because ProcessorFormatter's chain only ever sees stdlib records.
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=[structlog.stdlib.add_logger_name, *shared],
            processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(resolved)

    _configured_level = resolved


def get_logger(name: str, **initial: Any) -> Any:
    """Return a lazily-bound logger carrying its module name.

    Two deliberate choices:

    * The name is bound as a **field** rather than derived by a processor, so the same
      configuration works for both the first-party PrintLogger path and the stdlib path
      used by third-party libraries.
    * The proxy is returned **unbound**. Calling ``.bind()`` here would resolve the
      wrapper class — and therefore the level filter — at import time, so every
      ``log = get_logger(__name__)`` at module scope would permanently freeze the level
      that happened to be active when its module was first imported. Returning the lazy
      proxy means the level is read per call and a later ``configure_logging`` is honoured.
    """
    configure_logging()
    return structlog.get_logger(_logger_name=name, **initial)
