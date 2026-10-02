"""Index construction, kept out of both ingest and retrieval.

``arag.ingest`` decides what a chunk *is* and ``arag.retrieval`` decides which chunks
answer a question; neither owns the structure in between. Giving the index its own package
is what lets the offline build (PyMuPDF, a GPU) and the online query path (neither) share
one scoring implementation rather than growing two that drift apart.

## Why the re-exports are lazy

That shared-implementation claim is only true if the online path can import the *scorer*
without importing the *builder*. It could not: this file imported ``arag.index.build`` at
module scope, and ``build`` imports PyMuPDF, so ``import arag.index.lexical`` — the one
module a serverless function actually needs, to load a serialised index and score a query —
executed this file and pulled the whole offline stack in behind it.

Same defect as ``arag/ingest/__init__.py``, same root cause: importing a submodule runs the
package first. Instance 7 in ``docs/SILENT_WRONGNESS.md``. PEP 562 keeps the flat surface
that ``arag-ingest`` and the eval harness use while letting the query path pay only for
``lexical``.

A second benefit, which is not the reason but is worth recording: the eager import was one
edge of the ``arag.index`` → ``arag.index.build`` → ``arag.retrieval.lexical`` cycle whose
``ImportError`` ``build.py`` swallows into ``_MISSING_MODULES`` (instance 6). Deferring it
removes the edge. ``tests/test_import_cycle.py`` still guards the swallow, because the
``try/except`` in ``build.py`` remains and a future module-scope import could rebuild the
cycle from the other side.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arag.index.build import (
        BuildReport,
        CorpusIndex,
        QueryMeasurement,
        SerialisationComparison,
        build_corpus,
        compare_serialisations,
        estimate_tokens,
        index_size_bytes,
        latency_summary,
        percentile,
    )
    from arag.index.lexical import Bm25Index, tokenise

_LAZY: dict[str, str] = {
    "BuildReport": "build",
    "CorpusIndex": "build",
    "QueryMeasurement": "build",
    "SerialisationComparison": "build",
    "build_corpus": "build",
    "compare_serialisations": "build",
    "estimate_tokens": "build",
    "index_size_bytes": "build",
    "latency_summary": "build",
    "percentile": "build",
    "Bm25Index": "lexical",
    "tokenise": "lexical",
}

__all__ = [
    "Bm25Index",
    "BuildReport",
    "CorpusIndex",
    "QueryMeasurement",
    "SerialisationComparison",
    "build_corpus",
    "compare_serialisations",
    "estimate_tokens",
    "index_size_bytes",
    "latency_summary",
    "percentile",
    "tokenise",
]


def __getattr__(name: str) -> Any:
    """Resolve a re-export on first access (PEP 562)."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
