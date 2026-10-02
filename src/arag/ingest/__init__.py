"""Ingest package. Re-exports are **lazy**, and that is load-bearing, not tidiness.

## Why this file has a ``__getattr__``

Importing any submodule of a package executes the package's ``__init__.py`` first. When
this file imported ``arag.ingest.probe`` at module scope, that meant:

    import arag.ingest.chunk_types        # a file containing two dataclasses
      -> executes arag/ingest/__init__.py
      -> imports arag.ingest.probe
      -> imports pymupdf

``arag.retrieval.lexical`` needs ``Chunk`` from ``chunk_types``, and ``arag.index.lexical``
needs ``normalise_query`` for the N1 invariant — both correct dependencies. Both therefore
pulled PyMuPDF into the online request path, which is exactly what DESIGN §9's offline/
online split exists to prevent: a ~40MB native dependency and its cold-start cost inside a
serverless function that never opens a PDF.

Recorded as instance 7 in ``docs/SILENT_WRONGNESS.md``. It is the first instance whose
symptom is a production failure (bundle-size limit, cold start) rather than a wrong number,
and the only one no test caught: ``tests/test_import_boundaries.py`` walked *direct* imports
only, so every module involved looked innocent while the closure was not.

## Why lazy rather than deleting the re-exports

``from arag.ingest import Manifest`` is used across the ingest CLI and its tests, and the
flat surface is genuinely convenient for a package whose submodules are implementation
detail. PEP 562 keeps the surface and defers the cost: the attribute is resolved on first
access, so a consumer that only wants ``Chunk`` never imports ``probe``.

The ``TYPE_CHECKING`` block is not decoration — mypy does not follow ``__getattr__``, so
without it every re-export would become ``Any`` and the strict-mode guarantee would quietly
weaken to nothing.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arag.ingest.fetch import FetchOutcome, FetchResult, fetch_all, fetch_source
    from arag.ingest.manifest import (
        RANK_INSURER,
        RANK_MARKETING,
        RANK_REGULATOR,
        Manifest,
        Source,
        SourceKind,
    )
    from arag.ingest.normalise import (
        NormalisationStats,
        NumericCorruptionError,
        normalise_query,
        normalise_text,
    )
    from arag.ingest.probe import DocumentProbe, PageProbe, probe_document, probe_page

# name -> submodule it lives in. Kept explicit rather than derived, so an accidental
# rename produces an AttributeError naming the missing symbol instead of a silent miss.
_LAZY: dict[str, str] = {
    "FetchOutcome": "fetch",
    "FetchResult": "fetch",
    "fetch_all": "fetch",
    "fetch_source": "fetch",
    "RANK_INSURER": "manifest",
    "RANK_MARKETING": "manifest",
    "RANK_REGULATOR": "manifest",
    "Manifest": "manifest",
    "Source": "manifest",
    "SourceKind": "manifest",
    "NormalisationStats": "normalise",
    "NumericCorruptionError": "normalise",
    "normalise_query": "normalise",
    "normalise_text": "normalise",
    "DocumentProbe": "probe",
    "PageProbe": "probe",
    "probe_document": "probe",
    "probe_page": "probe",
}

__all__ = [
    "RANK_INSURER",
    "RANK_MARKETING",
    "RANK_REGULATOR",
    "DocumentProbe",
    "FetchOutcome",
    "FetchResult",
    "Manifest",
    "NormalisationStats",
    "NumericCorruptionError",
    "PageProbe",
    "Source",
    "SourceKind",
    "fetch_all",
    "fetch_source",
    "normalise_query",
    "normalise_text",
    "probe_document",
    "probe_page",
]


def __getattr__(name: str) -> Any:
    """Resolve a re-export on first access (PEP 562)."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value  # cache, so the indirection costs one lookup per process
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
