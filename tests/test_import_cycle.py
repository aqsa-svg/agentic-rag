"""Guard against a silently-absent module caused by a circular import.

## Why this test exists

``arag.index.build`` imports its collaborators — the chunker, the BM25 index, the lexical
retriever — inside ``try/except ImportError`` and records what failed in
``_MISSING_MODULES``, so that the module is importable before its collaborators exist.
That was the right call while three modules were being written in parallel. It also
created a trap.

``arag/index/__init__.py`` imports ``arag.index.build`` at module scope, and
``arag.index.build`` imports ``arag.retrieval.lexical`` at module scope. If anything in
that chain imports ``arag.index`` back, Python hands the partially-initialised module
back, the import raises, and the ``except ImportError`` **swallows it**. The result is not
a crash: it is ``build_corpus`` reporting that the lexical retriever is unavailable,
depending on which module the process happened to import first.

That is the sixth instance of the pattern in ``docs/SILENT_WRONGNESS.md``: no error, looks
fine, a module silently absent. It is worse than most, because the observable symptom
("lexical retriever unavailable") points at the wrong module entirely, and because it is
**import-order dependent** — it can pass every test and fail in the deployed service, or
vice versa.

## Why subprocesses

Python caches modules in ``sys.modules``, so the first import in a process fixes the
order for every later one. Testing three orders therefore requires three interpreters.
Asserting on the current process would test exactly one order — whichever pytest's
collection happened to produce — and would give a false pass for the other two.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# Each entry imports modules in a different order, then reports what build.py thinks is
# missing. The three orders are the ones that can realistically occur:
#   retrieval-first  - the API path, which reaches retrieval before any index module
#   build-first      - the eval path, which builds a corpus
#   package-first    - anything doing `import arag.index`
IMPORT_ORDERS: dict[str, str] = {
    "retrieval_first": "import arag.retrieval.lexical; import arag.index.build as b",
    "build_first": "import arag.index.build as b",
    "package_first": "import arag.index; import arag.index.build as b",
    "chunk_first": "import arag.ingest.chunk; import arag.index.build as b",
    "index_lexical_first": "import arag.index.lexical; import arag.index.build as b",
}

PROBE = (
    "{order}\n"
    "missing = getattr(b, '_MISSING_MODULES', None)\n"
    "if missing is None:\n"
    "    print('NO_ATTR')\n"
    "else:\n"
    "    print('MISSING=' + repr(sorted(missing)))\n"
)


def _probe(order: str) -> str:
    """Run one import order in a fresh interpreter and return its verdict line."""
    result = subprocess.run(
        [sys.executable, "-c", PROBE.format(order=order)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        pytest.fail(f"import order {order!r} raised:\n{result.stderr[-1500:]}")
    return result.stdout.strip().splitlines()[-1]


class TestNoSilentlyMissingModules:
    @pytest.mark.parametrize("name", sorted(IMPORT_ORDERS))
    def test_collaborators_resolve_under_every_import_order(self, name: str) -> None:
        """``_MISSING_MODULES`` must be empty regardless of what was imported first.

        A non-empty set here does NOT mean an exception was raised — it means one was
        caught and discarded, and that ``build_corpus`` will now fail or silently degrade
        for a reason no traceback will explain.
        """
        verdict = _probe(IMPORT_ORDERS[name])
        assert verdict != "NO_ATTR", (
            "build.py no longer exposes _MISSING_MODULES. If the try/except ImportError "
            "guards were removed, delete this test; if they were renamed, update it. Do "
            "not leave it passing vacuously."
        )
        assert verdict == "MISSING=[]", (
            f"import order {name!r} left modules silently absent: {verdict}. "
            "A circular import was swallowed by build.py's `except ImportError`. The fix "
            "belongs in build.py (lazy collaborator imports) or in arag/index/__init__.py "
            "(stop eagerly importing build), NOT in this test."
        )


class TestOrchestratorVerifiedModules:
    """The v1 modules were verified by the orchestrator, not independently reviewed.

    Recorded as a test so the provenance gap is visible in the suite rather than only in
    a document. Three of four build agents were killed by a session API limit before their
    adversarial reviewers ran, so `docs/BUILD_RECORD.md` must keep saying so until a review
    actually happens.
    """

    def test_build_record_states_the_review_gap(self, repo_root) -> None:  # type: ignore[no-untyped-def]
        record = repo_root / "docs" / "BUILD_RECORD.md"
        assert record.exists(), "docs/BUILD_RECORD.md is missing"
        text = record.read_text(encoding="utf-8")
        assert "verified by the orchestrator, not independently reviewed" in text, (
            "the build record must state the review gap verbatim - 481 passing tests are "
            "not the same as an adversarial review, and the distinction is the whole "
            "reason the reviewer stage existed"
        )
