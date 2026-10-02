"""The transitive half of the online/offline bundle boundary.

``tests/test_import_boundaries.py`` checks *direct* imports, per file. It passed for the
whole life of the project while the boundary was broken:

    arag.retrieval.lexical            # online package
      imports arag.ingest.chunk_types # two dataclasses - a reasonable dependency
        -> executes arag/ingest/__init__.py     (importing a submodule runs its package)
          -> imports arag.ingest.probe
            -> imports pymupdf                  (~40MB native, offline-only)

Every module on that chain is innocent under a per-file check. The import *closure* is not.
DESIGN §9 splits offline (PyMuPDF, GPU) from online (CPU, serverless) precisely to keep
that library out of a function that never opens a PDF, and the split was not enforced.

Recorded as instance 7 in ``docs/SILENT_WRONGNESS.md``. It is the first instance whose
symptom is a production failure — a bundle-size limit or seconds of cold start, found in a
deploy — rather than a number that is quietly wrong, and the only one no test caught.

Two tests, deliberately overlapping:

- **AST closure** — fast, imports nothing, and names the exact edge that reintroduced the
  dependency. Good failure messages, but it models Python's import system rather than
  running it, so it has blind spots (``importlib`` calls, ``__getattr__``, plugin loaders).
- **Subprocess** — ground truth. Imports the module in a fresh interpreter and asks what
  actually landed in ``sys.modules``. No blind spots, but it can only say *that* something
  leaked, not where from.

The first tells you where to look; the second tells you whether you are wrong.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

BOUNDARIES = Path(__file__).with_name("test_import_boundaries.py")
SRC = Path(__file__).resolve().parents[1] / "src" / "arag"

# Duplicated from test_import_boundaries.py rather than imported: `tests/` has no
# `__init__.py`, so a cross-test import makes mypy resolve the same file under two module
# names and abort. `test_the_two_files_agree_on_the_policy` below pins them together, which
# is a better guarantee than a shared import anyway - it fails loudly on drift instead of
# silently widening one file's rule from the other.
OFFLINE_ONLY_ROOTS = {"torch", "sentence_transformers", "transformers", "fitz", "pymupdf"}
ONLINE_PACKAGES = {"api", "agent", "retrieval", "guardrails"}

# Not in ONLINE_PACKAGES, because the package as a whole is offline. But `index.lexical` is
# the scorer a serverless function loads to answer a query, so it carries the same rule.
EXTRA_ONLINE_MODULES = ("arag.index.lexical",)


def _modules() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def _package_of(path: Path) -> str:
    rel = path.relative_to(SRC)
    return rel.parts[0] if len(rel.parts) > 1 else ""


def _runtime_imports(path: Path) -> list[tuple[str, int]]:
    """Module-scope imports that actually execute, with line numbers.

    Excludes two things ``test_import_boundaries._imports`` includes, because both are free
    at runtime and flagging them would make this test unusable:

    - ``if TYPE_CHECKING:`` bodies — erased at runtime, so they cost no bundle bytes. (The
      LangGraph containment test still counts them, and is right to: a type-only dependency
      on a framework still means the framework leaked into the signatures.)
    - imports inside functions — the deliberate laziness this project uses on purpose, e.g.
      ``arag.eval.cli._build_lazy_engine``, which exists to keep exactly this boundary.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[tuple[str, int]] = []

    def walk(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.Import):
                out.extend((alias.name, node.lineno) for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                out.append((node.module, node.lineno))
            elif isinstance(node, ast.If):
                # A TYPE_CHECKING guard's body never runs; its `else` does.
                if "TYPE_CHECKING" in ast.dump(node.test):
                    walk(node.orelse)
                else:
                    walk(node.body)
                    walk(node.orelse)
            elif isinstance(node, ast.Try):
                walk(node.body)
                for handler in node.handlers:
                    walk(handler.body)
                walk(node.orelse)
                walk(node.finalbody)
            elif isinstance(node, ast.ClassDef | ast.With):
                walk(node.body)
            # FunctionDef / AsyncFunctionDef bodies are skipped: see docstring.

    walk(tree.body)
    return out


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(["arag", *parts])


def _graph() -> tuple[dict[str, set[str]], dict[str, set[tuple[str, int]]]]:
    """Intra-``arag`` edges, plus the offline roots each module imports directly.

    The edge that matters is the implicit one: importing ``arag.a.b`` also executes
    ``arag.a``. Modelling only the explicit target is exactly how the direct-import test
    missed this, so every ancestor package is added as an edge in its own right.
    """
    edges: dict[str, set[str]] = {}
    offline: dict[str, set[tuple[str, int]]] = {}
    known = {_module_name(m) for m in _modules()}

    for module in _modules():
        name = _module_name(module)
        edges.setdefault(name, set())
        offline.setdefault(name, set())
        for target, lineno in _runtime_imports(module):
            root = target.split(".")[0]
            if root in OFFLINE_ONLY_ROOTS:
                offline[name].add((root, lineno))
                continue
            if root != "arag":
                continue
            bits = target.split(".")
            for i in range(2, len(bits) + 1):
                candidate = ".".join(bits[:i])
                if candidate in known and candidate != name:
                    edges[name].add(candidate)
    return edges, offline


def _closure(start: str, edges: dict[str, set[str]]) -> dict[str, list[str]]:
    """Reachable modules, each mapped to the shortest path taken to reach it."""
    paths = {start: [start]}
    queue = [start]
    while queue:
        current = queue.pop(0)
        for nxt in sorted(edges.get(current, ())):
            if nxt not in paths:
                paths[nxt] = [*paths[current], nxt]
                queue.append(nxt)
    return paths


class TestAstClosure:
    def test_request_path_closure_never_reaches_an_offline_only_dependency(self) -> None:
        edges, offline = _graph()
        starts = [_module_name(m) for m in _modules() if _package_of(m) in ONLINE_PACKAGES] + list(
            EXTRA_ONLINE_MODULES
        )

        violations: list[str] = []
        for name in sorted(set(starts)):
            for reached, chain in sorted(_closure(name, edges).items()):
                for root, lineno in sorted(offline.get(reached, ())):
                    violations.append(
                        f"{name} reaches {root} via {' -> '.join(chain)} [{reached}:{lineno}]"
                    )
        assert not violations, (
            "an online package's import closure reaches an offline-only dependency. The fix "
            "is a lazy re-export (PEP 562 `__getattr__`) in the package `__init__` that "
            "pulls it in, NOT a wider allowlist here:\n  " + "\n  ".join(violations)
        )

    def test_the_two_files_agree_on_the_policy(self) -> None:
        """The direct and transitive checks must police the same dependency list.

        Otherwise the weaker check could be widened on its own and this one would keep
        passing while enforcing a rule nobody stated.
        """
        source = BOUNDARIES.read_text(encoding="utf-8")
        for name, value in (
            ("OFFLINE_ONLY_ROOTS", OFFLINE_ONLY_ROOTS),
            ("ONLINE_PACKAGES", ONLINE_PACKAGES),
        ):
            line = next(
                (line for line in source.splitlines() if line.startswith(f"{name} = ")), None
            )
            assert line is not None, f"{name} is no longer defined in {BOUNDARIES.name}"
            declared = ast.literal_eval(line.split("=", 1)[1].strip())
            assert set(declared) == value, (
                f"{name} differs between {BOUNDARIES.name} and this file: "
                f"{sorted(declared)} vs {sorted(value)}. Update both, or the transitive "
                "check stops enforcing the policy the direct check states."
            )

    def test_the_walker_actually_sees_the_parent_package_edge(self) -> None:
        """Guard against this test passing because it models nothing.

        If ``_graph`` stopped adding ancestor-package edges, the closure test would go green
        and the boundary would be unguarded again — the precise failure being fixed here.
        """
        edges, _ = _graph()
        assert "arag.ingest" in edges["arag.retrieval.lexical"], (
            "the graph no longer records that importing arag.ingest.chunk_types executes "
            "arag.ingest. Without that edge the closure test cannot catch instance 7."
        )


class TestActualInterpreter:
    @pytest.mark.parametrize(
        "module",
        [
            "arag.api.app",
            "arag.agent.retrieval_only",
            "arag.agent.stub",
            "arag.retrieval.lexical",
            "arag.retrieval.stub",
            "arag.index.lexical",
        ],
    )
    def test_importing_a_request_path_module_loads_no_offline_dependency(self, module: str) -> None:
        """Import it in a fresh interpreter and ask what ended up in ``sys.modules``.

        A subprocess because ``sys.modules`` is process-global: by the time this runs pytest
        has already imported most of the tree, so an in-process assertion would either pass
        trivially or fail for reasons unrelated to ``module``.
        """
        probe = (
            "import importlib, sys\n"
            f"importlib.import_module({module!r})\n"
            f"roots = {sorted(OFFLINE_ONLY_ROOTS)!r}\n"
            "bad = sorted({m.split('.')[0] for m in sys.modules if m.split('.')[0] in roots})\n"
            "print('LOADED=' + repr(bad))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, timeout=180
        )
        assert result.returncode == 0, f"importing {module} failed:\n{result.stderr[-1500:]}"
        verdict = result.stdout.strip().splitlines()[-1]
        assert verdict == "LOADED=[]", (
            f"importing {module} pulled an offline-only dependency into sys.modules: "
            f"{verdict}. Find the package `__init__` on the chain and make its re-exports "
            "lazy; do not relax this assertion."
        )
