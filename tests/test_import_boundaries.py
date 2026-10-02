"""Architecture tests: the import direction is enforced, not merely documented.

DESIGN §5.7 claims that "framework choice is reversible". A claim like that is worth
nothing unless something breaks when it stops being true. These tests walk the AST of
every module under ``src/arag`` and fail the build if a boundary is crossed, which turns
the architecture section of the README from an aspiration into a property.

These tests check **direct** imports, one file at a time. That is not sufficient on its
own and was not: the online path reached PyMuPDF transitively for the whole life of the
project while every test here was green (instance 7 in ``docs/SILENT_WRONGNESS.md``).
The transitive half lives in ``tests/test_import_closure.py`` and is mutation-tested.

Why AST rather than ``import-linter``: no extra dependency, it runs in the same pytest
invocation as everything else, and the failure message can name the exact file and line.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "arag"

# Packages that may import a heavy agent framework. Everything else must not.
AGENT_FRAMEWORK_ROOTS = {"langchain", "langgraph", "langchain_core", "langchain_community"}
AGENT_PACKAGE = "agent"

# Offline-only dependencies. These must never be reachable from the online request path,
# because they would blow the Vercel bundle and pull torch into a CPU-only function.
OFFLINE_ONLY_ROOTS = {"torch", "sentence_transformers", "transformers", "fitz", "pymupdf"}
ONLINE_PACKAGES = {"api", "agent", "retrieval", "guardrails"}


def _modules() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def _imports(path: Path) -> list[tuple[str, int]]:
    """Top-level module roots imported by a file, with line numbers.

    Imports inside ``if TYPE_CHECKING`` blocks are included deliberately: a type-only
    import of LangGraph still means the type has leaked across the boundary.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name.split(".")[0], node.lineno))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append((node.module.split(".")[0], node.lineno))
    return out


def _package_of(path: Path) -> str:
    rel = path.relative_to(SRC)
    return rel.parts[0] if len(rel.parts) > 1 else ""


class TestAgentFrameworkContainment:
    def test_only_the_agent_package_may_import_langgraph(self) -> None:
        violations: list[str] = []
        for module in _modules():
            package = _package_of(module)
            if package == AGENT_PACKAGE:
                continue
            for root, lineno in _imports(module):
                if root in AGENT_FRAMEWORK_ROOTS:
                    violations.append(
                        f"{module.relative_to(SRC)}:{lineno} imports {root} "
                        f"(package '{package or 'arag'}' is outside arag.agent)"
                    )
        assert not violations, (
            "LangGraph/LangChain has leaked outside arag.agent, so the framework is no "
            "longer replaceable and DESIGN §5.7 is no longer true:\n  " + "\n  ".join(violations)
        )

    def test_eval_never_imports_a_concrete_engine_or_retriever(self) -> None:
        """The harness must depend on protocols only.

        If ``arag.eval`` imported the real pipeline, the harness could not be run against
        a stub, and Phase 2 (harness before pipeline) would have been impossible.
        """
        forbidden = {"langchain", "langgraph", "psycopg", "onnxruntime", "torch"}
        violations: list[str] = []
        for module in _modules():
            if _package_of(module) != "eval":
                continue
            for root, lineno in _imports(module):
                if root in forbidden:
                    violations.append(f"{module.relative_to(SRC)}:{lineno} imports {root}")
        assert not violations, (
            "eval must depend on protocols, not implementations:\n  " + "\n  ".join(violations)
        )


class TestOnlineBundleHygiene:
    def test_request_path_never_imports_offline_only_dependencies(self) -> None:
        """torch and PyMuPDF belong to the offline pipeline.

        The online function runs CPU-only ONNX. An accidental import here would either
        blow the package-size limit or add seconds of cold start, and it would be found in
        a deploy rather than in a test.
        """
        violations: list[str] = []
        for module in _modules():
            package = _package_of(module)
            if package not in ONLINE_PACKAGES:
                continue
            for root, lineno in _imports(module):
                if root in OFFLINE_ONLY_ROOTS:
                    violations.append(
                        f"{module.relative_to(SRC)}:{lineno} imports {root} in online package "
                        f"'{package}'"
                    )
        assert not violations, "offline-only dependency on the online path:\n  " + "\n  ".join(
            violations
        )


class TestObsAndConfigAreLeaves:
    @pytest.mark.parametrize("package", ["obs", "config"])
    def test_do_not_import_pipeline_packages(self, package: str) -> None:
        """Observability and configuration are imported by everything, so if they imported
        back into the pipeline the result would be a circular-import maze the first time
        someone adds a metric.
        """
        pipeline = {"ingest", "index", "retrieval", "agent", "eval", "api"}
        violations: list[str] = []
        for module in _modules():
            rel = module.relative_to(SRC)
            if not (rel.parts[0] == package or rel.stem == package):
                continue
            for root, lineno in _imports(module):
                if root != "arag":
                    continue
                # Resolve `from arag.x import y` style by re-reading the full module path.
                text = module.read_text(encoding="utf-8").splitlines()[lineno - 1]
                for name in pipeline:
                    if f"arag.{name}" in text:
                        violations.append(f"{rel}:{lineno} -> arag.{name}")
        assert not violations, f"arag.{package} must be a leaf:\n  " + "\n  ".join(violations)


class TestEveryModuleParses:
    def test_no_syntax_errors(self) -> None:
        """Cheap smoke test: catches a broken file even if no test imports it."""
        for module in _modules():
            ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
