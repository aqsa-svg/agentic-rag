"""Find signals that are produced and never consumed.

A field populated correctly, covered by a test asserting it is populated correctly, and
read by no branch anywhere. Every test passes; the signal does nothing. Worse than dead
code, because the field's presence reads as evidence that the case is handled and so
suppresses the question "what happens when this is true?"

Three of these appeared in one session — ``LLMError.retryable``,
``LLMError.retry_after_s``, ``LLMResponse.truncated`` — and the third cost 5 of 7
questions in the first live generation run. See "A SECOND pattern" in
``docs/SILENT_WRONGNESS.md``.

Importable: ``unread_signals()`` returns the current set, which
``tests/test_signal_consumption.py`` ratchets. Runnable: ``python tools/audit_signals.py``
prints the full per-field table.

## Why AST rather than grep

``grep truncated`` matches the field's own assignment at the producing site, which is the
half that was never in doubt. This counts ``ast.Attribute`` nodes in a **Load** context,
outside the module that defines the field — an actual read by someone else.
"""

from __future__ import annotations

import ast
import pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "arag"
TESTS = ROOT / "tests"

# class -> (defining module, fields whose consumption we care about)
TARGETS: dict[str, tuple[str, list[str]]] = {
    "LLMResponse": (
        "src/arag/agent/llm.py",
        ["text", "model", "input_tokens", "output_tokens", "finish_reason", "truncated", "raw"],
    ),
    "LLMError": ("src/arag/agent/llm.py", ["kind", "retryable", "retry_after_s"]),
    "LLMRequest": (
        "src/arag/agent/llm.py",
        [
            "model",
            "system",
            "user",
            "temperature",
            "max_output_tokens",
            "thinking_budget",
            "response_mime_type",
        ],
    ),
    "AnswerResult": (
        "src/arag/agent/types.py",
        [
            "answer",
            "citations",
            "abstained",
            "abstain_reason",
            "confidence",
            "surfaced_conflict",
            "degraded",
            "cold_start",
            "tool_calls",
            "retrieved",
            "input_tokens",
            "output_tokens",
            "marginal_usd",
            "shadow_usd",
        ],
    ),
    "RetrievedChunk": (
        "src/arag/retrieval/types.py",
        ["dense_score", "lexical_score", "fused_score", "rerank_score", "confidence"],
    ),
    "ChunkMeta": (
        "src/arag/retrieval/types.py",
        [
            "insurer",
            "product",
            "section_path",
            "effective_from",
            "effective_to",
            "superseded_by",
            "is_table",
            "injection_suspected",
            "clause_ids",
        ],
    ),
}

ENUMS: dict[str, str] = {
    "AbstainReason": "src/arag/agent/types.py",
    "DegradedComponent": "src/arag/agent/types.py",
    "LLMErrorKind": "src/arag/agent/llm.py",
}


def _files() -> list[pathlib.Path]:
    return [
        p for root in (SRC, TESTS) for p in root.rglob("*.py") if "__pycache__" not in p.parts
    ]


def _rel(path: pathlib.Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _attribute_loads() -> dict[str, set[str]]:
    reads: dict[str, set[str]] = defaultdict(set)
    for path in _files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                reads[node.attr].add(_rel(path))
    return reads


def _enum_member_uses() -> dict[str, set[str]]:
    uses: dict[str, set[str]] = defaultdict(set)
    for path in _files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in ENUMS
            ):
                uses[f"{node.value.id}.{node.attr}"].add(_rel(path))
    return uses


def _enum_members(enum_name: str, home: str) -> list[str]:
    tree = ast.parse((ROOT / home).read_text(encoding="utf-8"))
    return [
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == enum_name
        for stmt in node.body
        if isinstance(stmt, ast.Assign)
        for target in stmt.targets
        if isinstance(target, ast.Name)
    ]


def audit() -> dict[str, tuple[list[str], list[str]]]:
    """signal -> (production readers, test-only readers). Empty production list = unread."""
    reads = _attribute_loads()
    uses = _enum_member_uses()
    out: dict[str, tuple[list[str], list[str]]] = {}

    for cls, (home, fields) in TARGETS.items():
        for field in fields:
            sites = sorted(reads.get(field, set()))
            external = [s for s in sites if s != home]
            out[f"{cls}.{field}"] = (
                [s for s in external if s.startswith("src/")],
                [s for s in external if s.startswith("tests/")],
            )

    for enum_name, home in ENUMS.items():
        for member in _enum_members(enum_name, home):
            sites = sorted(uses.get(f"{enum_name}.{member}", set()))
            out[f"{enum_name}.{member}"] = (
                [s for s in sites if s.startswith("src/") and s != home],
                [s for s in sites if s.startswith("tests/")],
            )
    return out


def unread_signals() -> set[str]:
    """Signals with no production consumer. What the ratchet test asserts against."""
    return {name for name, (prod, _tests) in audit().items() if not prod}


def main() -> None:
    results = audit()
    print("=" * 92)
    print("SIGNAL AUDIT - is the signal read by production code outside its defining module?")
    print("=" * 92)
    for name, (prod, tests) in sorted(results.items()):
        if prod:
            where = ", ".join(p.replace("src/arag/", "") for p in prod[:3])
            print(f"  {name:<44} READ        {where}")
        elif tests:
            print(f"  {name:<44} TESTS-ONLY  no production branch")
        else:
            print(f"  {name:<44} UNREAD")
    unread = sorted(unread_signals())
    print("=" * 92)
    print(f"{len(unread)} signal(s) with no production consumer:")
    for name in unread:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
