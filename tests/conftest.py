from __future__ import annotations

from pathlib import Path

import pytest

from arag.retrieval.types import DocSpan, RetrievedChunk

REPO_ROOT = Path(__file__).resolve().parents[1]


def chunk(
    doc: str,
    clause: str | None = None,
    page: int | None = None,
    cid: str | None = None,
) -> RetrievedChunk:
    """Build a retrieved chunk with only the fields the metrics look at.

    A helper rather than fixtures because the metric tests need *specific* orderings, and
    a fixture that returns a canned list makes it hard to see at the assertion site what
    ranking is being scored.
    """
    return RetrievedChunk(
        chunk_id=cid or f"{doc}:{clause or page}",
        span=DocSpan(document_id=doc, clause_id=clause, page=page),
        text="",
    )


def truth(doc: str, clause: str | None = None, page: int | None = None) -> DocSpan:
    return DocSpan(document_id=doc, clause_id=clause, page=page)


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def thresholds_path(repo_root: Path) -> Path:
    return repo_root / "src" / "arag" / "eval" / "thresholds.yaml"


@pytest.fixture
def golden_path(repo_root: Path) -> Path:
    """The seed fixture, NOT the golden set.

    data/golden/ deliberately holds no labelled data yet - only TEMPLATE.jsonl. Tests must
    never depend on the real golden set existing, because it does not, and a test that
    silently passed against an absent file would be the exact self-deception the harness
    exists to prevent.
    """
    return repo_root / "tests" / "fixtures" / "golden_seed.jsonl"
