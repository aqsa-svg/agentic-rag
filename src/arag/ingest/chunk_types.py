"""The chunk record — the unit that gets indexed, retrieved, and cited.

Defined in its own module, separate from both the prose chunker and the table chunker,
because three things must agree on it exactly: what ingest produces, what the index
stores, and what ``SpanMatcher`` scores against. A type owned by one producer would drift
the moment the second producer needed a field.

Two design decisions worth defending:

**``span`` is the citable location, and it is not optional.** Every chunk carries a
``DocSpan`` because a chunk that cannot be cited cannot be scored — it would be present in
the index, retrievable, and invisible to every retrieval metric. The golden set keys on
spans, so a chunk without one is unmeasurable by construction.

**``kind`` records what produced the chunk.** Prose, a table serialisation, or a
definition. Not decoration: the T3 experiment compares two table serialisations against
each other, and the ingest coverage report needs to state how much of the corpus arrived
by which path. A single undifferentiated text field would make both impossible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from arag.retrieval.types import ChunkMeta, DocSpan


class ChunkKind(StrEnum):
    PROSE = "prose"
    DEFINITION = "definition"
    TABLE_MARKDOWN = "table_markdown"
    TABLE_ROW_NL = "table_row_nl"

    @property
    def is_table(self) -> bool:
        return self in (ChunkKind.TABLE_MARKDOWN, ChunkKind.TABLE_ROW_NL)


@dataclass(frozen=True)
class Chunk:
    """One indexed unit of the corpus."""

    chunk_id: str
    source_id: str
    kind: ChunkKind
    text: str
    span: DocSpan
    # Pages the chunk's content came from. Usually one; more when a table was stitched
    # across a page break, which is why this is a tuple rather than an int.
    pages: tuple[int, ...]
    meta: ChunkMeta = field(default_factory=ChunkMeta)
    # Free-form provenance: which heading strategy matched, which table it came from,
    # whether it is a fallback over a refused table region. Kept out of ChunkMeta because
    # ChunkMeta is the retrieval-filterable surface and this is diagnostic.
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError(f"chunk {self.chunk_id} has no text")
        if not self.pages:
            raise ValueError(f"chunk {self.chunk_id} has no pages")

    @property
    def char_len(self) -> int:
        return len(self.text)

    @property
    def clause_id(self) -> str | None:
        return self.span.clause_id
