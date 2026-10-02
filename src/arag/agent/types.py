"""The answer contract.

``AnswerResult`` is the single object the API returns, the eval harness scores, and the
observability layer logs. It is defined in this package — not in ``api`` — because the
route handler must be a translation layer over it and nothing more.

Two design decisions worth defending:

1. **Abstaining is a first-class outcome, not an error.** ``abstained=True`` with a typed
   ``AbstainReason`` means the harness can distinguish "refused because retrieval was
   weak" (correct, F7) from "refused because the model was rate limited" (a degradation,
   F4). Collapsing both into an error response would make the abstain-calibration metric
   unmeasurable.
2. **``degraded`` is a list, not a boolean.** A response served without the dense
   retriever is still a useful response, but a reviewer needs to know which leg was
   missing when they read the latency and quality numbers for that request.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from arag.retrieval.types import DocSpan, RetrievedChunk


class AbstainReason(StrEnum):
    LOW_CONFIDENCE = "low_confidence"  # F7
    NO_CANDIDATES = "no_candidates"  # empty corpus / filtered to nothing
    ONLY_SUPERSEDED_EVIDENCE = "only_superseded_evidence"  # F11
    CONTRADICTORY_UNRESOLVABLE = "contradictory_unresolvable"
    GENERATION_UNAVAILABLE = "generation_unavailable"  # F4 - quota or retries exhausted
    # Split out after two runs reported "7 abstained (generation_unavailable)" for two
    # completely different causes: once an un-prefixed env var meaning ZERO network calls
    # were made, once a suspended Google project. The results tables were identical. Three
    # causes, three consoles, three fixes - one statistic is not decomposable into them.
    GENERATION_NOT_CONFIGURED = "generation_not_configured"  # no credential: fix the env
    GENERATION_FORBIDDEN = "generation_forbidden"  # suspended/denied: fix the account
    GENERATION_BLOCKED = "generation_blocked"  # F6 safety block
    MALFORMED_GENERATION = "malformed_generation"  # F5 after repair failed
    SERVICE_DEGRADED = "service_degraded"  # F2
    NOT_IMPLEMENTED = "not_implemented"  # stub engines


class DegradedComponent(StrEnum):
    DENSE_RETRIEVAL = "dense_retrieval"
    LEXICAL_RETRIEVAL = "lexical_retrieval"
    RERANKER = "reranker"
    GENERATION = "generation"
    CHECKPOINTER = "checkpointer"


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    span: DocSpan
    chunk_id: str
    quote: str | None = None

    def key(self) -> str:
        return self.span.key()


class ToolCall(BaseModel):
    """One tool invocation. The sequence is the agent's audit trail: the multi-hop eval
    strata assert on *which* tools ran in *what order*, because an agent that reaches the
    right answer by luck through a single search call has not demonstrated planning.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    arguments: dict[str, object] = Field(default_factory=dict)
    latency_ms: float | None = None
    error: str | None = None
    result_count: int | None = None


class StageLatency(BaseModel):
    model_config = ConfigDict(frozen=True)

    embed_ms: float | None = None
    retrieve_ms: float | None = None
    rerank_ms: float | None = None
    generate_ms: float | None = None
    total_ms: float = 0.0


class AnswerResult(BaseModel):
    """What the system produces for one question."""

    question: str
    answer: str | None = None
    citations: tuple[Citation, ...] = ()

    abstained: bool = False
    abstain_reason: AbstainReason | None = None
    # Max retrieval confidence seen. Drives threshold calibration in v4.
    confidence: float = 0.0

    surfaced_conflict: bool = False
    degraded: tuple[DegradedComponent, ...] = ()
    cold_start: bool = False

    tool_calls: tuple[ToolCall, ...] = ()
    retrieved: tuple[RetrievedChunk, ...] = ()

    latency: StageLatency = Field(default_factory=StageLatency)
    input_tokens: int = 0
    output_tokens: int = 0
    marginal_usd: float = 0.0
    shadow_usd: float = 0.0

    trace_id: str | None = None
    engine: str = "unknown"
    as_of: date | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now().astimezone())

    def cited_keys(self) -> set[str]:
        return {c.key() for c in self.citations}

    def cited_documents(self) -> set[str]:
        return {c.span.document_id for c in self.citations}
