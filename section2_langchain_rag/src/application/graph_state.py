from __future__ import annotations

from typing import Optional, TypedDict

from src.core.models import Citation, GateStatus, QueryMetrics, RetrievedChunk


class RagState(TypedDict, total=False):
    """State threaded through the LangGraph graph for a single query."""

    question: str
    history: Optional[list[tuple[str, str]]]

    retrieved_chunks: list[RetrievedChunk]
    gate_status: GateStatus

    answer: str
    citations: list[Citation]
    retrieved_sources: list[Citation]

    metrics: QueryMetrics
