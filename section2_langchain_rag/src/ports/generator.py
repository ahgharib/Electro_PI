"""Port: Generator.

Wraps whichever chat/completion model answers the question given retrieved
context. Concrete adapters (Gemini, OpenAI, a deterministic fake for tests)
all return the same GenerationResult shape so the pipeline's metrics and
citation-formatting logic never change when the provider does.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.core.models import RetrievedChunk


@dataclass(frozen=True)
class GenerationResult:
    text: str  # raw model output text (fallback if parsed_answer is None; see below)
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    # Populated when the provider's native structured-output mode
    # (response_schema / json_schema, constrained server-side, not just
    # requested via prompt) successfully returned and validated a
    # {"answer": str, "cited_chunk_ids": [...]} object. When this is None,
    # the caller falls back to parsing `text` as JSON itself (the old,
    # weaker path -- kept as a fallback, not removed, since even
    # schema-constrained decoding can occasionally fail to return).
    parsed_answer: dict | None = field(default=None)


class Generator(ABC):
    """Answers a question given retrieved chunks as context."""

    @abstractmethod
    def generate(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[tuple[str, str]] | None = None,
    ) -> GenerationResult:
        """Generate an answer.

        history: optional list of (question, answer) pairs from prior turns.
        This is a pass-through parameter only -- the pipeline does not store
        or manage conversation state itself (see NOTES.md, "Memory"). A host
        application (chat UI, voice agent) may maintain history and pass it
        in; if omitted, each call is fully independent, matching the task's
        single-turn Q&A scope.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def context_window_tokens(self) -> int:
        """Max input tokens this model supports -- used for context-trimming."""
        raise NotImplementedError
