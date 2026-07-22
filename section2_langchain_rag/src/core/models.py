"""
Core domain models.

This module has ZERO dependencies on LangChain, LangGraph, Chroma, Gemini,
OpenAI, or any other external library. It is the center of the hexagonal
architecture: every other layer (ports, adapters, application) depends on
these types, but these types depend on nothing.

Keeping this dependency-free is what makes it possible to swap any adapter
(vector store, embedder, LLM provider) later without touching the domain
model, the graph logic, or the tests that exercise them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# --------------------------------------------------------------------------- #
# Ingestion-time models
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Document:
    """A single loaded source file, before chunking."""

    doc_id: str
    source_path: str
    source_type: str  # "markdown" | "pdf"
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A single indexable unit of text, produced by a Chunker from a Document."""

    chunk_id: str
    doc_id: str
    source_path: str
    text: str
    chunk_index: int  # position of this chunk within its source document
    section_path: Optional[str] = None  # e.g. "Shipping > International" (markdown headers)
    page_number: Optional[int] = None  # for PDFs
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Query-time models
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by the vector store for a given query, with its score."""

    chunk: Chunk
    score: float
    is_neighbor_expansion: bool = False  # True if pulled in for context, not by similarity


class GateStatus(str, Enum):
    """Outcome of the relevance gate that runs before generation."""

    PASS = "pass"
    EMPTY_INDEX = "empty_index"       # vector store has nothing indexed / nothing to search
    BELOW_THRESHOLD = "below_threshold"  # results exist but none are relevant enough
    RETRIEVAL_FAILED = "retrieval_failed"  # embedder/vector store call failed (system error,
                                            # not a content-relevance outcome -- logged/handled
                                            # differently, see NOTES.md "Fallback mechanisms")
    QUESTION_TOO_LONG = "question_too_long"  # rejected before any embedding/retrieval call


@dataclass(frozen=True)
class Citation:
    """A source attribution, grouped by source file (not one row per chunk)."""

    source_path: str
    chunk_ids: list[str]
    section_paths: list[str] = field(default_factory=list)


@dataclass
class QueryMetrics:
    """
    Everything we log/measure for a single query, covering both the retrieval
    (RAG) side and the generation (LLM) side. This is the same object used for
    the structured log file and for the verbose console summary.
    """

    # --- RAG / retrieval metrics ---
    retriever_latency_ms: float = 0.0
    num_chunks_retrieved: int = 0
    similarity_scores: list[float] = field(default_factory=list)
    retrieval_failed: bool = False
    quota_exceeded: bool = False
    no_context_event: Optional[str] = None  # None | "empty_index" | "below_threshold"
    neighbor_chunks_added: int = 0
    context_chunks_dropped_for_budget: int = 0
    # True when the provider's native structured output (response_schema)
    # did not return a valid parsed answer and the pipeline had to fall
    # back to parsing the raw text itself. Should be rare; previously this
    # failure mode was invisible (silently produced an uncited answer with
    # no signal anywhere) -- see CHANGELOG.md.
    structured_output_failed: bool = False

    # --- LLM / generation metrics ---
    model_used: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    context_length: int = 0  # characters of context sent to the model
    generation_latency_ms: float = 0.0
    generation_speed_tps: float = 0.0  # completion tokens / generation seconds
    estimated_cost_usd: float = 0.0
    generation_skipped: bool = False  # True when short-circuited by the gate

    # --- end-to-end ---
    total_latency_ms: float = 0.0
    provider: str = ""


@dataclass(frozen=True)
class AnswerResult:
    """The final output of the pipeline for a single question."""

    question: str
    answer: str
    citations: list[Citation]  # what the MODEL says it used -- depends on generation succeeding cleanly
    retrieved_sources: list[Citation]  # what retrieval ACTUALLY found and passed as context --
    # independent of the model; always populated whenever chunks were
    # retrieved, even if the model's own citation reporting fails. See
    # CHANGELOG.md for why both exist rather than just `citations`.
    gate_status: GateStatus
    retrieved_chunks: list[RetrievedChunk]
    metrics: QueryMetrics
