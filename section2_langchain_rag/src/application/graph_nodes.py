"""LangGraph node implementations.

Each node is a bound method on GraphNodes, taking/returning a partial
RagState dict (the LangGraph convention). GraphNodes holds the resolved
port instances (embedder, vector store, generator) and the Config -- it is
built once by the pipeline and its methods are registered as graph nodes.

Graph shape:

    retrieve -> gate --(PASS)---> generate -> format_citations -> END
                     `--(else)--> short_circuit -> END
"""

from __future__ import annotations

import json
import logging
import time

from src.application.config import Config
from src.application.exceptions import ComponentError, NoProviderAvailableError, QuotaExceededError
from src.application.graph_state import RagState
from src.core.models import Citation, GateStatus, QueryMetrics, RetrievedChunk
from src.ports.embedder import Embedder
from src.ports.generator import Generator
from src.ports.vector_store import VectorStore

logger = logging.getLogger("rag.graph")

SHORT_CIRCUIT_MESSAGES = {
    GateStatus.EMPTY_INDEX: (
        "The knowledge base appears to be empty -- no documents have been indexed yet. "
        "This looks like a setup issue rather than a question about your documents."
    ),
    GateStatus.BELOW_THRESHOLD: (
        "I don't have enough information in the provided documents to answer that question."
    ),
    GateStatus.RETRIEVAL_FAILED: (
        "I'm having trouble accessing the knowledge base right now. Please try again in a moment."
    ),
}

QUOTA_EXCEEDED_MESSAGE = (
    "The AI provider's usage quota has been exceeded (a rate limit or daily "
    "cap on the free tier). This isn't a bug in the request itself -- it "
    "usually clears within a minute (per-minute rate limit) or resets at "
    "midnight Pacific Time (daily quota). See README.md for ways to reduce "
    "API usage during development."
)


class GraphNodes:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, generator: Generator, config: Config):
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.config = config

    # ------------------------------------------------------------------ #
    # Node: retrieve
    # ------------------------------------------------------------------ #

    def retrieve(self, state: RagState) -> dict:
        question = state["question"]
        metrics = QueryMetrics()

        start = time.monotonic()
        retrieved: list[RetrievedChunk] = []
        try:
            query_embedding = self.embedder.embed_query(question)
            retrieved = self.vector_store.search(query_embedding, k=self.config.retrieval_k)
        except QuotaExceededError as exc:
            logger.error("retrieval_failed_quota_exceeded error=%s", exc)
            metrics.retrieval_failed = True
            metrics.quota_exceeded = True
        except (ComponentError, NoProviderAvailableError) as exc:
            logger.error("retrieval_failed error=%s", exc)
            metrics.retrieval_failed = True

        metrics.retriever_latency_ms = (time.monotonic() - start) * 1000
        metrics.num_chunks_retrieved = len(retrieved)
        metrics.similarity_scores = [rc.score for rc in retrieved]

        if self.config.enable_neighbor_expansion and retrieved:
            retrieved, added = self._expand_with_neighbors(retrieved)
            metrics.neighbor_chunks_added = added

        return {"retrieved_chunks": retrieved, "metrics": metrics}

    def _expand_with_neighbors(
        self, retrieved: list[RetrievedChunk]
    ) -> tuple[list[RetrievedChunk], int]:
        """Pull in the immediately adjacent chunk (same source, index +-1)
        for each retrieved chunk, if not already present. Neighbors are
        added as supporting context only -- they don't count toward the
        relevance gate or get auto-cited (see NOTES.md, "multi-doc /
        neighbor expansion")."""
        seen = {(rc.chunk.doc_id, rc.chunk.chunk_index) for rc in retrieved}
        expanded = list(retrieved)
        added = 0
        for rc in retrieved:
            for delta in (-1, 1):
                key = (rc.chunk.doc_id, rc.chunk.chunk_index + delta)
                if key in seen:
                    continue
                neighbor = self.vector_store.get_by_doc_and_index(key[0], key[1])
                if neighbor is not None:
                    expanded.append(RetrievedChunk(chunk=neighbor, score=0.0, is_neighbor_expansion=True))
                    seen.add(key)
                    added += 1
        return expanded, added

    # ------------------------------------------------------------------ #
    # Node: gate
    # ------------------------------------------------------------------ #

    def gate(self, state: RagState) -> dict:
        metrics = state["metrics"]

        if metrics.retrieval_failed:
            gate_status = GateStatus.RETRIEVAL_FAILED
        elif self.vector_store.count() == 0:
            gate_status = GateStatus.EMPTY_INDEX
        else:
            primary_scores = [
                rc.score for rc in state["retrieved_chunks"] if not rc.is_neighbor_expansion
            ]
            top_score = max(primary_scores, default=0.0)
            if not primary_scores or top_score < self.config.relevance_threshold:
                gate_status = GateStatus.BELOW_THRESHOLD
            else:
                gate_status = GateStatus.PASS

        if gate_status != GateStatus.PASS:
            metrics.no_context_event = gate_status.value

        return {"gate_status": gate_status, "metrics": metrics}

    # ------------------------------------------------------------------ #
    # Node: short_circuit (no LLM call -- guarantees no hallucination path)
    # ------------------------------------------------------------------ #

    def short_circuit(self, state: RagState) -> dict:
        metrics = state["metrics"]
        metrics.generation_skipped = True
        if metrics.quota_exceeded:
            answer = QUOTA_EXCEEDED_MESSAGE
        else:
            answer = SHORT_CIRCUIT_MESSAGES[state["gate_status"]]
        return {"answer": answer, "citations": [], "retrieved_sources": [], "metrics": metrics}

    # ------------------------------------------------------------------ #
    # Node: generate
    # ------------------------------------------------------------------ #

    def generate(self, state: RagState) -> dict:
        metrics = state["metrics"]
        question = state["question"]
        history = state.get("history")
        retrieved, dropped_count = self._trim_to_budget(state["retrieved_chunks"])
        metrics.context_chunks_dropped_for_budget = dropped_count

        try:
            result = self.generator.generate(question, retrieved, history)
        except QuotaExceededError as exc:
            logger.error("generation_failed_quota_exceeded error=%s", exc)
            metrics.generation_skipped = True
            metrics.quota_exceeded = True
            return {"answer": QUOTA_EXCEEDED_MESSAGE, "citations": [], "retrieved_sources": [], "metrics": metrics}
        except (ComponentError, NoProviderAvailableError) as exc:
            logger.error("generation_failed error=%s", exc)
            metrics.generation_skipped = True
            return {
                "answer": SHORT_CIRCUIT_MESSAGES[GateStatus.RETRIEVAL_FAILED],
                "citations": [],
                "retrieved_sources": [],
                "metrics": metrics,
            }

        # Prefer the provider's schema-validated structured output --
        # constrained server-side, not just requested via prompt. Only
        # fall back to parsing raw text ourselves when that's unavailable
        # (rare, but not impossible even with response_schema). This
        # fallback used to be silent; it's now recorded on metrics so it's
        # visible in monitoring/debug output rather than only a log line.
        if result.parsed_answer is not None:
            answer_text = str(result.parsed_answer.get("answer", "")).strip()
            cited_chunk_ids = list(result.parsed_answer.get("cited_chunk_ids", []))
        else:
            metrics.structured_output_failed = True
            answer_text, cited_chunk_ids = self._parse_generation_output(result.text)

        # `citations`: what the model says it used -- depends on the step
        # above succeeding cleanly.
        citations = self._build_citations(retrieved, cited_chunk_ids)
        # `retrieved_sources`: what retrieval actually found and handed to
        # the model as context, full stop -- computed straight from
        # `retrieved`, with no dependency on the model's output at all.
        # Grouped the same way, just over every chunk_id rather than only
        # the ones the model claims to have cited. This is what makes
        # sources visible even when structured_output_failed is True.
        retrieved_sources = self._build_citations(retrieved, [rc.chunk.chunk_id for rc in retrieved])

        provider = getattr(self.generator, "last_provider", self.config.generation_provider)
        model_used = getattr(self.generator, "last_model", self.generator.model_name)

        metrics.model_used = model_used
        metrics.provider = provider
        metrics.prompt_tokens = result.prompt_tokens
        metrics.completion_tokens = result.completion_tokens
        metrics.total_tokens = result.total_tokens
        metrics.context_length = sum(len(rc.chunk.text) for rc in retrieved)
        metrics.generation_latency_ms = result.latency_ms
        gen_seconds = result.latency_ms / 1000.0
        metrics.generation_speed_tps = (
            result.completion_tokens / gen_seconds if gen_seconds > 0 else 0.0
        )
        metrics.estimated_cost_usd = self.config.cost_for(
            model_used, result.prompt_tokens, result.completion_tokens
        )

        return {
            "answer": answer_text,
            "citations": citations,
            "retrieved_sources": retrieved_sources,
            "metrics": metrics,
        }

    def _trim_to_budget(self, retrieved: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], int]:
        """Keep the highest-priority chunks and drop the rest until the
        assembled context fits the configured token budget.

        Priority order: real retrieval hits by score (highest first), then
        neighbor-expansion chunks last (they support context but weren't
        themselves a relevance match). Chunks are dropped one at a time
        from the LOW-priority end of that ordering -- not a forward
        greedy-fill -- so the behavior is unambiguous: the kept set is
        always exactly "the N highest-priority chunks that fit," never a
        smaller/lower-priority chunk sneaking in past a larger one that was
        skipped. At least one chunk is always kept, even if it alone
        exceeds the budget, so a real answer is never starved down to
        nothing.

        Character-based approximation (~4 chars/token), not an exact
        tokenizer -- see src/application/token_utils.py for why.
        """
        if not retrieved:
            return [], 0

        budget_chars = self.config.context_token_budget * 4
        ordered = sorted(retrieved, key=lambda rc: (rc.is_neighbor_expansion, -rc.score))

        kept = list(ordered)
        total_chars = sum(len(rc.chunk.text) for rc in kept)
        dropped = 0
        while total_chars > budget_chars and len(kept) > 1:
            removed = kept.pop()  # lowest-priority remaining chunk
            total_chars -= len(removed.chunk.text)
            dropped += 1

        if dropped:
            logger.info(
                "context_trimmed_for_budget dropped=%d kept=%d budget_chars=%d",
                dropped, len(kept), budget_chars,
            )
        return kept, dropped

    @staticmethod
    def _parse_generation_output(raw_text: str) -> tuple[str, list[str]]:
        # Defense in depth: raw_text should always be a plain string by the
        # time it reaches here (every Generator adapter normalizes its
        # provider's response via response_text.extract_response_text
        # before returning). This guard means a malformed/unexpected value
        # degrades to a clean empty-ish answer instead of crashing the
        # graph.
        if not isinstance(raw_text, str):
            logger.warning("generation_output_not_a_string type=%s", type(raw_text).__name__)
            raw_text = str(raw_text) if raw_text is not None else ""

        try:
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:]
            payload = json.loads(cleaned)
            return payload.get("answer", "").strip(), list(payload.get("cited_chunk_ids", []))
        except json.JSONDecodeError:
            logger.warning("generation_output_not_json falling back to raw text")
            return raw_text.strip(), []

    @staticmethod
    def _build_citations(retrieved: list[RetrievedChunk], cited_chunk_ids: list[str]) -> list[Citation]:
        by_id = {rc.chunk.chunk_id: rc.chunk for rc in retrieved}
        valid_ids = [cid for cid in cited_chunk_ids if cid in by_id]

        grouped: dict[str, list[str]] = {}
        sections: dict[str, list[str]] = {}
        for cid in valid_ids:
            chunk = by_id[cid]
            grouped.setdefault(chunk.source_path, []).append(cid)
            if chunk.section_path:
                sections.setdefault(chunk.source_path, [])
                if chunk.section_path not in sections[chunk.source_path]:
                    sections[chunk.source_path].append(chunk.section_path)

        return [
            Citation(source_path=source, chunk_ids=ids, section_paths=sections.get(source, []))
            for source, ids in grouped.items()
        ]
