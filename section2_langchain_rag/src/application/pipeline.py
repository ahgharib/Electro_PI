"""RAGPipeline -- the single entrypoint the rest of the world talks to.

This is deliberately the ONLY class a host application (a CLI, a FastAPI
endpoint, a voice agent tool call) needs to import to use this system:

    pipeline = RAGPipeline()
    pipeline.ingest("./docs")
    result = pipeline.answer("What does RFC 2119 mean by MUST NOT?")

Everything else (ports, adapters, the graph, monitoring, reliability) is an
implementation detail behind this class.
"""

from __future__ import annotations

import logging
import time

from src.application.config import Config
from src.application.content_hash import compute_content_hash
from src.application.exceptions import QuestionTooLongError
from src.application.factory import build_chunker, build_embedder, build_generator, build_loader_registry, build_vector_store
from src.application.graph_builder import build_graph
from src.application.graph_nodes import GraphNodes
from src.application.graph_state import RagState
from src.application.monitoring import Monitor
from src.application.reliability import RateLimiter
from src.application.token_utils import approximate_token_count
from src.core.models import AnswerResult, GateStatus, QueryMetrics
from src.ports.chunker import Chunker
from src.ports.document_loader import DocumentLoader
from src.ports.embedder import Embedder
from src.ports.generator import Generator
from src.ports.vector_store import VectorStore

logger = logging.getLogger("rag.pipeline")


class RAGPipeline:
    def __init__(
        self,
        config: Config | None = None,
        *,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        generator: Generator | None = None,
        chunker: Chunker | None = None,
        loader_registry: DocumentLoader | None = None,
    ):
        """All dependencies can be injected directly (used by tests with
        fake adapters, and available to any host application that wants to
        assemble its own components). If omitted, each is built from
        Config via the factory -- the normal path for real usage."""
        self.config = config or Config()

        self.embedder = embedder or build_embedder(self.config)
        self.vector_store = vector_store or build_vector_store(self.config)
        self.generator = generator or build_generator(self.config)
        self.chunker = chunker or build_chunker(self.config)
        self.loader_registry = loader_registry or build_loader_registry()

        self.nodes = GraphNodes(self.embedder, self.vector_store, self.generator, self.config)
        self.graph = build_graph(self.nodes)

        self.monitor = Monitor(self.config.log_dir, self.config.verbose_console)
        self._embedding_rate_limiter = RateLimiter(self.config.embedding_rpm_limit)

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #

    def ingest(self, docs_dir: str | None = None, force: bool = False) -> int:
        """Loads, chunks, embeds, and upserts every supported file in
        docs_dir. Returns the number of chunks actually (re-)embedded.

        By default (force=False), chunks whose content hash already matches
        what's stored are skipped entirely -- no embedding API call is made
        for them. This is what makes repeated `ingest.py` runs during
        development cheap: re-running it after adding one new document, or
        just to check it still works, does not re-burn embedding quota on
        every unchanged chunk from every previous run. Pass force=True to
        bypass this and re-embed everything (e.g. after changing embedding
        model or chunking config, where the same chunk_id may now need a
        different embedding).
        """
        directory = docs_dir or self.config.docs_dir
        documents = self.loader_registry.load_directory(directory)

        total_embedded = 0
        total_skipped = 0
        batch_size = max(1, self.config.embedding_batch_size)

        for document in documents:
            chunks = self.chunker.chunk(document)
            if not chunks:
                logger.warning("no_chunks_produced doc=%s", document.source_path)
                continue

            to_embed = chunks
            skipped_in_doc = 0
            if not force and self.config.skip_unchanged_chunks_on_ingest:
                existing_hashes = self.vector_store.get_existing_content_hashes(
                    [c.chunk_id for c in chunks]
                )
                to_embed = [
                    c for c in chunks
                    if existing_hashes.get(c.chunk_id) != compute_content_hash(c.text)
                ]
                skipped_in_doc = len(chunks) - len(to_embed)
                total_skipped += skipped_in_doc

            for i in range(0, len(to_embed), batch_size):
                batch = to_embed[i : i + batch_size]
                self._embedding_rate_limiter.wait()
                embeddings = self.embedder.embed_documents([c.text for c in batch])
                self.vector_store.upsert(batch, embeddings)
                total_embedded += len(batch)

            if self.config.verbose_console:
                print(
                    f"[ingest] {document.source_path}: {len(to_embed)} embedded, "
                    f"{skipped_in_doc} unchanged (skipped)"
                )

        logger.info(
            "ingest_complete documents=%d embedded=%d skipped=%d",
            len(documents), total_embedded, total_skipped,
        )
        if self.config.verbose_console and total_skipped:
            print(
                f"[ingest] {total_skipped} chunk(s) were already indexed and unchanged -- "
                f"skipped re-embedding them. Use ingest(force=True) to re-embed everything."
            )
        return total_embedded

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #

    def answer(
        self,
        question: str,
        history: list[tuple[str, str]] | None = None,
        debug: bool | None = None,
    ) -> AnswerResult:
        """debug: opt-in full per-chunk diagnostic printout (chunk previews,
        per-chunk scores vs. threshold, whether trimming/neighbor-expansion
        happened). Defaults to Config.debug_retrieval (DEBUG_RETRIEVAL in
        .env) if not passed explicitly -- pass True/False here to override
        that for a single call regardless of the global setting."""
        show_debug = self.config.debug_retrieval if debug is None else debug
        start = time.monotonic()

        # Input token-limit guardrail: reject early and clearly rather than
        # silently truncating the user's question (see NOTES.md, "token
        # limit handling" -- truncation can silently change the meaning of
        # the question, which is exactly the wrong failure mode here).
        token_count = approximate_token_count(question)
        if token_count > self.config.question_token_limit:
            metrics = QueryMetrics(generation_skipped=True, no_context_event="question_too_long")
            metrics.total_latency_ms = (time.monotonic() - start) * 1000
            message = (
                f"Your question is {token_count} tokens long, which is over the "
                f"{self.config.question_token_limit} token limit. Please ask a shorter, "
                f"more specific question."
            )
            result = AnswerResult(
                question=question,
                answer=message,
                citations=[],
                retrieved_sources=[],
                gate_status=GateStatus.QUESTION_TOO_LONG,
                retrieved_chunks=[],
                metrics=metrics,
            )
            self.monitor.log_query(question, result.gate_status, metrics, result.answer)
            if show_debug:
                self.monitor.print_debug_retrieval(
                    question, [], result.gate_status, self.config.relevance_threshold, metrics
                )
            return result

        initial_state: RagState = {"question": question, "history": history}
        final_state = self.graph.invoke(initial_state)

        metrics: QueryMetrics = final_state["metrics"]
        metrics.total_latency_ms = (time.monotonic() - start) * 1000

        result = AnswerResult(
            question=question,
            answer=final_state["answer"],
            citations=final_state.get("citations", []),
            retrieved_sources=final_state.get("retrieved_sources", []),
            gate_status=final_state["gate_status"],
            retrieved_chunks=final_state.get("retrieved_chunks", []),
            metrics=metrics,
        )
        self.monitor.log_query(question, result.gate_status, metrics, result.answer)
        if show_debug:
            self.monitor.print_debug_retrieval(
                question, result.retrieved_chunks, result.gate_status,
                self.config.relevance_threshold, metrics,
            )
        return result

    def stats(self) -> dict:
        return self.monitor.get_stats()
