from __future__ import annotations

from src.application.pipeline import RAGPipeline
from src.core.models import GateStatus


def _build_pipeline(test_config, embedder, vector_store, generator, chunker, loader_registry):
    return RAGPipeline(
        config=test_config,
        embedder=embedder,
        vector_store=vector_store,
        generator=generator,
        chunker=chunker,
        loader_registry=loader_registry,
    )


class TestIngestion:
    def test_ingest_indexes_all_sample_docs(
        self, test_config, vector_store, fake_embedder, fake_generator, chunker, loader_registry
    ):
        pipeline = _build_pipeline(test_config, fake_embedder, vector_store, fake_generator, chunker, loader_registry)
        total = pipeline.ingest(docs_dir=str(__import__("pathlib").Path(__file__).resolve().parent.parent / "docs"))
        assert total > 0
        assert pipeline.vector_store.count() == total


class TestEndToEndAnswer:
    def test_answerable_question_returns_grounded_answer_with_citations(
        self, test_config, indexed_vector_store, fake_embedder, fake_generator, chunker, loader_registry
    ):
        pipeline = _build_pipeline(
            test_config, fake_embedder, indexed_vector_store, fake_generator, chunker, loader_registry
        )
        result = pipeline.answer("What does the RFC 2119 keyword MUST NOT mean?")

        assert result.gate_status == GateStatus.PASS
        assert result.answer
        assert result.citations  # fake generator always cites the top chunk when context is passed
        assert result.metrics.total_tokens > 0
        assert result.metrics.total_latency_ms >= 0

    def test_empty_index_never_calls_generator(
        self, test_config, vector_store, fake_embedder, fake_generator, chunker, loader_registry
    ):
        pipeline = _build_pipeline(test_config, fake_embedder, vector_store, fake_generator, chunker, loader_registry)
        result = pipeline.answer("Anything at all?")

        assert result.gate_status == GateStatus.EMPTY_INDEX
        assert result.metrics.generation_skipped is True
        assert result.citations == []

    def test_stats_reflect_queries_made(
        self, test_config, indexed_vector_store, fake_embedder, fake_generator, chunker, loader_registry
    ):
        pipeline = _build_pipeline(
            test_config, fake_embedder, indexed_vector_store, fake_generator, chunker, loader_registry
        )
        pipeline.answer("What does MUST NOT mean under RFC 2119?")
        pipeline.answer("Which GDPR article covers the right to erasure?")

        stats = pipeline.stats()
        assert stats["query_count"] == 2
