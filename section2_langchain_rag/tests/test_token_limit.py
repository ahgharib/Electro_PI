from __future__ import annotations

from src.application.pipeline import RAGPipeline
from src.core.models import GateStatus


class TestQuestionTokenLimit:
    def test_short_question_is_not_rejected(
        self, test_config, indexed_vector_store, fake_embedder, fake_generator, chunker, loader_registry
    ):
        test_config.question_token_limit = 300
        pipeline = RAGPipeline(
            config=test_config,
            embedder=fake_embedder,
            vector_store=indexed_vector_store,
            generator=fake_generator,
            chunker=chunker,
            loader_registry=loader_registry,
        )
        result = pipeline.answer("What is the return window?")
        assert result.gate_status != GateStatus.QUESTION_TOO_LONG

    def test_long_question_is_rejected_before_embedding(
        self, test_config, indexed_vector_store, fake_embedder, fake_generator, chunker, loader_registry
    ):
        test_config.question_token_limit = 5  # deliberately tiny
        pipeline = RAGPipeline(
            config=test_config,
            embedder=fake_embedder,
            vector_store=indexed_vector_store,
            generator=fake_generator,
            chunker=chunker,
            loader_registry=loader_registry,
        )
        long_question = "What is the return window for products bought during a holiday sale event?"
        result = pipeline.answer(long_question)

        assert result.gate_status == GateStatus.QUESTION_TOO_LONG
        assert "token limit" in result.answer
        assert result.metrics.num_chunks_retrieved == 0  # never reached retrieval
