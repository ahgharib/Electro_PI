from __future__ import annotations

from src.application.graph_builder import build_graph
from src.application.graph_nodes import GraphNodes
from src.core.models import GateStatus


class TestGateLogic:
    def test_empty_index_short_circuits_without_calling_generator(
        self, test_config, vector_store, fake_embedder, fake_generator
    ):
        # vector_store fixture is NOT pre-loaded -- index is empty
        nodes = GraphNodes(fake_embedder, vector_store, fake_generator, test_config)
        graph = build_graph(nodes)

        result = graph.invoke({"question": "What is the return window?", "history": None})

        assert result["gate_status"] == GateStatus.EMPTY_INDEX
        assert result["metrics"].generation_skipped is True
        assert result["citations"] == []

    def test_pass_when_relevant_chunks_found(
        self, test_config, indexed_vector_store, fake_embedder, fake_generator
    ):
        nodes = GraphNodes(fake_embedder, indexed_vector_store, fake_generator, test_config)
        graph = build_graph(nodes)

        result = graph.invoke({"question": "What is the return window for a refund?", "history": None})

        assert result["gate_status"] == GateStatus.PASS
        assert result["metrics"].generation_skipped is False
        assert result["metrics"].num_chunks_retrieved > 0

    def test_below_threshold_short_circuits_without_calling_generator(
        self, test_config, indexed_vector_store, fake_embedder, fake_generator
    ):
        # Set an unreachably high threshold so every real query fails the gate.
        test_config.relevance_threshold = 1.5
        nodes = GraphNodes(fake_embedder, indexed_vector_store, fake_generator, test_config)
        graph = build_graph(nodes)

        result = graph.invoke({"question": "What is the return window?", "history": None})

        assert result["gate_status"] == GateStatus.BELOW_THRESHOLD
        assert result["metrics"].generation_skipped is True
        assert "don't have enough information" in result["answer"]

    def test_gate_records_no_context_event(
        self, test_config, indexed_vector_store, fake_embedder, fake_generator
    ):
        test_config.relevance_threshold = 1.5
        nodes = GraphNodes(fake_embedder, indexed_vector_store, fake_generator, test_config)
        graph = build_graph(nodes)

        result = graph.invoke({"question": "irrelevant", "history": None})
        assert result["metrics"].no_context_event == "below_threshold"

    def test_quota_exceeded_during_retrieval_produces_a_distinct_clear_message(
        self, test_config, indexed_vector_store, fake_generator
    ):
        from src.application.exceptions import QuotaExceededError

        class _QuotaExceededEmbedder:
            def embed_query(self, text):
                raise QuotaExceededError("embedder", "gemini", original=RuntimeError("429 quota exceeded"))

            def embed_documents(self, texts):
                raise QuotaExceededError("embedder", "gemini", original=RuntimeError("429 quota exceeded"))

            @property
            def model_name(self):
                return "quota-exceeded-fake"

        nodes = GraphNodes(_QuotaExceededEmbedder(), indexed_vector_store, fake_generator, test_config)
        graph = build_graph(nodes)

        result = graph.invoke({"question": "What is the return window?", "history": None})

        assert result["metrics"].quota_exceeded is True
        assert result["metrics"].generation_skipped is True
        assert "quota" in result["answer"].lower()
        # not the generic "trouble accessing" message -- a distinct one
        assert "please try again in a moment" not in result["answer"].lower()
