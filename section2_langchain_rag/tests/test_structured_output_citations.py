from __future__ import annotations

from src.adapters.generators.fake_generator import FakeGenerator
from src.application.graph_builder import build_graph
from src.application.graph_nodes import GraphNodes
from src.core.models import GateStatus


class TestPrefersStructuredOutput:
    def test_uses_parsed_answer_directly_without_text_parsing(
        self, test_config, indexed_vector_store, fake_embedder
    ):
        generator = FakeGenerator()  # default: populates parsed_answer
        nodes = GraphNodes(fake_embedder, indexed_vector_store, generator, test_config)
        graph = build_graph(nodes)

        result = graph.invoke({"question": "What is the return window?", "history": None})

        assert result["gate_status"] == GateStatus.PASS
        assert result["metrics"].structured_output_failed is False
        assert result["citations"]  # structured path populated citations correctly

    def test_falls_back_to_text_parsing_when_structured_output_fails(
        self, test_config, indexed_vector_store, fake_embedder
    ):
        generator = FakeGenerator(simulate_structured_failure=True)
        nodes = GraphNodes(fake_embedder, indexed_vector_store, generator, test_config)
        graph = build_graph(nodes)

        result = graph.invoke({"question": "What is the return window?", "history": None})

        assert result["gate_status"] == GateStatus.PASS
        assert result["metrics"].structured_output_failed is True
        # The fallback text doesn't parse as JSON either (by design of the
        # simulated failure), so the model-cited citations end up empty --
        # this is the exact scenario that used to fail silently.
        assert result["citations"] == []
        assert result["answer"]  # an answer is still returned, just uncited


class TestRetrievedSourcesIndependentOfModel:
    def test_retrieved_sources_populated_even_when_model_cites_nothing(
        self, test_config, indexed_vector_store, fake_embedder
    ):
        generator = FakeGenerator(simulate_structured_failure=True)
        nodes = GraphNodes(fake_embedder, indexed_vector_store, generator, test_config)
        graph = build_graph(nodes)

        result = graph.invoke({"question": "What is the return window?", "history": None})

        # citations (model-asserted) are empty due to the simulated failure,
        # but retrieved_sources reflects what was actually retrieved and
        # handed to the model as context -- entirely independent of it.
        assert result["citations"] == []
        assert result["retrieved_sources"] != []

    def test_retrieved_sources_covers_every_retrieved_chunk_not_just_cited_ones(
        self, test_config, indexed_vector_store, fake_embedder
    ):
        # Default FakeGenerator only cites the FIRST retrieved chunk, but
        # multiple chunks are typically retrieved (k=4 by default).
        generator = FakeGenerator()
        nodes = GraphNodes(fake_embedder, indexed_vector_store, generator, test_config)
        graph = build_graph(nodes)

        result = graph.invoke({"question": "What is the return window?", "history": None})

        cited_chunk_count = sum(len(c.chunk_ids) for c in result["citations"])
        retrieved_chunk_count = sum(len(c.chunk_ids) for c in result["retrieved_sources"])
        assert retrieved_chunk_count >= cited_chunk_count

    def test_short_circuit_paths_have_empty_retrieved_sources(
        self, test_config, vector_store, fake_embedder, fake_generator
    ):
        # Empty index -> short_circuit -> both citation lists empty, no crash
        nodes = GraphNodes(fake_embedder, vector_store, fake_generator, test_config)
        graph = build_graph(nodes)

        result = graph.invoke({"question": "anything", "history": None})

        assert result["gate_status"] == GateStatus.EMPTY_INDEX
        assert result["citations"] == []
        assert result["retrieved_sources"] == []


class TestEndToEndThroughPipeline:
    def test_answer_result_exposes_both_citation_lists(
        self, test_config, indexed_vector_store, fake_embedder, chunker, loader_registry
    ):
        from src.application.pipeline import RAGPipeline

        pipeline = RAGPipeline(
            config=test_config,
            embedder=fake_embedder,
            vector_store=indexed_vector_store,
            generator=FakeGenerator(simulate_structured_failure=True),
            chunker=chunker,
            loader_registry=loader_registry,
        )
        result = pipeline.answer("What is the return window?")

        assert result.citations == []
        assert result.retrieved_sources != []
        assert result.metrics.structured_output_failed is True
