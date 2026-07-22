from __future__ import annotations

from src.application.graph_nodes import GraphNodes
from src.core.models import Chunk, RetrievedChunk


def _rc(chunk_id, text, score=0.5, is_neighbor=False):
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id="d",
            source_path="d.md",
            text=text,
            chunk_index=0,
        ),
        score=score,
        is_neighbor_expansion=is_neighbor,
    )


class TestTrimToBudget:
    def test_keeps_everything_when_under_budget(self, test_config, fake_embedder, fake_generator):
        test_config.context_token_budget = 10_000  # generous
        nodes = GraphNodes(fake_embedder, None, fake_generator, test_config)
        retrieved = [_rc("a", "short text", score=0.9), _rc("b", "also short", score=0.5)]

        kept, dropped = nodes._trim_to_budget(retrieved)

        assert dropped == 0
        assert len(kept) == 2

    def test_drops_lowest_scored_chunks_first(self, test_config, fake_embedder, fake_generator):
        # Each chunk is ~400 chars (~100 tokens). Budget of 150 tokens
        # (~600 chars) should fit only 1.
        test_config.context_token_budget = 150
        nodes = GraphNodes(fake_embedder, None, fake_generator, test_config)
        big_text = "word " * 100  # ~500 chars
        retrieved = [
            _rc("low", big_text, score=0.2),
            _rc("high", big_text, score=0.9),
            _rc("mid", big_text, score=0.5),
        ]

        kept, dropped = nodes._trim_to_budget(retrieved)

        kept_ids = {rc.chunk.chunk_id for rc in kept}
        assert "high" in kept_ids  # highest score always survives
        assert "low" not in kept_ids  # lowest score dropped first
        assert dropped == 2

    def test_always_keeps_at_least_one_chunk_even_if_it_exceeds_budget(
        self, test_config, fake_embedder, fake_generator
    ):
        test_config.context_token_budget = 1  # impossibly tiny
        nodes = GraphNodes(fake_embedder, None, fake_generator, test_config)
        retrieved = [_rc("only", "some reasonably long chunk of text " * 20, score=0.9)]

        kept, dropped = nodes._trim_to_budget(retrieved)

        assert len(kept) == 1
        assert dropped == 0

    def test_neighbor_expansion_chunks_are_deprioritized_below_real_hits(
        self, test_config, fake_embedder, fake_generator
    ):
        test_config.context_token_budget = 150
        nodes = GraphNodes(fake_embedder, None, fake_generator, test_config)
        big_text = "word " * 100
        retrieved = [
            _rc("neighbor", big_text, score=0.0, is_neighbor=True),
            _rc("real_hit_low_score", big_text, score=0.1),
        ]

        kept, dropped = nodes._trim_to_budget(retrieved)

        # A real (if weak) retrieval hit outranks a neighbor-expansion chunk.
        kept_ids = {rc.chunk.chunk_id for rc in kept}
        assert "real_hit_low_score" in kept_ids
        assert "neighbor" not in kept_ids

    def test_empty_input_returns_empty_output(self, test_config, fake_embedder, fake_generator):
        nodes = GraphNodes(fake_embedder, None, fake_generator, test_config)
        kept, dropped = nodes._trim_to_budget([])
        assert kept == []
        assert dropped == 0
