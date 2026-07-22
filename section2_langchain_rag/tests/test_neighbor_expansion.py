from __future__ import annotations

from src.application.graph_nodes import GraphNodes
from src.core.models import Chunk, RetrievedChunk


def _chunk(doc_id, index):
    return Chunk(
        chunk_id=f"{doc_id}::chunk::{index}",
        doc_id=doc_id,
        source_path=f"{doc_id}.md",
        text=f"text {index}",
        chunk_index=index,
    )


class _FakeStore:
    """Minimal stand-in vector store exposing only get_by_doc_and_index,
    which is all _expand_with_neighbors needs."""

    def __init__(self, chunks_by_key):
        self._chunks = chunks_by_key

    def get_by_doc_and_index(self, doc_id, chunk_index):
        return self._chunks.get((doc_id, chunk_index))


class TestNeighborExpansion:
    def test_pulls_in_adjacent_chunks(self, test_config, fake_embedder, fake_generator):
        store = _FakeStore(
            {
                ("docA", 4): _chunk("docA", 4),
                ("docA", 5): _chunk("docA", 5),
                ("docA", 6): _chunk("docA", 6),
            }
        )
        nodes = GraphNodes(fake_embedder, store, fake_generator, test_config)
        retrieved = [RetrievedChunk(chunk=_chunk("docA", 5), score=0.9)]

        expanded, added = nodes._expand_with_neighbors(retrieved)

        keys = {(rc.chunk.doc_id, rc.chunk.chunk_index) for rc in expanded}
        assert keys == {("docA", 4), ("docA", 5), ("docA", 6)}
        assert added == 2

    def test_neighbors_are_flagged_and_not_duplicated(self, test_config, fake_embedder, fake_generator):
        store = _FakeStore({("docA", 4): _chunk("docA", 4), ("docA", 6): _chunk("docA", 6)})
        nodes = GraphNodes(fake_embedder, store, fake_generator, test_config)
        retrieved = [
            RetrievedChunk(chunk=_chunk("docA", 5), score=0.9),
            RetrievedChunk(chunk=_chunk("docA", 4), score=0.8),  # already retrieved directly
        ]

        expanded, added = nodes._expand_with_neighbors(retrieved)

        # chunk 4 should not be duplicated even though it's a neighbor of chunk 5
        keys = [(rc.chunk.doc_id, rc.chunk.chunk_index) for rc in expanded]
        assert keys.count(("docA", 4)) == 1
        neighbor_flags = {rc.chunk.chunk_index: rc.is_neighbor_expansion for rc in expanded}
        assert neighbor_flags[6] is True  # pulled in as a neighbor
        assert neighbor_flags[4] is False  # was a real retrieval hit, not a neighbor

    def test_missing_neighbor_is_skipped_gracefully(self, test_config, fake_embedder, fake_generator):
        store = _FakeStore({})  # no neighbors exist
        nodes = GraphNodes(fake_embedder, store, fake_generator, test_config)
        retrieved = [RetrievedChunk(chunk=_chunk("docA", 5), score=0.9)]

        expanded, added = nodes._expand_with_neighbors(retrieved)

        assert added == 0
        assert len(expanded) == 1
