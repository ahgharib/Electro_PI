"""Port: VectorStore.

Chroma is the MVP adapter (see src/adapters/vectorstores/chroma_store.py).
FAISS, pgvector, Pinecone, Qdrant, MongoDB Atlas Vector Search, etc. are all
drop-in replacements behind this same interface -- see NOTES.md for why
Chroma was chosen for this task specifically and when each alternative
would be the better call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.models import Chunk, RetrievedChunk


class VectorStore(ABC):
    """Stores chunk embeddings and retrieves nearest neighbors for a query vector."""

    @abstractmethod
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]:
        raise NotImplementedError

    @abstractmethod
    def get_by_doc_and_index(self, doc_id: str, chunk_index: int) -> Chunk | None:
        """Fetch a specific chunk by its position in its source document.

        Used for neighbor-chunk expansion (pulling in chunk N-1 / N+1 around
        a chunk that scored well), without needing a second similarity search.
        """
        raise NotImplementedError

    @abstractmethod
    def get_existing_content_hashes(self, chunk_ids: list[str]) -> dict[str, str]:
        """Return {chunk_id: content_hash} for whichever of the given IDs
        are already indexed (IDs not present in the store are simply absent
        from the result, not mapped to None).

        Used by RAGPipeline.ingest() to skip re-embedding chunks whose
        content hasn't changed since the last run -- repeated ingest runs
        during development would otherwise re-burn embedding API quota on
        every re-run, which is unnecessary and, on a free tier, can exhaust
        the daily quota surprisingly fast on larger document sets.
        """
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Total number of chunks currently indexed. Used to distinguish an
        EMPTY_INDEX gate outcome from a BELOW_THRESHOLD one."""
        raise NotImplementedError
