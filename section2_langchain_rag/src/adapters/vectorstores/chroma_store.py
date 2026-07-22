from __future__ import annotations

import chromadb

from src.application.content_hash import compute_content_hash
from src.core.models import Chunk, RetrievedChunk
from src.ports.vector_store import VectorStore


class ChromaVectorStore(VectorStore):
    """Chroma adapter -- see NOTES.md for why Chroma was chosen over FAISS
    for this task (native metadata storage, no separate id->metadata store
    to maintain) and when FAISS/pgvector/Pinecone/Qdrant would be preferred
    instead. Persisted locally to disk (no server required).
    """

    def __init__(self, persist_directory: str, collection_name: str):
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        # In-memory lookup for neighbor-chunk expansion, hydrated from the
        # persisted collection on startup so it survives process restarts.
        self._chunks_by_id: dict[str, Chunk] = {}
        self._index_lookup: dict[tuple[str, int], str] = {}
        self._hydrate_lookup()

    def _hydrate_lookup(self) -> None:
        if self._collection.count() == 0:
            return
        existing = self._collection.get(include=["metadatas", "documents"])
        for chunk_id, meta, text in zip(
            existing.get("ids", []), existing.get("metadatas", []), existing.get("documents", [])
        ):
            chunk = self._chunk_from_record(chunk_id, meta, text)
            self._chunks_by_id[chunk_id] = chunk
            self._index_lookup[(chunk.doc_id, chunk.chunk_index)] = chunk_id

    @staticmethod
    def _chunk_from_record(chunk_id: str, meta: dict, text: str) -> Chunk:
        page_number = meta.get("page_number", -1)
        return Chunk(
            chunk_id=chunk_id,
            doc_id=meta.get("doc_id", ""),
            source_path=meta.get("source_path", ""),
            text=text,
            chunk_index=int(meta.get("chunk_index", 0)),
            section_path=meta.get("section_path") or None,
            page_number=None if page_number in (-1, None) else int(page_number),
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "source_path": c.source_path,
                "chunk_index": c.chunk_index,
                "section_path": c.section_path or "",
                "page_number": c.page_number if c.page_number is not None else -1,
                "content_hash": compute_content_hash(c.text),
            }
            for c in chunks
        ]
        documents = [c.text for c in chunks]
        self._collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
        for c in chunks:
            self._chunks_by_id[c.chunk_id] = c
            self._index_lookup[(c.doc_id, c.chunk_index)] = c.chunk_id

    def search(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]:
        total = self.count()
        if total == 0:
            return []
        results = self._collection.query(query_embeddings=[query_embedding], n_results=min(k, total))
        retrieved: list[RetrievedChunk] = []
        for chunk_id, distance, meta, text in zip(
            results["ids"][0], results["distances"][0], results["metadatas"][0], results["documents"][0]
        ):
            chunk = self._chunk_from_record(chunk_id, meta, text)
            # Collection uses cosine space -> distance in [0, 2], similarity = 1 - distance
            score = 1.0 - distance
            retrieved.append(RetrievedChunk(chunk=chunk, score=score))
        return retrieved

    def get_by_doc_and_index(self, doc_id: str, chunk_index: int) -> Chunk | None:
        chunk_id = self._index_lookup.get((doc_id, chunk_index))
        return self._chunks_by_id.get(chunk_id) if chunk_id else None

    def get_existing_content_hashes(self, chunk_ids: list[str]) -> dict[str, str]:
        if not chunk_ids:
            return {}
        # Chroma raises if asked for zero matching ids in some versions, and
        # ids not present are simply omitted from the response either way --
        # a single bulk get() call regardless of how many of these ids exist.
        existing = self._collection.get(ids=chunk_ids, include=["metadatas"])
        result: dict[str, str] = {}
        for chunk_id, meta in zip(existing.get("ids", []), existing.get("metadatas", [])):
            content_hash = (meta or {}).get("content_hash")
            if content_hash:
                result[chunk_id] = content_hash
        return result

    def count(self) -> int:
        return self._collection.count()
