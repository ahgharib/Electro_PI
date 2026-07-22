"""Port: Embedder.

Wraps whichever embedding provider is behind it (Gemini, OpenAI, a local
model, a fake for tests). The vector store and retriever only ever talk to
this interface, never to a specific SDK.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    """Turns text into vectors."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of chunk texts (ingestion time)."""
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single user question (query time)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError
