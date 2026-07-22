"""Port: Chunker.

Isolates chunking strategy from everything else. Swapping fixed-size for
semantic chunking, changing chunk size/overlap, or adding hierarchical
(parent/child) chunking later is a change to ONE adapter class -- nothing
in the retriever, generator, or graph needs to know.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.models import Chunk, Document


class Chunker(ABC):
    """Splits a Document into indexable Chunks."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        raise NotImplementedError
