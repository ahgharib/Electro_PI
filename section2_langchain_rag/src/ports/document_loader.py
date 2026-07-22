"""Port: DocumentLoader.

Adapters implement this to turn a file on disk into one or more Document
objects. The rest of the system never knows or cares whether a Document
came from markdown, PDF, or (later) a Confluence page or S3 bucket.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.models import Document


class DocumentLoader(ABC):
    """Loads raw files into normalized Document objects."""

    @abstractmethod
    def load(self, path: str) -> list[Document]:
        """Load a single file path and return the Document(s) it contains.

        Most loaders return exactly one Document per file. The list return
        type exists for formats that may reasonably split into more than
        one logical document (kept for interface flexibility, not used yet).
        """
        raise NotImplementedError

    @abstractmethod
    def supports(self, path: str) -> bool:
        """Return True if this loader can handle the given file path."""
        raise NotImplementedError
