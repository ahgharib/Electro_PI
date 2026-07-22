from __future__ import annotations

from src.adapters.loaders.markdown_loader import MarkdownLoader
from src.adapters.loaders.pdf_loader import PdfLoader
from src.core.models import Document
from src.ports.document_loader import DocumentLoader


class LoaderRegistry:
    """Picks the right DocumentLoader for a file based on extension.

    Adding support for a new format later (.txt, .docx, ...) means writing
    one new DocumentLoader adapter and adding it to this list -- nothing
    else in the system changes.
    """

    def __init__(self, loaders: list[DocumentLoader] | None = None):
        self._loaders = loaders or [MarkdownLoader(), PdfLoader()]

    def load_file(self, path: str) -> list[Document]:
        for loader in self._loaders:
            if loader.supports(path):
                return loader.load(path)
        raise ValueError(f"No loader registered for file: {path}")

    def load_directory(self, directory: str) -> list[Document]:
        from pathlib import Path

        documents: list[Document] = []
        for path in sorted(Path(directory).iterdir()):
            if path.is_file() and any(loader.supports(str(path)) for loader in self._loaders):
                documents.extend(self.load_file(str(path)))
        return documents
