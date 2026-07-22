from __future__ import annotations

from pathlib import Path

from src.core.models import Document
from src.ports.document_loader import DocumentLoader


class MarkdownLoader(DocumentLoader):
    """Loads .md / .markdown files as-is; header structure is handled later
    by the chunker (see adapters/chunking/structure_aware_chunker.py)."""

    def supports(self, path: str) -> bool:
        return Path(path).suffix.lower() in (".md", ".markdown")

    def load(self, path: str) -> list[Document]:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        doc_id = p.stem
        return [
            Document(
                doc_id=doc_id,
                source_path=str(p),
                source_type="markdown",
                text=text,
                metadata={},
            )
        ]
