from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from src.core.models import Document
from src.ports.document_loader import DocumentLoader


class PdfLoader(DocumentLoader):
    """Loads .pdf files, keeping each page's text separate.

    Document.metadata["pages"] holds a list of {"page": N, "text": "..."}
    dicts, one per page. The chunker then splits EACH page's text
    independently (see StructureAwareChunker._chunk_pdf), so a chunk can
    never straddle a page boundary and page attribution is always exact --
    no offset-guessing heuristic involved. Document.text still holds the
    full concatenated text for anything that wants the whole document
    (e.g. a future full-document display), but it is not used for chunking.
    """

    def supports(self, path: str) -> bool:
        return Path(path).suffix.lower() == ".pdf"

    def load(self, path: str) -> list[Document]:
        p = Path(path)
        reader = PdfReader(str(p))

        pages: list[dict] = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            pages.append({"page": page_number, "text": page_text})

        full_text = "\n".join(page["text"] for page in pages)
        doc_id = p.stem

        return [
            Document(
                doc_id=doc_id,
                source_path=str(p),
                source_type="pdf",
                text=full_text,
                metadata={"pages": pages},
            )
        ]
