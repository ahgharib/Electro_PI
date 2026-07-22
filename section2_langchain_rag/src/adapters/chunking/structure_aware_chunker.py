"""Structure-aware recursive chunker.

Strategy (see NOTES.md for the full trade-off discussion):
  - Markdown: split on headers (#, ##, ###) first to get structural
    sections, then recursively split each section by paragraph/sentence/
    character down to the target token size, with overlap. Each chunk
    keeps its header path (e.g. "Shipping > International") as metadata.
  - PDF: no reliable structural markers, so each PAGE's text (kept
    separate by PdfLoader) is split independently with the same recursive
    splitter. A chunk is never built from text spanning two pages, so page
    attribution is always exact -- no offset-guessing involved.

chunk_index is assigned sequentially across the WHOLE document (not reset
per section/page), so neighbor-expansion (doc_id, chunk_index -1/+1) works
correctly regardless of which structural section/page a chunk came from.
"""

from __future__ import annotations

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.application.token_utils import approximate_token_count
from src.core.models import Chunk, Document
from src.ports.chunker import Chunker


def _token_len(text: str) -> int:
    # Used as an approximate, provider-agnostic length function for the
    # splitter. It won't exactly match any specific LLM's tokenizer, but it
    # keeps chunk sizing consistent and predictable regardless of which
    # provider is configured, without a hard runtime dependency on
    # downloading tokenizer data from an external service (see
    # src/application/token_utils.py for the full reasoning).
    return approximate_token_count(text)


class StructureAwareChunker(Chunker):
    def __init__(self, chunk_size_tokens: int = 600, chunk_overlap_tokens: int = 80):
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size_tokens,
            chunk_overlap=chunk_overlap_tokens,
            length_function=_token_len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=False,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        if document.source_type == "markdown":
            return self._chunk_markdown(document)
        return self._chunk_pdf(document)

    # ------------------------------------------------------------------ #

    def _chunk_markdown(self, document: Document) -> list[Chunk]:
        sections = self._md_splitter.split_text(document.text)
        chunks: list[Chunk] = []
        chunk_index = 0

        for section in sections:
            header_values = [v for v in section.metadata.values() if v]
            section_path = " > ".join(header_values) if header_values else None

            sub_texts = self._splitter.split_text(section.page_content)
            for sub_text in sub_texts:
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}::chunk::{chunk_index}",
                        doc_id=document.doc_id,
                        source_path=document.source_path,
                        text=sub_text,
                        chunk_index=chunk_index,
                        section_path=section_path,
                        page_number=None,
                    )
                )
                chunk_index += 1
        return chunks

    def _chunk_pdf(self, document: Document) -> list[Chunk]:
        pages = document.metadata.get("pages", [])
        chunks: list[Chunk] = []
        chunk_index = 0

        for page in pages:
            page_text = page["text"]
            if not page_text.strip():
                continue  # blank/unextractable page -- nothing to index

            sub_texts = self._splitter.split_text(page_text)
            for sub_text in sub_texts:
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}::chunk::{chunk_index}",
                        doc_id=document.doc_id,
                        source_path=document.source_path,
                        text=sub_text,
                        chunk_index=chunk_index,
                        section_path=None,
                        page_number=page["page"],
                    )
                )
                chunk_index += 1
        return chunks
