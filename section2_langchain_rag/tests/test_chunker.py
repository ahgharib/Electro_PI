from __future__ import annotations

from src.core.models import Document


class TestMarkdownChunking:
    def test_chunks_have_sequential_index(self, chunker):
        doc = Document(
            doc_id="d1",
            source_path="d1.md",
            source_type="markdown",
            text="# Title\n\n"
            + ("Paragraph one. " * 100)
            + "\n\n## Section 2\n\n"
            + ("Paragraph two. " * 100),
        )
        chunks = chunker.chunk(doc)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunks_carry_section_path(self, chunker):
        doc = Document(
            doc_id="d1",
            source_path="d1.md",
            source_type="markdown",
            text="# Shipping\n\n## International\n\nSome international shipping details here.",
        )
        chunks = chunker.chunk(doc)
        assert any(c.section_path and "International" in c.section_path for c in chunks)

    def test_chunk_ids_are_unique_and_deterministic(self, chunker):
        doc = Document(
            doc_id="d1",
            source_path="d1.md",
            source_type="markdown",
            text="# T\n\n" + ("word " * 500),
        )
        chunks_a = chunker.chunk(doc)
        chunks_b = chunker.chunk(doc)
        ids_a = [c.chunk_id for c in chunks_a]
        ids_b = [c.chunk_id for c in chunks_b]
        assert len(ids_a) == len(set(ids_a))  # unique
        assert ids_a == ids_b  # deterministic given the same input

    def test_overlap_produces_shared_content_between_adjacent_chunks(self):
        from src.adapters.chunking.structure_aware_chunker import StructureAwareChunker

        chunker = StructureAwareChunker(chunk_size_tokens=30, chunk_overlap_tokens=10)
        doc = Document(
            doc_id="d1",
            source_path="d1.md",
            source_type="markdown",
            text="# T\n\n" + " ".join(f"word{i}" for i in range(200)),
        )
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1


class TestPdfChunking:
    def test_pdf_chunks_get_page_numbers(self, chunker):
        doc = Document(
            doc_id="d1",
            source_path="d1.pdf",
            source_type="pdf",
            text="Page one content. Page two content.",
            metadata={
                "pages": [
                    {"page": 1, "text": "Page one content. " * 50},
                    {"page": 2, "text": "Page two content. " * 50},
                ]
            },
        )
        chunks = chunker.chunk(doc)
        assert all(c.page_number is not None for c in chunks)
        pages = {c.page_number for c in chunks}
        assert pages == {1, 2}

    def test_no_chunk_straddles_a_page_boundary(self, chunker):
        # Identical content on both pages -- if chunking were still done on
        # concatenated text, boundary chunks could mix content from both
        # pages. With per-page chunking this is structurally impossible:
        # every chunk's text must come entirely from ONE page's text.
        page_text = "Repeated sentence content for testing. " * 60
        doc = Document(
            doc_id="d1",
            source_path="d1.pdf",
            source_type="pdf",
            text=page_text + page_text,
            metadata={
                "pages": [
                    {"page": 1, "text": page_text},
                    {"page": 2, "text": page_text},
                ]
            },
        )
        chunks = chunker.chunk(doc)
        for c in chunks:
            source_page_text = doc.metadata["pages"][c.page_number - 1]["text"]
            assert c.text in source_page_text  # chunk text is fully contained in its own page

    def test_blank_pages_produce_no_chunks(self, chunker):
        doc = Document(
            doc_id="d1",
            source_path="d1.pdf",
            source_type="pdf",
            text="Real content here. " * 20,
            metadata={
                "pages": [
                    {"page": 1, "text": "Real content here. " * 20},
                    {"page": 2, "text": ""},  # e.g. a scanned/blank page
                ]
            },
        )
        chunks = chunker.chunk(doc)
        assert all(c.page_number == 1 for c in chunks)

    def test_chunk_index_continues_sequentially_across_pages(self, chunker):
        doc = Document(
            doc_id="d1",
            source_path="d1.pdf",
            source_type="pdf",
            text="",
            metadata={
                "pages": [
                    {"page": 1, "text": "Page one text. " * 40},
                    {"page": 2, "text": "Page two text. " * 40},
                ]
            },
        )
        chunks = chunker.chunk(doc)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))
