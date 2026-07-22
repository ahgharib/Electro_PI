from __future__ import annotations

from src.application.graph_nodes import GraphNodes
from src.core.models import Chunk, RetrievedChunk


def _rc(chunk_id, doc_id, source, section=None, score=0.9):
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            source_path=source,
            text="some text",
            chunk_index=0,
            section_path=section,
        ),
        score=score,
    )


class TestParseGenerationOutput:
    def test_parses_valid_json(self):
        raw = '{"answer": "The window is 30 days.", "cited_chunk_ids": ["a::chunk::0"]}'
        answer, cited = GraphNodes._parse_generation_output(raw)
        assert answer == "The window is 30 days."
        assert cited == ["a::chunk::0"]

    def test_strips_markdown_code_fences(self):
        raw = '```json\n{"answer": "Yes.", "cited_chunk_ids": []}\n```'
        answer, cited = GraphNodes._parse_generation_output(raw)
        assert answer == "Yes."
        assert cited == []

    def test_falls_back_to_raw_text_on_invalid_json(self):
        raw = "This is not JSON at all."
        answer, cited = GraphNodes._parse_generation_output(raw)
        assert answer == "This is not JSON at all."
        assert cited == []


class TestBuildCitations:
    def test_groups_citations_by_source_file(self):
        retrieved = [
            _rc("a::chunk::0", "a", "docs/a.md", section="Intro"),
            _rc("a::chunk::1", "a", "docs/a.md", section="Details"),
            _rc("b::chunk::0", "b", "docs/b.md", section="Other"),
        ]
        citations = GraphNodes._build_citations(
            retrieved, cited_chunk_ids=["a::chunk::0", "a::chunk::1", "b::chunk::0"]
        )
        by_source = {c.source_path: c for c in citations}
        assert set(by_source.keys()) == {"docs/a.md", "docs/b.md"}
        assert set(by_source["docs/a.md"].chunk_ids) == {"a::chunk::0", "a::chunk::1"}
        assert set(by_source["docs/a.md"].section_paths) == {"Intro", "Details"}

    def test_ignores_cited_ids_not_in_retrieved_set(self):
        retrieved = [_rc("a::chunk::0", "a", "docs/a.md")]
        citations = GraphNodes._build_citations(retrieved, cited_chunk_ids=["a::chunk::0", "made-up-id"])
        assert len(citations) == 1
        assert citations[0].chunk_ids == ["a::chunk::0"]

    def test_no_citations_when_nothing_cited(self):
        retrieved = [_rc("a::chunk::0", "a", "docs/a.md")]
        citations = GraphNodes._build_citations(retrieved, cited_chunk_ids=[])
        assert citations == []
