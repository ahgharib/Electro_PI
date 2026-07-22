from __future__ import annotations

from src.adapters.generators.response_text import extract_response_text
from src.application.graph_nodes import GraphNodes


class TestExtractResponseText:
    def test_plain_string_passes_through(self):
        assert extract_response_text("plain text") == "plain text"

    def test_single_dict_block_extracts_text(self):
        content = {"type": "text", "text": "hello world", "extras": {"signature": "abc123"}}
        assert extract_response_text(content) == "hello world"

    def test_list_of_dict_blocks_extracts_and_joins_text(self):
        content = [
            {"type": "text", "text": "part one. "},
            {"type": "text", "text": "part two."},
        ]
        assert extract_response_text(content) == "part one. part two."

    def test_list_ignores_non_text_blocks(self):
        content = [
            {"type": "text", "text": "the real answer"},
            {"type": "thinking", "text": "internal reasoning that should not leak"},
            {"type": "tool_use", "id": "x", "name": "y", "input": {}},
        ]
        # Only "text" (and untyped) blocks are kept -- "thinking"/"tool_use" are dropped.
        result = extract_response_text(content)
        assert result == "the real answer"
        assert "internal reasoning" not in result

    def test_list_of_plain_strings(self):
        assert extract_response_text(["a", "b", "c"]) == "abc"

    def test_none_returns_empty_string(self):
        assert extract_response_text(None) == ""

    def test_signature_metadata_never_appears_in_output(self):
        # Regression test for the exact bug reported: a Gemini response
        # content block carrying a reasoning "signature" alongside the text.
        content = {
            "type": "text",
            "text": '```json\n{"answer": "The window is 30 days.", "cited_chunk_ids": ["a::chunk::0"]}\n```',
            "extras": {"signature": "EoYZCoMZ...very-long-base64-blob..."},
        }
        result = extract_response_text(content)
        assert "signature" not in result
        assert "EoYZ" not in result
        assert result.startswith("```json")


class TestEndToEndGeminiStyleResponseParsing:
    """Simulates the exact response shape reported as a bug: a single
    content-block dict with markdown-fenced JSON plus a signature field."""

    def test_dict_content_block_parses_cleanly_through_the_full_pipeline(self):
        raw_content = {
            "type": "text",
            "text": (
                '```json\n{\n  "answer": "Products may be returned within 30 days.",\n'
                '  "cited_chunk_ids": ["03_returns_and_refunds::chunk::0"]\n}\n```'
            ),
            "extras": {"signature": "some-long-opaque-signature-blob"},
        }

        # Step 1: adapter-level normalization (what GeminiGenerator now does)
        normalized_text = extract_response_text(raw_content)

        # Step 2: graph-level JSON parsing (what _parse_generation_output does)
        answer, cited_ids = GraphNodes._parse_generation_output(normalized_text)

        assert answer == "Products may be returned within 30 days."
        assert cited_ids == ["03_returns_and_refunds::chunk::0"]
        assert "signature" not in answer
