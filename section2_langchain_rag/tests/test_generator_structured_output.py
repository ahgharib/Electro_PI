from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.core.models import Chunk, RetrievedChunk


def _sample_chunk():
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id="a::0", doc_id="a", source_path="docs/a.md", text="some content", chunk_index=0
        ),
        score=0.8,
    )


class TestGeminiGeneratorStructuredOutput:
    def test_success_populates_parsed_answer_and_skips_text_parsing_need(self):
        with patch("src.adapters.generators.gemini_generator.ChatGoogleGenerativeAI") as MockClient:
            mock_raw = MagicMock()
            mock_raw.content = "irrelevant when structured parsing succeeds"
            mock_raw.usage_metadata = {"input_tokens": 500, "output_tokens": 40, "total_tokens": 540}

            mock_structured = MagicMock()
            mock_structured.invoke.return_value = {
                "raw": mock_raw,
                "parsed": MagicMock(model_dump=lambda: {"answer": "The window is 30 days.", "cited_chunk_ids": ["a::0"]}),
                "parsing_error": None,
            }
            MockClient.return_value.with_structured_output.return_value = mock_structured

            from src.adapters.generators.gemini_generator import GeminiGenerator

            gen = GeminiGenerator(api_key="fake-key", model="gemini-2.5-flash")
            result = gen.generate("What is the window?", [_sample_chunk()])

            assert result.parsed_answer == {"answer": "The window is 30 days.", "cited_chunk_ids": ["a::0"]}
            assert result.prompt_tokens == 500
            assert result.completion_tokens == 40
            assert result.total_tokens == 540

    def test_parsing_failure_falls_back_to_none_parsed_answer_with_raw_text_kept(self):
        with patch("src.adapters.generators.gemini_generator.ChatGoogleGenerativeAI") as MockClient:
            mock_raw = MagicMock()
            mock_raw.content = "The model said something that didn't match the schema."
            mock_raw.usage_metadata = {"input_tokens": 300, "output_tokens": 20, "total_tokens": 320}

            mock_structured = MagicMock()
            mock_structured.invoke.return_value = {
                "raw": mock_raw,
                "parsed": None,
                "parsing_error": ValueError("schema validation failed"),
            }
            MockClient.return_value.with_structured_output.return_value = mock_structured

            from src.adapters.generators.gemini_generator import GeminiGenerator

            gen = GeminiGenerator(api_key="fake-key", model="gemini-2.5-flash")
            result = gen.generate("What is the window?", [_sample_chunk()])

            assert result.parsed_answer is None
            assert result.text == "The model said something that didn't match the schema."
            # token usage is still captured even when structured parsing failed
            assert result.total_tokens == 320

    def test_uses_json_schema_method_with_include_raw(self):
        with patch("src.adapters.generators.gemini_generator.ChatGoogleGenerativeAI") as MockClient:
            from src.adapters.generators.gemini_generator import GeminiGenerator

            GeminiGenerator(api_key="fake-key", model="gemini-2.5-flash")

            _, kwargs = MockClient.return_value.with_structured_output.call_args
            assert kwargs.get("include_raw") is True
            assert kwargs.get("method") == "json_schema"


class TestOpenAIGeneratorStructuredOutput:
    def test_success_populates_parsed_answer(self):
        with patch("src.adapters.generators.openai_generator.ChatOpenAI") as MockClient:
            mock_raw = MagicMock()
            mock_raw.content = "irrelevant when structured parsing succeeds"
            mock_raw.usage_metadata = {"input_tokens": 200, "output_tokens": 15, "total_tokens": 215}

            mock_structured = MagicMock()
            mock_structured.invoke.return_value = {
                "raw": mock_raw,
                "parsed": MagicMock(model_dump=lambda: {"answer": "Yes.", "cited_chunk_ids": []}),
                "parsing_error": None,
            }
            MockClient.return_value.with_structured_output.return_value = mock_structured

            from src.adapters.generators.openai_generator import OpenAIGenerator

            gen = OpenAIGenerator(api_key="fake-key", model="gpt-4o-mini")
            result = gen.generate("Is this true?", [_sample_chunk()])

            assert result.parsed_answer == {"answer": "Yes.", "cited_chunk_ids": []}
            assert result.total_tokens == 215

    def test_parsing_failure_returns_none_parsed_answer(self):
        with patch("src.adapters.generators.openai_generator.ChatOpenAI") as MockClient:
            mock_raw = MagicMock()
            mock_raw.content = "raw fallback text"
            mock_raw.usage_metadata = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}

            mock_structured = MagicMock()
            mock_structured.invoke.return_value = {"raw": mock_raw, "parsed": None, "parsing_error": ValueError("x")}
            MockClient.return_value.with_structured_output.return_value = mock_structured

            from src.adapters.generators.openai_generator import OpenAIGenerator

            gen = OpenAIGenerator(api_key="fake-key", model="gpt-4o-mini")
            result = gen.generate("Is this true?", [_sample_chunk()])

            assert result.parsed_answer is None
            assert result.text == "raw fallback text"
