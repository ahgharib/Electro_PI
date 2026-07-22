from __future__ import annotations

import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.adapters.generators.prompt import SYSTEM_PROMPT, build_user_message
from src.adapters.generators.response_text import extract_response_text
from src.adapters.generators.schema import StructuredAnswer
from src.core.models import RetrievedChunk
from src.ports.generator import GenerationResult, Generator


class OpenAIGenerator(Generator):
    """Generator adapter for OpenAI. Used as the configured fallback
    provider, and as proof that swapping LLM providers is a config change,
    not a code change -- see FALLBACK_GENERATION_PROVIDER in .env.

    Uses response_schema-constrained decoding (with_structured_output),
    matching GeminiGenerator -- see that file's docstring for why.
    """

    def __init__(self, api_key: str, model: str, context_window_tokens: int = 128_000):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIGenerator")
        self._model = model
        self._context_window_tokens = context_window_tokens
        self._client = ChatOpenAI(model=model, api_key=api_key, temperature=0)
        self._structured_client = self._client.with_structured_output(StructuredAnswer, include_raw=True)

    def generate(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[tuple[str, str]] | None = None,
    ) -> GenerationResult:
        user_message = build_user_message(question, context_chunks, history)
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_message)]

        start = time.monotonic()
        result = self._structured_client.invoke(messages)
        latency_ms = (time.monotonic() - start) * 1000

        raw = result.get("raw")
        parsed = result.get("parsed")
        parsing_error = result.get("parsing_error")

        usage = getattr(raw, "usage_metadata", None) or {}
        prompt_tokens = usage.get("input_tokens", 0) or 0
        completion_tokens = usage.get("output_tokens", 0) or 0
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens) or (
            prompt_tokens + completion_tokens
        )

        response_text = extract_response_text(getattr(raw, "content", "") if raw is not None else "")

        parsed_answer: dict | None = None
        if parsed is not None and parsing_error is None:
            parsed_answer = parsed.model_dump()

        return GenerationResult(
            text=response_text,
            parsed_answer=parsed_answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def context_window_tokens(self) -> int:
        return self._context_window_tokens
