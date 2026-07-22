from __future__ import annotations

import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.adapters.generators.prompt import SYSTEM_PROMPT, build_user_message
from src.adapters.generators.response_text import extract_response_text
from src.adapters.generators.schema import StructuredAnswer
from src.core.models import RetrievedChunk
from src.ports.generator import GenerationResult, Generator


class GeminiGenerator(Generator):
    """Generator adapter for Google's Gemini API (free tier by default).

    Requires GOOGLE_API_KEY. Model name configurable via .env
    (GEMINI_CHAT_MODEL) -- default is gemini-2.5-flash, the current
    free-tier default at time of writing; update the .env value if Google
    renames/deprecates it later, no code change needed.

    Uses response_schema-constrained decoding (with_structured_output),
    not just a prompt instruction to "return JSON" -- the API itself
    enforces the shape server-side. include_raw=True keeps the underlying
    AIMessage available for token usage and as a fallback if structured
    parsing still fails (rare, but not impossible -- see CHANGELOG.md for
    why relying on prompt-only JSON formatting was a real bug here before).
    """

    def __init__(self, api_key: str, model: str, context_window_tokens: int = 1_000_000):
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for GeminiGenerator")
        self._model = model
        self._context_window_tokens = context_window_tokens
        self._client = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0)
        self._structured_client = self._client.with_structured_output(
            StructuredAnswer, method="json_schema", include_raw=True
        )

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

        # response.content is NOT reliably a plain string -- newer Gemini
        # models return a list/dict of content blocks (and attach internal
        # metadata like a reasoning "signature" alongside the text). Kept
        # as a fallback/reference text even when structured parsing
        # succeeds; see response_text.py for the full reasoning.
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
