from __future__ import annotations

import json
import time

from src.core.models import RetrievedChunk
from src.ports.generator import GenerationResult, Generator


class FakeGenerator(Generator):
    """Deterministic generator for tests -- no network call, no API key.

    By default, populates parsed_answer directly (mirroring what a real
    provider's structured-output call should normally return), citing the
    first retrieved chunk if any were given. Pass
    simulate_structured_failure=True to instead simulate the case where
    provider-native structured output itself failed to parse -- text is
    set to something that ALSO doesn't parse as JSON, exercising the full
    fallback path in GraphNodes._parse_generation_output.
    """

    def __init__(
        self,
        model_name: str = "fake-generator",
        context_window_tokens: int = 8000,
        simulate_structured_failure: bool = False,
    ):
        self._model_name = model_name
        self._context_window_tokens = context_window_tokens
        self._simulate_structured_failure = simulate_structured_failure

    def generate(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[tuple[str, str]] | None = None,
    ) -> GenerationResult:
        start = time.monotonic()
        if context_chunks:
            cited = [context_chunks[0].chunk.chunk_id]
            answer = f"[fake answer from {len(context_chunks)} chunk(s)] {context_chunks[0].chunk.text[:80]}"
        else:
            cited = []
            answer = "I don't have enough information in the provided documents to answer that."

        if self._simulate_structured_failure:
            text = f"Sorry, here's an answer without following the requested format: {answer}"
            parsed_answer = None
        else:
            text = json.dumps({"answer": answer, "cited_chunk_ids": cited})
            parsed_answer = {"answer": answer, "cited_chunk_ids": cited}

        latency_ms = (time.monotonic() - start) * 1000
        prompt_tokens = len(question.split()) + sum(len(rc.chunk.text.split()) for rc in context_chunks)
        completion_tokens = len(answer.split())
        return GenerationResult(
            text=text,
            parsed_answer=parsed_answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def context_window_tokens(self) -> int:
        return self._context_window_tokens
