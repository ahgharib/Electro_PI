"""Resilient wrappers.

ResilientEmbedder and ResilientGenerator both implement their respective
port, so from the rest of the system's point of view they ARE "the"
embedder / generator -- the retry, circuit-breaker, and fallback-provider
logic is entirely invisible to callers. This is what lets component
failure handling live in exactly one place instead of being duplicated
at every call site.
"""

from __future__ import annotations

from src.application.config import Config
from src.application.reliability import CircuitBreaker, call_with_fallback
from src.core.models import RetrievedChunk
from src.ports.embedder import Embedder
from src.ports.generator import GenerationResult, Generator


class ResilientEmbedder(Embedder):
    def __init__(
        self,
        primary: Embedder,
        primary_provider_name: str,
        fallback: Embedder | None,
        fallback_provider_name: str | None,
        config: Config,
    ):
        self._primary = primary
        self._primary_name = primary_provider_name
        self._fallback = fallback
        self._fallback_name = fallback_provider_name
        self._config = config
        self._primary_breaker = CircuitBreaker(
            config.circuit_breaker_failure_threshold, config.circuit_breaker_reset_seconds
        )
        self._fallback_breaker = (
            CircuitBreaker(config.circuit_breaker_failure_threshold, config.circuit_breaker_reset_seconds)
            if fallback
            else None
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return call_with_fallback(
            lambda: self._primary.embed_documents(texts),
            (lambda: self._fallback.embed_documents(texts)) if self._fallback else None,
            component="embedder",
            primary_provider=self._primary_name,
            fallback_provider=self._fallback_name,
            max_attempts=self._config.retry_max_attempts,
            backoff_seconds=self._config.retry_backoff_seconds,
            primary_breaker=self._primary_breaker,
            fallback_breaker=self._fallback_breaker,
        )

    def embed_query(self, text: str) -> list[float]:
        return call_with_fallback(
            lambda: self._primary.embed_query(text),
            (lambda: self._fallback.embed_query(text)) if self._fallback else None,
            component="embedder",
            primary_provider=self._primary_name,
            fallback_provider=self._fallback_name,
            max_attempts=self._config.retry_max_attempts,
            backoff_seconds=self._config.retry_backoff_seconds,
            primary_breaker=self._primary_breaker,
            fallback_breaker=self._fallback_breaker,
        )

    @property
    def model_name(self) -> str:
        return self._primary.model_name


class ResilientGenerator(Generator):
    def __init__(
        self,
        primary: Generator,
        primary_provider_name: str,
        fallback: Generator | None,
        fallback_provider_name: str | None,
        config: Config,
    ):
        self._primary = primary
        self._primary_name = primary_provider_name
        self._fallback = fallback
        self._fallback_name = fallback_provider_name
        self._config = config
        self._primary_breaker = CircuitBreaker(
            config.circuit_breaker_failure_threshold, config.circuit_breaker_reset_seconds
        )
        self._fallback_breaker = (
            CircuitBreaker(config.circuit_breaker_failure_threshold, config.circuit_breaker_reset_seconds)
            if fallback
            else None
        )
        # Tracks which provider actually served the most recent call, so
        # monitoring can log the true model_used even when a fallback fired
        # (model_name alone would always report the primary's model).
        self.last_provider: str = primary_provider_name
        self.last_model: str = primary.model_name

    def generate(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[tuple[str, str]] | None = None,
    ) -> GenerationResult:
        def _call_primary() -> GenerationResult:
            result = self._primary.generate(question, context_chunks, history)
            self.last_provider = self._primary_name
            self.last_model = self._primary.model_name
            return result

        def _call_fallback() -> GenerationResult:
            result = self._fallback.generate(question, context_chunks, history)  # type: ignore[union-attr]
            self.last_provider = self._fallback_name or "fallback"
            self.last_model = self._fallback.model_name  # type: ignore[union-attr]
            return result

        return call_with_fallback(
            _call_primary,
            _call_fallback if self._fallback else None,
            component="generator",
            primary_provider=self._primary_name,
            fallback_provider=self._fallback_name,
            max_attempts=self._config.retry_max_attempts,
            backoff_seconds=self._config.retry_backoff_seconds,
            primary_breaker=self._primary_breaker,
            fallback_breaker=self._fallback_breaker,
        )

    @property
    def model_name(self) -> str:
        return self._primary.model_name

    @property
    def context_window_tokens(self) -> int:
        return self._primary.context_window_tokens
