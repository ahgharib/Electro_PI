"""Reliability: retry-with-backoff, circuit breaker, and provider fallback.

This lives at the component boundary (wrapping an Embedder or Generator),
so the pipeline/graph never has to know HOW a component recovers from a
failure -- only whether it eventually succeeded or raised ComponentError.

Failure handling policy (see NOTES.md for the full reasoning):
  1. Retry the call a small number of times with linear backoff (handles
     transient network blips / rate limits).
  2. If retries are exhausted, trip a circuit breaker for that
     component+provider pair so we stop hammering a service that is down.
  3. If a fallback provider is configured, switch to it automatically.
  4. If there is no fallback, or the fallback also fails, raise
     ComponentError -- the caller (the graph node) turns this into a clear
     "service unavailable" response rather than a stack trace reaching the
     user.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from src.application.exceptions import ComponentError, NoProviderAvailableError, QuotaExceededError

logger = logging.getLogger("rag.reliability")

T = TypeVar("T")

# Substrings that show up across providers' 429 / rate-limit / daily-quota
# error messages. This is a heuristic (providers don't expose a clean
# shared exception type through LangChain), but it's enough to distinguish
# "you're being throttled" from "something is actually broken" for
# messaging purposes.
_QUOTA_ERROR_MARKERS = ("429", "quota", "resource_exhausted", "rate limit", "rate_limit")


def _looks_like_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _QUOTA_ERROR_MARKERS)


class RateLimiter:
    """A minimal proactive pacer: sleeps as needed to keep calls under a
    configured requests-per-minute rate, rather than only reacting to a 429
    after it's already happened. Disabled (no-op) when max_per_minute <= 0.

    This is intentionally simple (fixed inter-call spacing, not a true
    token bucket with burst allowance) -- for the batch-ingestion use case
    this is built for, steady pacing is exactly what's wanted.
    """

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self._min_interval = 60.0 / max_per_minute if max_per_minute > 0 else 0.0
        self._last_call: float | None = None

    def wait(self) -> None:
        if self._min_interval <= 0:
            return
        now = time.monotonic()
        if self._last_call is not None:
            elapsed = now - self._last_call
            remaining = self._min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_call = time.monotonic()


@dataclass
class CircuitBreakerState:
    failure_count: int = 0
    opened_at: float | None = None


class CircuitBreaker:
    """A minimal in-memory circuit breaker, one instance per component+provider.

    States:
      closed    -> calls go through normally.
      open      -> calls are rejected immediately (no network call at all)
                   until `reset_seconds` has elapsed.
      half-open -> (implicit) after reset_seconds, the next call is allowed
                   through; success closes the breaker, failure re-opens it.
    """

    def __init__(self, failure_threshold: int, reset_seconds: float):
        self.failure_threshold = failure_threshold
        self.reset_seconds = reset_seconds
        self._state = CircuitBreakerState()

    def is_open(self) -> bool:
        if self._state.opened_at is None:
            return False
        if time.monotonic() - self._state.opened_at >= self.reset_seconds:
            # cool-down elapsed -> move to half-open (allow one attempt through)
            return False
        return True

    def record_success(self) -> None:
        self._state = CircuitBreakerState()

    def record_failure(self) -> None:
        self._state.failure_count += 1
        if self._state.failure_count >= self.failure_threshold:
            self._state.opened_at = time.monotonic()


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    component: str,
    provider: str,
    max_attempts: int,
    backoff_seconds: float,
    breaker: CircuitBreaker,
) -> T:
    """Run fn() with retries + linear backoff, guarded by a circuit breaker.

    Raises ComponentError if all attempts are exhausted or the breaker is open.
    """
    if breaker.is_open():
        logger.warning("circuit_open component=%s provider=%s -- skipping call", component, provider)
        raise ComponentError(component, provider, original=RuntimeError("circuit breaker open"))

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = fn()
            breaker.record_success()
            return result
        except Exception as exc:  # noqa: BLE001 -- intentionally broad: adapter-agnostic
            last_error = exc
            breaker.record_failure()
            is_quota_error = _looks_like_quota_error(exc)
            logger.warning(
                "component_call_failed component=%s provider=%s attempt=%d/%d quota_error=%s error=%s",
                component, provider, attempt, max_attempts, is_quota_error, exc,
            )
            if is_quota_error:
                # Retrying immediately never helps a rate-limit/quota
                # rejection -- either it's a per-minute limit (needs much
                # longer than our backoff window to clear) or a daily quota
                # (needs until the next reset). Fail fast instead of
                # burning through remaining attempts and further backoff
                # delay for no benefit.
                raise QuotaExceededError(component, provider, original=exc) from exc
            if attempt < max_attempts:
                time.sleep(backoff_seconds * attempt)

    raise ComponentError(component, provider, original=last_error)


def call_with_fallback(
    primary_fn: Callable[[], T],
    fallback_fn: Callable[[], T] | None,
    *,
    component: str,
    primary_provider: str,
    fallback_provider: str | None,
    max_attempts: int,
    backoff_seconds: float,
    primary_breaker: CircuitBreaker,
    fallback_breaker: CircuitBreaker | None,
) -> T:
    """Try the primary provider (with retry+breaker); on exhaustion, try fallback."""
    try:
        return retry_with_backoff(
            primary_fn,
            component=component,
            provider=primary_provider,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            breaker=primary_breaker,
        )
    except ComponentError as primary_error:
        if fallback_fn is None or fallback_breaker is None:
            raise
        logger.warning(
            "falling_back component=%s from=%s to=%s", component, primary_provider, fallback_provider
        )
        try:
            return retry_with_backoff(
                fallback_fn,
                component=component,
                provider=fallback_provider or "unknown",
                max_attempts=max_attempts,
                backoff_seconds=backoff_seconds,
                breaker=fallback_breaker,
            )
        except ComponentError as fallback_error:
            raise NoProviderAvailableError(
                f"{component}: both primary ({primary_provider}) and fallback "
                f"({fallback_provider}) failed. primary={primary_error} fallback={fallback_error}"
            ) from fallback_error
