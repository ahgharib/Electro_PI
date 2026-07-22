from __future__ import annotations

import time

import pytest

from src.application.exceptions import ComponentError, QuotaExceededError
from src.application.reliability import CircuitBreaker, RateLimiter, retry_with_backoff


class TestRateLimiter:
    def test_disabled_when_limit_is_zero(self):
        limiter = RateLimiter(max_per_minute=0)
        start = time.monotonic()
        for _ in range(5):
            limiter.wait()
        assert time.monotonic() - start < 0.05  # effectively instant, no pacing

    def test_paces_calls_to_stay_under_limit(self):
        # 600 calls/minute = one call every 0.1s
        limiter = RateLimiter(max_per_minute=600)
        start = time.monotonic()
        for _ in range(4):
            limiter.wait()
        elapsed = time.monotonic() - start
        # 4 calls at 0.1s spacing -> at least ~0.3s elapsed (first call is free)
        assert elapsed >= 0.25

    def test_first_call_does_not_wait(self):
        limiter = RateLimiter(max_per_minute=10)  # would otherwise be a 6s wait
        start = time.monotonic()
        limiter.wait()
        assert time.monotonic() - start < 0.1


class TestQuotaErrorDetection:
    def test_429_error_raises_quota_exceeded(self):
        breaker = CircuitBreaker(failure_threshold=10, reset_seconds=30)

        def fn():
            raise RuntimeError("429 Resource has been exhausted (e.g. check quota).")

        with pytest.raises(QuotaExceededError):
            retry_with_backoff(
                fn, component="embedder", provider="gemini", max_attempts=3, backoff_seconds=0, breaker=breaker
            )

    def test_resource_exhausted_raises_quota_exceeded(self):
        breaker = CircuitBreaker(failure_threshold=10, reset_seconds=30)

        def fn():
            raise RuntimeError("google.api_core.exceptions.ResourceExhausted: 429 RESOURCE_EXHAUSTED")

        with pytest.raises(QuotaExceededError):
            retry_with_backoff(
                fn, component="generator", provider="gemini", max_attempts=3, backoff_seconds=0, breaker=breaker
            )

    def test_quota_error_does_not_burn_all_retry_attempts(self):
        # Quota errors should fail fast on the FIRST attempt, not retry
        # max_attempts times -- retrying immediately never helps quota.
        breaker = CircuitBreaker(failure_threshold=10, reset_seconds=30)
        calls = []

        def fn():
            calls.append(1)
            raise RuntimeError("429 quota exceeded")

        with pytest.raises(QuotaExceededError):
            retry_with_backoff(
                fn, component="embedder", provider="gemini", max_attempts=5, backoff_seconds=0, breaker=breaker
            )
        assert len(calls) == 1  # did not retry 5 times

    def test_non_quota_error_still_retries_normally(self):
        breaker = CircuitBreaker(failure_threshold=10, reset_seconds=30)
        calls = []

        def fn():
            calls.append(1)
            raise RuntimeError("connection reset by peer")

        with pytest.raises(ComponentError) as exc_info:
            retry_with_backoff(
                fn, component="embedder", provider="gemini", max_attempts=3, backoff_seconds=0, breaker=breaker
            )
        assert len(calls) == 3  # normal retries happened
        assert not isinstance(exc_info.value, QuotaExceededError)
