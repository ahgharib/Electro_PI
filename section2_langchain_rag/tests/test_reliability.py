from __future__ import annotations

import pytest

from src.application.exceptions import ComponentError, NoProviderAvailableError
from src.application.reliability import CircuitBreaker, call_with_fallback, retry_with_backoff


class TestRetryWithBackoff:
    def test_succeeds_on_first_try(self):
        breaker = CircuitBreaker(failure_threshold=3, reset_seconds=30)
        calls = []

        def fn():
            calls.append(1)
            return "ok"

        result = retry_with_backoff(
            fn, component="x", provider="p", max_attempts=3, backoff_seconds=0, breaker=breaker
        )
        assert result == "ok"
        assert len(calls) == 1

    def test_retries_then_succeeds(self):
        breaker = CircuitBreaker(failure_threshold=5, reset_seconds=30)
        attempts = {"count": 0}

        def fn():
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise RuntimeError("transient failure")
            return "recovered"

        result = retry_with_backoff(
            fn, component="x", provider="p", max_attempts=5, backoff_seconds=0, breaker=breaker
        )
        assert result == "recovered"
        assert attempts["count"] == 3

    def test_raises_component_error_after_exhausting_attempts(self):
        breaker = CircuitBreaker(failure_threshold=10, reset_seconds=30)

        def always_fails():
            raise RuntimeError("permanent failure")

        with pytest.raises(ComponentError):
            retry_with_backoff(
                always_fails, component="x", provider="p", max_attempts=3, backoff_seconds=0, breaker=breaker
            )


class TestCircuitBreaker:
    def test_opens_after_threshold_failures(self):
        breaker = CircuitBreaker(failure_threshold=2, reset_seconds=30)
        assert breaker.is_open() is False
        breaker.record_failure()
        assert breaker.is_open() is False
        breaker.record_failure()
        assert breaker.is_open() is True

    def test_success_resets_failure_count(self):
        breaker = CircuitBreaker(failure_threshold=2, reset_seconds=30)
        breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()
        assert breaker.is_open() is False  # only 1 consecutive failure since reset

    def test_open_breaker_rejects_calls_without_invoking_fn(self):
        breaker = CircuitBreaker(failure_threshold=1, reset_seconds=9999)
        breaker.record_failure()  # trips the breaker
        assert breaker.is_open() is True

        calls = []

        def fn():
            calls.append(1)
            return "should not run"

        with pytest.raises(ComponentError):
            retry_with_backoff(
                fn, component="x", provider="p", max_attempts=3, backoff_seconds=0, breaker=breaker
            )
        assert len(calls) == 0  # breaker rejected before fn was ever called


class TestFallback:
    def test_falls_back_when_primary_exhausted(self):
        primary_breaker = CircuitBreaker(failure_threshold=1, reset_seconds=30)
        fallback_breaker = CircuitBreaker(failure_threshold=1, reset_seconds=30)

        def primary_fn():
            raise RuntimeError("primary down")

        def fallback_fn():
            return "fallback result"

        result = call_with_fallback(
            primary_fn,
            fallback_fn,
            component="generator",
            primary_provider="gemini",
            fallback_provider="openai",
            max_attempts=1,
            backoff_seconds=0,
            primary_breaker=primary_breaker,
            fallback_breaker=fallback_breaker,
        )
        assert result == "fallback result"

    def test_raises_no_provider_available_when_both_fail(self):
        primary_breaker = CircuitBreaker(failure_threshold=1, reset_seconds=30)
        fallback_breaker = CircuitBreaker(failure_threshold=1, reset_seconds=30)

        def always_fails():
            raise RuntimeError("down")

        with pytest.raises(NoProviderAvailableError):
            call_with_fallback(
                always_fails,
                always_fails,
                component="generator",
                primary_provider="gemini",
                fallback_provider="openai",
                max_attempts=1,
                backoff_seconds=0,
                primary_breaker=primary_breaker,
                fallback_breaker=fallback_breaker,
            )

    def test_no_fallback_configured_raises_component_error(self):
        primary_breaker = CircuitBreaker(failure_threshold=1, reset_seconds=30)

        def always_fails():
            raise RuntimeError("down")

        with pytest.raises(ComponentError):
            call_with_fallback(
                always_fails,
                None,
                component="generator",
                primary_provider="gemini",
                fallback_provider=None,
                max_attempts=1,
                backoff_seconds=0,
                primary_breaker=primary_breaker,
                fallback_breaker=None,
            )
