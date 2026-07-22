"""
ModelService: the layer between FastAPI routes and the InferenceEngine.

Two responsibilities that matter for production-mindedness, not just wiring:

1. Concurrency control. A single model instance on a single GPU/CPU can only
   run one generate() call at a time without corrupting state or OOMing.
   `max_concurrent_generations` (default 1) is enforced with a plain
   threading.Semaphore, guarding both the streaming and non-streaming paths
   identically. This IS the "queueing" primitive the Section 4 write-up
   talks about -- it's implemented, not just described. Blocking calls are
   offloaded to FastAPI's threadpool (via starlette's run_in_threadpool) so
   the async event loop stays responsive (health checks etc. never block on
   a running generation).

2. Reusing Section 3's MetricsLogger. Every request is wrapped in
   `metrics.track()`, the exact same call pattern the benchmark runner used
   -- this is the "shared monitoring foundation" promised back in Section 3,
   cashed in here.
"""
import logging
import threading
import time
from typing import Iterator

from starlette.concurrency import run_in_threadpool

from src.engines.base import GenerationResult, InferenceEngine
from src.exceptions import ModelNotReadyError
from src.monitoring import MetricsLogger

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self, engine: InferenceEngine, metrics: MetricsLogger, max_concurrent_generations: int = 1):
        self.engine = engine
        self.metrics = metrics
        self.max_concurrent_generations = max_concurrent_generations
        self._gpu_lock = threading.Semaphore(max_concurrent_generations)
        self.ready = False
        self.load_error: str | None = None

        # Live stats for the /stats endpoint -- a plain lock is enough here,
        # every critical section below is a handful of integer/float ops.
        self._stats_lock = threading.Lock()
        self._active_requests = 0
        self._queue_depth = 0
        self._total_requests_served = 0
        self._total_requests_failed = 0
        self._total_generation_time_s = 0.0

    def _mark_queued(self) -> None:
        with self._stats_lock:
            self._queue_depth += 1

    def _mark_active(self) -> None:
        with self._stats_lock:
            self._queue_depth -= 1
            self._active_requests += 1

    def _mark_done(self, generation_time_s: float, failed: bool) -> None:
        with self._stats_lock:
            self._active_requests -= 1
            self._total_requests_served += 1
            if failed:
                self._total_requests_failed += 1
            self._total_generation_time_s += generation_time_s

    def get_stats(self) -> dict:
        with self._stats_lock:
            served = self._total_requests_served
            avg = (self._total_generation_time_s / served) if served else 0.0
            return {
                "active_requests": self._active_requests,
                "queue_depth": self._queue_depth,
                "total_requests_served": served,
                "total_requests_failed": self._total_requests_failed,
                "avg_generation_time_s": round(avg, 3),
                "max_concurrent_generations": self.max_concurrent_generations,
            }

    async def startup(self) -> None:
        try:
            await run_in_threadpool(self.engine.load)
            self.ready = True
            logger.info(
                "Model ready: %s (device=%s, precision=%s)",
                getattr(self.engine, "resolved_model_id", "?"),
                getattr(self.engine, "_device", "?"),
                self.engine.precision_label,
            )
        except Exception as exc:  # noqa: BLE001 - startup must not crash the whole process
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.ready = False
            logger.exception("Model failed to load at startup")

    async def shutdown(self) -> None:
        if self.ready:
            await run_in_threadpool(self.engine.unload)
        self.ready = False

    def _generate_blocking(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        self._mark_queued()
        with self._gpu_lock:
            self._mark_active()
            start = time.perf_counter()
            result = self.engine.generate(prompt, max_new_tokens)
            elapsed = time.perf_counter() - start
            self._mark_done(elapsed, failed=bool(result.error))
            return result

    async def generate(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        if not self.ready:
            raise ModelNotReadyError(self.load_error or "Model is not loaded yet.")

        with self.metrics.track("api_generate", prompt_chars=len(prompt), max_new_tokens=max_new_tokens) as ctx:
            result = await run_in_threadpool(self._generate_blocking, prompt, max_new_tokens)
            ctx["output_tokens"] = result.output_tokens
            ctx["generation_error"] = result.error
        return result

    def stream_generate_sync(self, prompt: str, max_new_tokens: int) -> Iterator[str]:
        """
        Sync generator -- deliberately NOT async. Starlette's StreamingResponse
        runs sync generators in its own threadpool automatically
        (starlette.concurrency.iterate_in_threadpool), so the semaphore below
        is a plain threading.Semaphore, consistent with _generate_blocking above.
        """
        if not self.ready:
            raise ModelNotReadyError(self.load_error or "Model is not loaded yet.")

        self._mark_queued()
        with self._gpu_lock:
            self._mark_active()
            start = time.perf_counter()
            failed = False
            token_count = 0
            try:
                with self.metrics.track("api_generate_stream", prompt_chars=len(prompt), max_new_tokens=max_new_tokens) as ctx:
                    for chunk in self.engine.generate_stream(prompt, max_new_tokens):
                        token_count += 1
                        yield chunk
                    ctx["output_tokens"] = token_count
            except Exception:
                failed = True
                raise
            finally:
                elapsed = time.perf_counter() - start
                self._mark_done(elapsed, failed=failed)
