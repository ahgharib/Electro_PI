"""
MetricsLogger: append-only structured JSONL event log.

This is the shared monitoring foundation for BOTH sections:
  - Section 3 (this module) uses it to log model_load and generation events
    during offline benchmarking.
  - Section 4's FastAPI service will import this exact class and call
    log_event() / track() from request middleware, giving consistent,
    greppable, machine-parseable logs with zero extra infrastructure --
    no Prometheus/Grafana/LangSmith account needed for an MVP.

Each line in the log file is one JSON object: timestamp, event_type, duration,
error (if any), a resource snapshot, and whatever extra fields the caller adds.
That's enough to reconstruct a full timeline and feed the dashboard generator.
"""
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter, time
from typing import Any, Iterator

from .snapshot import ResourceMonitor

logger = logging.getLogger(__name__)


class MetricsLogger:
    def __init__(self, log_path: str = "results/metrics.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._monitor = ResourceMonitor()

    def log_event(self, event_type: str, **payload: Any) -> None:
        record = {"timestamp": time(), "event_type": event_type, **payload}
        try:
            with self.log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as exc:
            # Monitoring must never crash the workload it's observing.
            logger.error("Failed to write metrics log entry: %s", exc)

    @contextmanager
    def track(self, event_type: str, **extra_payload: Any) -> Iterator[dict]:
        """
        Wrap any block of work (a generation call, later an API request) and
        automatically log duration + resource snapshot on exit -- including
        on exceptions, so failures show up in the log too.

        Usage:
            with metrics.track("generation", prompt_index=3) as ctx:
                result = engine.generate(prompt)
                ctx["output_tokens"] = result.output_tokens
        """
        start = perf_counter()
        self._monitor.reset_peak()
        ctx: dict = {}
        error = None
        try:
            yield ctx
        except Exception as exc:  # noqa: BLE001 - recorded here, then re-raised for the caller to handle
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed = perf_counter() - start
            snapshot = self._monitor.snapshot()
            self.log_event(
                event_type,
                duration_s=round(elapsed, 4),
                error=error,
                resource_snapshot=snapshot.to_dict(),
                **extra_payload,
                **ctx,
            )

    def read_all(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        return [json.loads(line) for line in self.log_path.read_text().splitlines() if line.strip()]
