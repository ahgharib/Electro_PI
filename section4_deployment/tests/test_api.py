"""
Tests use a FakeEngine (same philosophy as Section 3's tests) so the whole
API surface -- health, generate, streaming, error handling -- can be
verified with no GPU, no model download, no network.
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from app.main import create_app
from app.model_service import ModelService
from src.engines.base import InferenceEngine
from src.monitoring import MetricsLogger


class FakeEngine(InferenceEngine):
    def __init__(self, fail_generation: bool = False, delay_s: float = 0.0):
        self.precision_label = "fake"
        self.model_id = "fake/model"
        self.resolved_model_id = "fake/model"
        self._device = "cpu"
        self.auto_fallback_triggered = False
        self.fallback_reason = None
        self.fail_generation = fail_generation
        self.delay_s = delay_s
        self.loaded = False

    def load(self):
        self.loaded = True

    def _generate_raw(self, prompt, max_new_tokens):
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail_generation:
            raise RuntimeError("simulated generation failure")
        text = f"echo: {prompt}"
        return text, len(text.split())

    def generate_stream(self, prompt, max_new_tokens):
        if self.fail_generation:
            raise RuntimeError("simulated streaming failure")
        for word in f"echo: {prompt}".split():
            yield word + " "

    def unload(self):
        self.loaded = False


class FailingLoadEngine(FakeEngine):
    def load(self):
        raise RuntimeError("simulated load failure")


def make_client(engine: InferenceEngine, tmp_path: Path, max_concurrent: int = 1) -> TestClient:
    metrics = MetricsLogger(str(tmp_path / "metrics.jsonl"))
    service = ModelService(engine=engine, metrics=metrics, max_concurrent_generations=max_concurrent)
    app = create_app(service)
    return TestClient(app)


def test_health_reports_not_ready_before_startup(tmp_path):
    client = make_client(FakeEngine(), tmp_path)
    # No "with" block -> lifespan startup never runs -> service.ready stays False
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["ready"] is False


def test_health_reports_ready_after_startup(tmp_path):
    client = make_client(FakeEngine(), tmp_path)
    with client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ready"] is True
        assert body["model_id"] == "fake/model"
        assert body["device"] == "cpu"


def test_health_reports_not_ready_when_load_fails(tmp_path):
    client = make_client(FailingLoadEngine(), tmp_path)
    with client:
        resp = client.get("/health")
        assert resp.json()["ready"] is False


def test_generate_returns_expected_fields(tmp_path):
    client = make_client(FakeEngine(), tmp_path)
    with client:
        resp = client.post("/generate", json={"prompt": "hello", "max_new_tokens": 50})
        assert resp.status_code == 200
        body = resp.json()
        assert body["output"] == "echo: hello"
        assert body["output_tokens"] == 2
        assert body["precision"] == "fake"
        assert body["device"] == "cpu"
        assert body["tokens_per_second"] > 0


def test_generate_returns_503_before_model_ready(tmp_path):
    client = make_client(FakeEngine(), tmp_path)
    resp = client.post("/generate", json={"prompt": "hello", "max_new_tokens": 50})
    assert resp.status_code == 503


def test_generate_returns_500_on_generation_failure(tmp_path):
    client = make_client(FakeEngine(fail_generation=True), tmp_path)
    with client:
        resp = client.post("/generate", json={"prompt": "hello", "max_new_tokens": 50})
        assert resp.status_code == 500


def test_generate_rejects_empty_prompt(tmp_path):
    client = make_client(FakeEngine(), tmp_path)
    with client:
        resp = client.post("/generate", json={"prompt": "", "max_new_tokens": 50})
        assert resp.status_code == 422  # pydantic validation, min_length=1


def test_generate_stream_returns_chunks_and_done_marker(tmp_path):
    client = make_client(FakeEngine(), tmp_path)
    with client:
        resp = client.post("/generate/stream", json={"prompt": "hello world", "max_new_tokens": 50})
        assert resp.status_code == 200
        body = resp.text
        assert "echo:" in body
        assert body.strip().endswith("data: [DONE]")


def test_generate_stream_reports_error_as_sse_event_not_a_crash(tmp_path):
    client = make_client(FakeEngine(fail_generation=True), tmp_path)
    with client:
        resp = client.post("/generate/stream", json={"prompt": "hello", "max_new_tokens": 50})
        assert resp.status_code == 200  # stream already started -- error surfaces as an SSE event, not an HTTP error
        assert "[ERROR]" in resp.text


def test_metrics_are_logged_for_generate_requests(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics = MetricsLogger(str(metrics_path))
    service = ModelService(engine=FakeEngine(), metrics=metrics, max_concurrent_generations=1)
    app = create_app(service)
    with TestClient(app) as client:
        client.post("/generate", json={"prompt": "hello", "max_new_tokens": 50})

    records = metrics.read_all()
    event_types = [r["event_type"] for r in records]
    assert "api_generate" in event_types


def test_stats_endpoint_reports_zero_when_idle(tmp_path):
    client = make_client(FakeEngine(), tmp_path)
    with client:
        resp = client.get("/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["active_requests"] == 0
        assert body["queue_depth"] == 0
        assert body["total_requests_served"] == 0
        assert body["total_requests_failed"] == 0


def test_stats_endpoint_increments_after_requests(tmp_path):
    client = make_client(FakeEngine(), tmp_path)
    with client:
        client.post("/generate", json={"prompt": "hello", "max_new_tokens": 10})
        client.post("/generate", json={"prompt": "world", "max_new_tokens": 10})
        resp = client.get("/stats")
        body = resp.json()
        assert body["total_requests_served"] == 2
        assert body["active_requests"] == 0  # both finished by the time we check
        assert body["avg_generation_time_s"] >= 0


def test_stats_tracks_failures_separately(tmp_path):
    client = make_client(FakeEngine(fail_generation=True), tmp_path)
    with client:
        client.post("/generate", json={"prompt": "hello", "max_new_tokens": 10})
        body = client.get("/stats").json()
        assert body["total_requests_served"] == 1
        assert body["total_requests_failed"] == 1


def test_stats_show_real_queueing_under_concurrent_load(tmp_path):
    """Proves the semaphore actually queues requests -- not just that the code compiles."""
    client = make_client(FakeEngine(delay_s=0.3), tmp_path, max_concurrent=1)
    with client:
        def fire(prompt):
            client.post("/generate", json={"prompt": prompt, "max_new_tokens": 10})

        t1 = threading.Thread(target=fire, args=("first",))
        t2 = threading.Thread(target=fire, args=("second",))
        t1.start()
        time.sleep(0.05)  # let the first request acquire the semaphore and start generating
        t2.start()
        time.sleep(0.1)  # second request should now be queued behind the first

        mid_stats = client.get("/stats").json()
        assert mid_stats["active_requests"] == 1
        assert mid_stats["queue_depth"] == 1

        t1.join()
        t2.join()

        final_stats = client.get("/stats").json()
        assert final_stats["active_requests"] == 0
        assert final_stats["queue_depth"] == 0
        assert final_stats["total_requests_served"] == 2
