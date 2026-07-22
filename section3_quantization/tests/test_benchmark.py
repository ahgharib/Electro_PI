"""
Tests use a FakeEngine so they run anywhere (no GPU, no model download, no
network) -- this validates the harness logic itself, independent of any
specific model or hardware.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark import BenchmarkRunner
from src.engines.base import InferenceEngine


class FakeEngine(InferenceEngine):
    """Deterministic stand-in for a real model, with a knob to simulate failures."""

    def __init__(self, precision_label="fake", fail_on_prompt_index=None, tokens_per_prompt=10, seconds_per_prompt=0.0):
        self.precision_label = precision_label
        self.model_id = "fake/model"
        self.resolved_model_id = "fake/model"
        self._device = "cpu"
        self.auto_fallback_triggered = False
        self.fallback_reason = None
        self.fail_on_prompt_index = fail_on_prompt_index
        self.tokens_per_prompt = tokens_per_prompt
        self.seconds_per_prompt = seconds_per_prompt
        self._call_count = 0
        self.loaded = False

    def load(self):
        self.loaded = True

    def _generate_raw(self, prompt, max_new_tokens):
        self._call_count += 1
        if self.fail_on_prompt_index == self._call_count:
            raise RuntimeError("simulated failure")
        time.sleep(self.seconds_per_prompt)
        return f"echo: {prompt[:10]}", self.tokens_per_prompt

    def unload(self):
        self.loaded = False


def test_generation_result_tokens_per_second():
    engine = FakeEngine(tokens_per_prompt=20, seconds_per_prompt=0.01)
    engine.load()
    result = engine.generate("hello world", max_new_tokens=50)
    assert result.output_tokens == 20
    assert result.tokens_per_second > 0
    assert result.error is None


def test_engine_context_manager_loads_and_unloads():
    engine = FakeEngine()
    assert not engine.loaded
    with engine:
        assert engine.loaded
    assert not engine.loaded


def test_benchmark_runner_aggregates_correctly(tmp_path):
    prompts = ["p1", "p2", "p3"]
    engine = FakeEngine(precision_label="fake-4bit", tokens_per_prompt=10)
    runner = BenchmarkRunner(engine=engine, prompts=prompts, max_new_tokens=32, metrics_log_path=str(tmp_path / "m.jsonl"))
    report = runner.run()

    assert report["precision_label"] == "fake-4bit"
    assert report["device"] == "cpu"
    assert report["n_prompts"] == 3
    assert report["n_failed"] == 0
    assert len(report["results"]) == 3


def test_benchmark_runner_handles_partial_failure(tmp_path):
    prompts = ["p1", "p2", "p3"]
    engine = FakeEngine(fail_on_prompt_index=2)  # second call fails
    runner = BenchmarkRunner(engine=engine, prompts=prompts, metrics_log_path=str(tmp_path / "m.jsonl"))
    report = runner.run()

    assert report["n_failed"] == 1
    assert report["results"][1]["error"] is not None
    assert report["results"][0]["error"] is None
    assert report["results"][2]["error"] is None


def test_benchmark_runner_writes_metrics_log(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    engine = FakeEngine()
    runner = BenchmarkRunner(engine=engine, prompts=["only prompt"], metrics_log_path=str(metrics_path))
    runner.run()

    assert metrics_path.exists()
    lines = metrics_path.read_text().splitlines()
    # 1 model_load event + 1 generation event
    assert len(lines) == 2


def test_save_and_reload_report(tmp_path):
    engine = FakeEngine()
    runner = BenchmarkRunner(engine=engine, prompts=["only prompt"], metrics_log_path=str(tmp_path / "m.jsonl"))
    report = runner.run()

    out_file = tmp_path / "report.json"
    BenchmarkRunner.save_report(report, str(out_file))
    assert out_file.exists()

    import json

    reloaded = json.loads(out_file.read_text())
    assert reloaded["precision_label"] == report["precision_label"]
