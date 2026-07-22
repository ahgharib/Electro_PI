import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring import MetricsLogger, ResourceMonitor


def test_resource_monitor_snapshot_has_ram():
    monitor = ResourceMonitor()
    snap = monitor.snapshot()
    assert snap.ram_used_gb > 0


def test_metrics_logger_writes_jsonl(tmp_path):
    log_path = tmp_path / "metrics.jsonl"
    logger = MetricsLogger(str(log_path))
    logger.log_event("test_event", foo="bar")

    assert log_path.exists()
    lines = log_path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event_type"] == "test_event"
    assert record["foo"] == "bar"


def test_metrics_logger_track_context_manager(tmp_path):
    log_path = tmp_path / "metrics.jsonl"
    logger = MetricsLogger(str(log_path))

    with logger.track("generation", prompt_index=1) as ctx:
        ctx["output_tokens"] = 42

    records = logger.read_all()
    assert len(records) == 1
    assert records[0]["event_type"] == "generation"
    assert records[0]["output_tokens"] == 42
    assert records[0]["error"] is None
    assert "duration_s" in records[0]


def test_metrics_logger_track_records_errors_and_reraises(tmp_path):
    log_path = tmp_path / "metrics.jsonl"
    logger = MetricsLogger(str(log_path))

    raised = False
    try:
        with logger.track("generation", prompt_index=2):
            raise ValueError("boom")
    except ValueError:
        raised = True

    assert raised, "the original exception must still propagate to the caller"
    records = logger.read_all()
    assert len(records) == 1
    assert "boom" in records[0]["error"]


def test_metrics_logger_read_all_empty_when_no_file(tmp_path):
    logger = MetricsLogger(str(tmp_path / "does_not_exist.jsonl"))
    assert logger.read_all() == []
