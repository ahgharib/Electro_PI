import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.load_test import summarize


def test_summarize_computes_stats_correctly():
    results = [
        {"request_id": i, "total_latency_s": t, "time_to_first_token_s": t / 2, "output_tokens": 50, "error": None}
        for i, t in enumerate([1.0, 2.0, 3.0, 4.0, 5.0])
    ]
    summary = summarize(results, wall_elapsed=5.0)
    assert summary["n_requests"] == 5
    assert summary["n_failed"] == 0
    assert summary["total_latency_stats_s"]["avg"] == 3.0
    assert summary["total_latency_stats_s"]["min"] == 1.0
    assert summary["total_latency_stats_s"]["max"] == 5.0


def test_summarize_handles_failures():
    results = [
        {"request_id": 0, "total_latency_s": 1.0, "time_to_first_token_s": 0.5, "output_tokens": 10, "error": None},
        {"request_id": 1, "total_latency_s": 0.1, "time_to_first_token_s": None, "output_tokens": None, "error": "TimeoutError: boom"},
    ]
    summary = summarize(results, wall_elapsed=1.0)
    assert summary["n_requests"] == 2
    assert summary["n_failed"] == 1
    # only the successful request's latency contributes to stats
    assert summary["total_latency_stats_s"]["avg"] == 1.0


def test_summarize_computes_aggregate_throughput():
    results = [
        {"request_id": i, "total_latency_s": 2.0, "time_to_first_token_s": 1.0, "output_tokens": 100, "error": None}
        for i in range(10)
    ]
    summary = summarize(results, wall_elapsed=4.0)  # 10 requests * 100 tokens = 1000 tokens / 4s wall clock
    assert summary["aggregate_throughput_tok_s"] == 250.0


def test_summarize_returns_none_ttft_stats_when_all_missing():
    results = [
        {"request_id": 0, "total_latency_s": 1.0, "time_to_first_token_s": None, "output_tokens": 10, "error": None},
    ]
    summary = summarize(results, wall_elapsed=1.0)
    assert summary["time_to_first_token_stats_s"] is None
