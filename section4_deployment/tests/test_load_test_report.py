import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.load_test_report import generate_html_dashboard, generate_markdown_report

SAMPLE_SUMMARY = {
    "n_requests": 2,
    "n_failed": 0,
    "wall_clock_s": 10.0,
    "aggregate_throughput_tok_s": 12.5,
    "total_latency_stats_s": {"min": 4.0, "avg": 5.0, "median": 5.0, "p95": 6.0, "max": 6.0},
    "time_to_first_token_stats_s": {"min": 2.0, "avg": 2.5, "median": 2.5, "p95": 3.0, "max": 3.0},
    "mode": "stream",
    "concurrency": 2,
    "stats_timeline": [
        {"active_requests": 1, "queue_depth": 1, "total_requests_served": 0, "t": 0.1},
        {"active_requests": 1, "queue_depth": 0, "total_requests_served": 1, "t": 0.6},
    ],
    "per_request": [
        {"request_id": 0, "total_latency_s": 4.0, "time_to_first_token_s": 2.0, "output_tokens": 30, "error": None},
        {"request_id": 1, "total_latency_s": 6.0, "time_to_first_token_s": 3.0, "output_tokens": 30, "error": None},
    ],
}


def test_generate_markdown_report_includes_key_numbers():
    md = generate_markdown_report(SAMPLE_SUMMARY)
    assert "# Load Test Report" in md
    assert "12.5" in md  # throughput
    assert "req" not in md.split("\n")[0]  # sanity: header line is clean
    assert "| 0 | 4.0 | 2.0 | 30 |" in md
    assert "| 1 | 6.0 | 3.0 | 30 |" in md


def test_generate_html_dashboard_creates_file_and_embeds_data(tmp_path):
    out_path = tmp_path / "dashboard.html"
    result_path = generate_html_dashboard(SAMPLE_SUMMARY, output_path=str(out_path))
    assert Path(result_path).exists()
    html = Path(result_path).read_text()
    assert "chart.umd" in html
    assert '"aggregate_throughput_tok_s": 12.5' in html
    assert "queue_depth" in html  # timeline chart present since stats_timeline is non-empty


def test_generate_html_dashboard_handles_missing_timeline(tmp_path):
    summary = {**SAMPLE_SUMMARY, "stats_timeline": []}
    out_path = tmp_path / "dashboard.html"
    generate_html_dashboard(summary, output_path=str(out_path))
    html = out_path.read_text()
    assert "timelineCard" in html  # element exists but stays hidden via the empty-array check in JS
