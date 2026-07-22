import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring import generate_dashboard


def test_generate_dashboard_creates_html(tmp_path):
    report = {
        "precision_label": "fp16",
        "device": "cuda",
        "auto_fallback_triggered": False,
        "avg_tokens_per_second": 12.3,
        "vram_peak_gb": 5.5,
        "ram_used_gb": 1.2,
        "n_failed": 0,
        "n_prompts": 5,
        "results": [{"prompt": "x", "output": "y"}],  # should be stripped from the dashboard payload
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))

    out_path = tmp_path / "dashboard.html"
    result_path = generate_dashboard([str(report_path)], output_path=str(out_path))

    assert Path(result_path).exists()
    html = Path(result_path).read_text()
    assert "fp16" in html
    assert "chart.umd" in html
    assert '"results"' not in html  # heavy per-prompt data must be stripped before embedding


def test_generate_dashboard_handles_multiple_reports(tmp_path):
    reports = [
        {"precision_label": "fp16", "device": "cuda", "auto_fallback_triggered": False,
         "avg_tokens_per_second": 10.0, "vram_peak_gb": 6.0, "ram_used_gb": 1.0, "n_failed": 0, "n_prompts": 5, "results": []},
        {"precision_label": "4bit", "device": "cuda", "auto_fallback_triggered": False,
         "avg_tokens_per_second": 18.0, "vram_peak_gb": 2.5, "ram_used_gb": 1.0, "n_failed": 0, "n_prompts": 5, "results": []},
    ]
    paths = []
    for i, r in enumerate(reports):
        p = tmp_path / f"report_{i}.json"
        p.write_text(json.dumps(r))
        paths.append(str(p))

    out_path = tmp_path / "dashboard.html"
    generate_dashboard([str(p) for p in paths], output_path=str(out_path))
    html = out_path.read_text()
    assert "fp16" in html and "4bit" in html
