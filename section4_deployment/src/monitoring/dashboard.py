"""
Generates a single self-contained HTML dashboard (no server, no account,
no external service) from saved benchmark reports. Open the output file
directly in a browser.

This is the lightweight stand-in for something like LangSmith: appropriate
for MVP scope because it's one static file using Chart.js from a CDN, with
no infrastructure to run or maintain.
"""
import json
from pathlib import Path

_DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Quantization Benchmark Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#0f1117; color:#e6e6e6; margin:0; padding:32px; }
  h1 { font-weight: 600; }
  .grid { display:grid; grid-template-columns: 1fr 1fr; gap:24px; margin-top:24px; }
  .card { background:#181b24; border-radius:12px; padding:20px; box-shadow:0 2px 8px rgba(0,0,0,.3); }
  table { width:100%; border-collapse:collapse; font-size:14px; }
  th, td { text-align:left; padding:8px 10px; border-bottom:1px solid #2a2e3a; }
  th { color:#9aa4b2; font-weight:500; }
  .badge { display:inline-block; padding:2px 8px; border-radius:6px; font-size:12px; background:#2a2e3a; }
</style>
</head>
<body>
<h1>Quantization Benchmark Dashboard</h1>
<p class="badge">Generated from __N_REPORTS__ report(s)</p>
<div class="grid">
  <div class="card"><canvas id="tpsChart"></canvas></div>
  <div class="card"><canvas id="memChart"></canvas></div>
</div>
<div class="card" style="margin-top:24px;">
  <table id="detailTable"></table>
</div>
<script>
const reports = __REPORTS_JSON__;

const labels = reports.map(r => r.precision_label);
new Chart(document.getElementById('tpsChart'), {
  type: 'bar',
  data: { labels, datasets: [{ label: 'Avg tokens/sec', data: reports.map(r => r.avg_tokens_per_second), backgroundColor: '#5b8def' }] },
  options: { plugins: { title: { display: true, text: 'Throughput (tokens/sec)', color: '#e6e6e6' } },
             scales: { x: { ticks: { color: '#e6e6e6' } }, y: { ticks: { color: '#e6e6e6' } } } }
});

new Chart(document.getElementById('memChart'), {
  type: 'bar',
  data: { labels, datasets: [
    { label: 'VRAM peak (GB)', data: reports.map(r => r.vram_peak_gb ?? 0), backgroundColor: '#e3627a' },
    { label: 'RAM (GB)', data: reports.map(r => r.ram_used_gb ?? 0), backgroundColor: '#5be0a0' }
  ] },
  options: { plugins: { title: { display: true, text: 'Memory Footprint', color: '#e6e6e6' } },
             scales: { x: { ticks: { color: '#e6e6e6' } }, y: { ticks: { color: '#e6e6e6' } } } }
});

const table = document.getElementById('detailTable');
let html = '<tr><th>Precision</th><th>Device</th><th>Tok/s</th><th>VRAM peak</th><th>RAM</th><th>Fallback?</th><th>Failed</th></tr>';
for (const r of reports) {
  html += `<tr><td>${r.precision_label}</td><td>${r.device ?? '?'}</td><td>${r.avg_tokens_per_second}</td><td>${r.vram_peak_gb ?? 'N/A'}</td><td>${r.ram_used_gb}</td><td>${r.auto_fallback_triggered ? 'yes' : 'no'}</td><td>${r.n_failed}/${r.n_prompts}</td></tr>`;
}
table.innerHTML = html;
</script>
</body>
</html>
"""


def generate_dashboard(report_paths: list[str], output_path: str = "results/dashboard.html") -> str:
    reports = [json.loads(Path(p).read_text()) for p in report_paths]
    # Strip the heavy per-prompt "results" field -- the dashboard only needs the summary.
    slim = [{k: v for k, v in r.items() if k != "results"} for r in reports]

    html = _DASHBOARD_TEMPLATE.replace("__N_REPORTS__", str(len(reports))).replace(
        "__REPORTS_JSON__", json.dumps(slim)
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return str(out)
