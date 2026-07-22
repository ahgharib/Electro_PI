"""
Turns a load_test.json summary (as produced by scripts/load_test.py) into a
readable markdown report and a static HTML dashboard -- same "make raw JSON
readable" job as Section 3's benchmark dashboard, applied to load-test
results instead of precision comparisons.
"""
import json
from pathlib import Path


def _stats_table(stats: dict | None) -> str:
    if not stats:
        return "_N/A_"
    return (
        "| min | avg | median | p95 | max |\n"
        "|---|---|---|---|---|\n"
        f"| {stats.get('min')} | {stats.get('avg')} | {stats.get('median')} | {stats.get('p95')} | {stats.get('max')} |"
    )


def generate_markdown_report(summary: dict) -> str:
    lines = [
        "# Load Test Report",
        "",
        f"- Mode: {summary.get('mode', '?')}",
        f"- Concurrency: {summary.get('concurrency', '?')}",
        f"- Requests: {summary['n_requests']} ({summary['n_failed']} failed)",
        f"- Wall clock: {summary['wall_clock_s']}s",
        f"- Aggregate throughput: {summary['aggregate_throughput_tok_s']} tok/s",
        "",
        "## Total latency (s)",
        _stats_table(summary.get("total_latency_stats_s")),
        "",
        "## Time to first token (s)",
        _stats_table(summary.get("time_to_first_token_stats_s")),
        "",
        "## Per-request detail",
        "| # | Total latency (s) | TTFT (s) | Output tokens | Error |",
        "|---|---|---|---|---|",
    ]
    for r in summary.get("per_request", []):
        lines.append(
            f"| {r['request_id']} | {r['total_latency_s']} | {r['time_to_first_token_s']} | "
            f"{r['output_tokens']} | {r['error'] or ''} |"
        )
    return "\n".join(lines)


_DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Load Test Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#0f1117; color:#e6e6e6; margin:0; padding:32px; }
  h1 { font-weight: 600; }
  .cards { display:grid; grid-template-columns: repeat(4, 1fr); gap:16px; margin-top:24px; }
  .card { background:#181b24; border-radius:12px; padding:16px 20px; box-shadow:0 2px 8px rgba(0,0,0,.3); }
  .card .label { color:#9aa4b2; font-size:12px; text-transform:uppercase; letter-spacing:.05em; }
  .card .value { font-size:28px; font-weight:600; margin-top:4px; }
  .chart-card { background:#181b24; border-radius:12px; padding:20px; margin-top:24px; box-shadow:0 2px 8px rgba(0,0,0,.3); }
  table { width:100%; border-collapse:collapse; font-size:13px; margin-top:16px; }
  th, td { text-align:left; padding:6px 10px; border-bottom:1px solid #2a2e3a; }
  th { color:#9aa4b2; font-weight:500; }
  .err { color:#e3627a; }
</style>
</head>
<body>
<h1>Load Test Dashboard</h1>
<p style="color:#9aa4b2;">mode: __MODE__ &middot; concurrency: __CONCURRENCY__</p>

<div class="cards">
  <div class="card"><div class="label">Requests</div><div class="value" id="statRequests"></div></div>
  <div class="card"><div class="label">Failed</div><div class="value" id="statFailed"></div></div>
  <div class="card"><div class="label">Wall clock (s)</div><div class="value" id="statWall"></div></div>
  <div class="card"><div class="label">Throughput (tok/s)</div><div class="value" id="statThroughput"></div></div>
</div>

<div class="chart-card"><canvas id="latencyChart"></canvas></div>
<div class="chart-card" id="timelineCard" style="display:none;"><canvas id="timelineChart"></canvas></div>

<div class="chart-card">
  <table id="detailTable"></table>
</div>

<script>
const summary = __SUMMARY_JSON__;
const perRequest = __PER_REQUEST_JSON__;
const statsTimeline = __STATS_TIMELINE_JSON__;

document.getElementById('statRequests').textContent = summary.n_requests;
document.getElementById('statFailed').textContent = summary.n_failed;
document.getElementById('statWall').textContent = summary.wall_clock_s;
document.getElementById('statThroughput').textContent = summary.aggregate_throughput_tok_s;

const labels = perRequest.map(r => 'req ' + r.request_id);
new Chart(document.getElementById('latencyChart'), {
  type: 'line',
  data: {
    labels,
    datasets: [
      { label: 'Total latency (s)', data: perRequest.map(r => r.total_latency_s), borderColor: '#5b8def', backgroundColor: '#5b8def', tension: 0.2 },
      { label: 'Time to first token (s)', data: perRequest.map(r => r.time_to_first_token_s), borderColor: '#5be0a0', backgroundColor: '#5be0a0', tension: 0.2 }
    ]
  },
  options: {
    plugins: { title: { display: true, text: 'Per-request latency (the queueing "staircase")', color: '#e6e6e6' }, legend: { labels: { color: '#e6e6e6' } } },
    scales: { x: { ticks: { color: '#e6e6e6' } }, y: { ticks: { color: '#e6e6e6' }, title: { display: true, text: 'seconds', color: '#e6e6e6' } } }
  }
});

if (statsTimeline && statsTimeline.length > 0) {
  document.getElementById('timelineCard').style.display = 'block';
  new Chart(document.getElementById('timelineChart'), {
    type: 'line',
    data: {
      labels: statsTimeline.map(s => s.t.toFixed(1) + 's'),
      datasets: [
        { label: 'active_requests', data: statsTimeline.map(s => s.active_requests), borderColor: '#5b8def', backgroundColor: '#5b8def', stepped: true },
        { label: 'queue_depth', data: statsTimeline.map(s => s.queue_depth), borderColor: '#e3627a', backgroundColor: '#e3627a', stepped: true }
      ]
    },
    options: {
      plugins: { title: { display: true, text: 'Live /stats during the load test', color: '#e6e6e6' }, legend: { labels: { color: '#e6e6e6' } } },
      scales: { x: { ticks: { color: '#e6e6e6', maxTicksLimit: 15 } }, y: { ticks: { color: '#e6e6e6', stepSize: 1 }, beginAtZero: true } }
    }
  });
}

const table = document.getElementById('detailTable');
let html = '<tr><th>#</th><th>Total latency (s)</th><th>TTFT (s)</th><th>Output tokens</th><th>Error</th></tr>';
for (const r of perRequest) {
  html += `<tr><td>${r.request_id}</td><td>${r.total_latency_s ?? 'N/A'}</td><td>${r.time_to_first_token_s ?? 'N/A'}</td><td>${r.output_tokens ?? 'N/A'}</td><td class="err">${r.error ?? ''}</td></tr>`;
}
table.innerHTML = html;
</script>
</body>
</html>
"""


def generate_html_dashboard(summary: dict, output_path: str = "results/load_test_dashboard.html") -> str:
    per_request = summary.get("per_request", [])
    stats_timeline = summary.get("stats_timeline", [])

    html = (
        _DASHBOARD_TEMPLATE
        .replace("__MODE__", str(summary.get("mode", "?")))
        .replace("__CONCURRENCY__", str(summary.get("concurrency", "?")))
        .replace("__SUMMARY_JSON__", json.dumps(summary))
        .replace("__PER_REQUEST_JSON__", json.dumps(per_request))
        .replace("__STATS_TIMELINE_JSON__", json.dumps(stats_timeline))
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return str(out)
