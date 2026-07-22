"""
Fires N concurrent requests at a running instance of the API and reports
time-to-first-token (streaming mode) and total latency, aggregated into
min/avg/median/p95/max -- the load/latency test Task 4.1 asks for.

Usage:
    # Non-streaming: measures total request latency only
    python scripts/load_test.py --url http://localhost:8000 --concurrency 10

    # Streaming: also measures real time-to-first-token
    python scripts/load_test.py --url http://localhost:8000 --concurrency 10 --stream
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_PROMPT = "Explain what quantization means for a large language model, in 2 sentences."


async def run_single_request(
    client: httpx.AsyncClient, url: str, prompt: str, max_new_tokens: int, stream: bool, request_id: int, timeout: float
) -> dict:
    start = time.perf_counter()
    first_byte_time = None
    output_tokens = None
    error = None
    total_time = None

    try:
        if stream:
            token_count = 0
            async with client.stream(
                "POST", f"{url}/generate/stream",
                json={"prompt": prompt, "max_new_tokens": max_new_tokens},
                timeout=timeout,
            ) as resp:
                resp.raise_for_status()
                # SSE framing: lines are "data: <chunk>", one chunk per generated token,
                # ending with "data: [DONE]". aiter_lines() respects line boundaries even
                # when HTTP chunks split mid-line, unlike aiter_text().
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[len("data: "):]
                    if first_byte_time is None:
                        first_byte_time = time.perf_counter()
                    if payload == "[DONE]":
                        break
                    if payload.startswith("[ERROR]"):
                        error = payload
                        break
                    token_count += 1
            output_tokens = token_count
            total_time = time.perf_counter() - start
        else:
            resp = await client.post(
                f"{url}/generate", json={"prompt": prompt, "max_new_tokens": max_new_tokens}, timeout=timeout
            )
            resp.raise_for_status()
            data = resp.json()
            output_tokens = data.get("output_tokens")
            total_time = time.perf_counter() - start
    except Exception as exc:  # noqa: BLE001 - record any failure (timeout, connection refused, 5xx) per-request
        error = f"{type(exc).__name__}: {exc}"
        total_time = time.perf_counter() - start

    if stream:
        ttft = (first_byte_time - start) if first_byte_time else None
    else:
        ttft = total_time if error is None else None  # non-streaming: no meaningful TTFT distinct from total latency

    return {
        "request_id": request_id,
        "total_latency_s": round(total_time, 3) if total_time is not None else None,
        "time_to_first_token_s": round(ttft, 3) if ttft is not None else None,
        "output_tokens": output_tokens,
        "error": error,
    }


async def _poll_stats(client: httpx.AsyncClient, url: str, interval: float, stop_event: asyncio.Event) -> list[dict]:
    """Polls GET /stats every `interval` seconds until stop_event is set, so the
    dashboard can show active_requests/queue_depth changing live during the run
    -- not just inferred after the fact from per-request latency."""
    snapshots = []
    start = time.perf_counter()
    while not stop_event.is_set():
        try:
            resp = await client.get(f"{url}/stats", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                data["t"] = round(time.perf_counter() - start, 3)
                snapshots.append(data)
        except Exception:  # noqa: BLE001 - /stats may not exist on older servers; timeline just stays empty
            pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
    return snapshots


async def run_load_test(
    url: str, concurrency: int, prompt: str, max_new_tokens: int, stream: bool, timeout: float,
    poll_stats: bool = True, stats_interval: float = 0.5,
):
    async with httpx.AsyncClient() as client:
        stop_event = asyncio.Event()
        stats_task = asyncio.create_task(_poll_stats(client, url, stats_interval, stop_event)) if poll_stats else None

        tasks = [
            run_single_request(client, url, prompt, max_new_tokens, stream, i, timeout) for i in range(concurrency)
        ]
        wall_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        wall_elapsed = time.perf_counter() - wall_start

        stats_timeline = []
        if stats_task:
            stop_event.set()
            stats_timeline = await stats_task

    return results, wall_elapsed, stats_timeline


def _stats(values: list[float]) -> dict:
    if not values:
        return {}
    sorted_v = sorted(values)
    p95_idx = min(len(sorted_v) - 1, int(len(sorted_v) * 0.95))
    return {
        "min": round(min(values), 3),
        "avg": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "p95": round(sorted_v[p95_idx], 3),
        "max": round(max(values), 3),
    }


def summarize(results: list[dict], wall_elapsed: float) -> dict:
    ok = [r for r in results if not r["error"]]
    latencies = [r["total_latency_s"] for r in ok if r["total_latency_s"] is not None]
    ttfts = [r["time_to_first_token_s"] for r in ok if r["time_to_first_token_s"] is not None]
    total_tokens = sum(r["output_tokens"] or 0 for r in ok)

    return {
        "n_requests": len(results),
        "n_failed": len(results) - len(ok),
        "wall_clock_s": round(wall_elapsed, 3),
        "aggregate_throughput_tok_s": round(total_tokens / wall_elapsed, 3) if wall_elapsed > 0 else 0.0,
        "total_latency_stats_s": _stats(latencies),
        "time_to_first_token_stats_s": _stats(ttfts) if ttfts else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--stream", action="store_true", help="Use the streaming endpoint (measures real time-to-first-token)")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-request timeout in seconds. Raise this for slow (e.g. CPU, serialized) backends.")
    parser.add_argument("--no-stats-poll", action="store_true", help="Disable polling GET /stats during the run")
    parser.add_argument("--stats-interval", type=float, default=0.5, help="Seconds between /stats polls")
    parser.add_argument("--report", action="store_true", help="Also generate load_test_report.md and load_test_dashboard.html")
    parser.add_argument("--output", default="results/load_test.json")
    args = parser.parse_args()

    results, wall_elapsed, stats_timeline = asyncio.run(
        run_load_test(
            args.url, args.concurrency, args.prompt, args.max_new_tokens, args.stream, args.timeout,
            poll_stats=not args.no_stats_poll, stats_interval=args.stats_interval,
        )
    )
    summary = summarize(results, wall_elapsed)
    summary["mode"] = "stream" if args.stream else "non-stream"
    summary["concurrency"] = args.concurrency
    summary["stats_timeline"] = stats_timeline
    summary["per_request"] = results

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_request", "stats_timeline")}, indent=2))
    print(f"\nSaved full results to {out_path}")

    if args.report:
        from src.monitoring.load_test_report import generate_html_dashboard, generate_markdown_report

        out_dir = out_path.parent
        md_path = out_dir / "load_test_report.md"
        md_path.write_text(generate_markdown_report(summary))
        html_path = generate_html_dashboard(summary, output_path=str(out_dir / "load_test_dashboard.html"))
        print(f"Saved report to {md_path}")
        print(f"Saved dashboard to {html_path}")


if __name__ == "__main__":
    main()
