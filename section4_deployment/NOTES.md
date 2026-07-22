# Section 4 — Model Deployment: Write-up

## Setup
- Reuses Section 3's `HFTransformersEngine` unchanged (same fallback mechanisms, same device modes) plus a new `generate_stream()` method for token streaming.
- Model: Qwen2.5-1.5B-Instruct, CPU (Docker default — see "Load test results" below).
- API: FastAPI, chosen over vLLM/TGI — justification below.
- Added after initial load testing: a `GET /stats` endpoint for live monitoring (`active_requests`, `queue_depth`, `total_requests_served`, `avg_generation_time_s`), instrumenting the same semaphore that already controls concurrency — not a separate bookkeeping system.

## Why FastAPI instead of vLLM/TGI
FastAPI is sufficient for a small deployment and allows full control over the inference pipeline. 
vLLM's CPU backend is experimental and not reliable to build in a 10-minute setup; TGI is Docker-first but still primarily GPU-oriented. FastAPI wrapping Section 3's already-tested, device-portable engine reuses real code instead of a rewrite, and satisfies the portability requirement directly. vLLM/TGI are the correct answer for *production throughput under real concurrency* (continuous batching, PagedAttention) — see the 50-user write-up below for where they'd actually get used.

## Load test results (10 concurrent requests, streaming, CPU/Docker, Qwen2.5-1.5B-Instruct)

| Metric | Value |
|---|---|
| Requests | 10 (0 failed) |
| Wall clock | 123.66s |
| Aggregate throughput | 5.26 tok/s |
| Total latency — min / avg / median / p95 / max | 14.26 / 68.91 / 68.87 / 123.65 / 123.65 s |
| Time-to-first-token — min / avg / median / p95 / max | 2.15 / 57.30 / 57.37 / 111.89 / 111.89 s |

Full per-request detail and charts: `results/load_test_report.md` / `results/load_test_dashboard.html`.

**Interpretation:** with `max_concurrent_generations=1`, all 10 requests were fully serialized on a single CPU worker — total latency for the last-served request (123.65s) is close to the full wall clock, and each request's latency roughly reflects its position in the FIFO queue behind the semaphore (visible as a "staircase" in the dashboard chart). Every successful request produced exactly 65 output tokens — this is expected, not a bug: generation is deterministic (`do_sample=False`, chosen in Section 3 for reproducible benchmarking), and every concurrent request used the same default prompt, so identical input reliably produces identical output length. It's also useful evidence that the semaphore correctly isolates each generation with no cross-request state corruption under concurrency.

## Write-up: how would this change to serve 50 concurrent users in production?

At 10 concurrent requests, a single-worker FastAPI wrapper processing one generation at a time (our `max_concurrent_generations=1` semaphore) is survivable — worst-case queue depth is roughly 10× a single request's latency, which is a defensible MVP result, and CPU/Docker numbers above already show this concretely: 123.65s worst-case for just 10 users. **At 50 concurrent requests this same design breaks down hard**: the 50th request in line could wait 10+ minutes on CPU, or several minutes even on GPU — a real UX failure.

Here is what i would do:

- **Continuous batching** — swap the naive one-at-a-time FastAPI+transformers loop for vLLM or TGI. Both let multiple requests' forward passes share GPU compute in the same batch instead of running serially, which is where the real throughput gain comes from at this scale.
- **Request queueing with backpressure** — replace the unbounded semaphore wait with an actual bounded queue that returns a 429/503 once it's full, rather than letting wait times grow unbounded. Users get a fast "try again" instead of a silent multi-minute hang. `/stats` endpoint's `queue_depth` is exactly the signal you'd wire an autoscaler or a backpressure policy to.
- **Horizontal scaling** — once a single GPU's batched throughput ceiling is hit, run multiple model replicas behind a load balancer.
- **Autoscaling** — Kubernetes HPA (or a serverless GPU platform) to add/remove replicas based on queue depth or GPU utilization, rather than provisioning for peak load permanently.
- **Caching** — if many requests share a system prompt or common prefix, prefix/KV-cache reuse (supported natively by vLLM) avoids redundant compute.

**Bottom line:** 10 concurrent users is a queueing problem solvable with a semaphore; 50 concurrent users is a throughput problem that requires a real batching inference server plus horizontal scaling. The gap between them is architectural, not incremental — and it's visible in miniature in our own 10-user CPU numbers above.

## Known limitations

- Docker image defaults to CPU + the smaller 1.5B model for guaranteed portability.
- `max_concurrent_generations` defaults to 1 (single model instance, single worker) — deliberately conservative rather than allowing concurrent `generate()` calls to race for VRAM/RAM.
- Identical output length (65 tokens) across all 10 load-test requests is a consequence of using one fixed default prompt with deterministic (greedy) decoding — not a defect. Varying the prompt per request would produce varying output lengths.
