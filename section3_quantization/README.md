# Section 3 — Quantization

Compares **Qwen2.5-3B-Instruct** at full precision (fp16) against
bitsandbytes 4-bit (NF4) quantization, with real measured VRAM/RAM,
tokens/sec, and output quality on a fixed 5-prompt set.

> **Why this model / this technique:** Qwen2.5-3B-Instruct's fp16 baseline
> (~6GB) fits comfortably in an 8GB-VRAM laptop GPU, leaving headroom for KV
> cache. bitsandbytes NF4 was chosen over GPTQ/AWQ because it requires no
> separate offline calibration pass — one flag, fully reproducible in the
> same script as the baseline. GPTQ/AWQ/GGUF trade-offs are discussed
> (not implemented) in `NOTES.md`.

## 1. Install PyTorch (hardware-specific)

**Standard (most machines):**
```bash
pip install torch torchvision torchaudio
```

**RTX 50-series (Blackwell) / CUDA 12.8 nightly fallback** — only needed if
the standard install doesn't detect your GPU (see Troubleshooting below):
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```
**In Case the the command above didn't work**
```bash
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu128
# or
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall --no-cache-dir
```
**CPU-only machine (examiner without a GPU):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 2. Install everything else

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Verify your environment

```bash
python scripts/check_environment.py
```
Tells you plainly which `--device` values will work on this machine, and
whether your GPU actually passes a CUDA sanity check (catches the "detected
but unusable" case some very new GPUs hit on stable PyTorch).

## 4. Sanity-check the code itself (no GPU/model needed)

```bash
pytest tests/ -v
```
19 tests validate device resolution, the monitoring system, the dashboard
generator, and the full benchmark harness using a fake engine — proving the
logic is correct independent of any specific model or hardware. Anyone
without a GPU can run this and confirm the engineering is sound.

## 5. Pre-download the model (recommended)

```bash
python scripts/download_model.py
```
Downloads both the primary model (Qwen2.5-3B-Instruct) and the smaller
fallback model (Qwen2.5-1.5B-Instruct) with automatic retry on network
errors. If skipped, the first benchmark run downloads on demand instead.

## 6. Run the benchmarks

fp16 baseline, auto device selection (uses GPU if present, else CPU):
```bash
python run_benchmark.py --model Qwen/Qwen2.5-3B-Instruct --precision fp16 --device auto --output results/fp16.json
```

4-bit quantized (requires a CUDA GPU — bitsandbytes has no CPU kernel):
```bash
python run_benchmark.py --model Qwen/Qwen2.5-3B-Instruct --precision 4bit --device gpu --output results/bnb_4bit.json
```

Force CPU explicitly (works on any machine, useful to prove portability):
```bash
python run_benchmark.py --model Qwen/Qwen2.5-3B-Instruct --precision fp16 --device cpu --max-new-tokens 32 --output results/fp16_cpu.json
```

### The three `--device` modes

| Mode | Behavior |
|---|---|
| `auto` | Uses GPU if detected, else CPU. Never fails. Safe default. |
| `cpu` | Always CPU, even if a GPU is present. For controlled CPU-only runs. |
| `gpu` | Requires a CUDA GPU; **raises a clear error instead of silently using CPU** if none is found — so a forced-GPU run never quietly produces misleading CPU numbers. |

## 7. Compare results + generate a dashboard

```bash
python run_benchmark.py --compare results/fp16.json results/bnb_4bit.json --dashboard
```
Prints a markdown table (paste into `NOTES.md`) and writes
`results/dashboard.html` — a single self-contained file (Chart.js via CDN,
no server, no account) you can open directly in a browser to see throughput
and memory charts side by side.

## Fallback mechanisms (built in, not incidental)

1. **CUDA OOM → auto-downgrade to 4-bit.** If loading fp16/bf16 runs out of
   VRAM, the engine automatically retries once at 4-bit instead of crashing,
   and flags this in the report (`auto_fallback_triggered: true`). Keeps a
   tight-VRAM run alive instead of failing outright.
2. **Model fallback chain.** If the primary model id fails to load for any
   other reason (network issue, bad id), the engine retries with a smaller,
   known-good fallback model (`Qwen2.5-1.5B-Instruct` by default). Disable
   with `--fallback-model ""` if you want a strict fail-loud run.
3. **Device fallback is intentionally NOT automatic for `--device gpu`** —
   see the table above. Silent fallback there would misreport results.

## Monitoring system

`src/monitoring/` is a small, reusable package
helper:
- `ResourceMonitor` — RAM always, VRAM if CUDA is present.
- `MetricsLogger` — append-only structured JSONL event log with a
  `track()` context manager that records duration + resource snapshot for
  any block of work, including on exceptions.
- `generate_dashboard()` — turns saved JSON reports into a static HTML
  dashboard.

All results land in `results/`: `*.json` reports, `metrics.jsonl` (raw event
log), `dashboard.html`.

### Results
![Load test dashboard preview](section3_quantization/results/Section3.png)
---

## Project structure

```
section3_quantization/
├── run_benchmark.py              # CLI entry point
├── requirements.txt
├── scripts/
│   ├── check_environment.py      # run first: diagnoses Python/torch/CUDA setup
│   └── download_model.py         # pre-downloads models with retry
├── src/
│   ├── exceptions.py             # custom exception hierarchy
│   ├── device.py                 # auto/cpu/gpu resolution (pure logic + torch wrapper)
│   ├── utils.py                  # retry decorator
│   ├── benchmark.py              # BenchmarkRunner: ties engine + monitoring together
│   ├── engines/
│   │   ├── base.py               # abstract InferenceEngine contract
│   │   └── transformers_engine.py  # fp16/bf16 + bitsandbytes 4bit/8bit, with fallbacks
│   └── monitoring/
│       ├── snapshot.py           # ResourceMonitor / ResourceSnapshot
│       ├── metrics_logger.py     # MetricsLogger -- shared base, reused by Section 4
│       └── dashboard.py          # static HTML dashboard generator
├── prompts/fixed_prompts.json    # the 5 fixed prompts, same for every run
├── tests/                        # 19 tests, no GPU/model/network needed
│   ├── test_device.py
│   ├── test_monitoring.py
│   ├── test_dashboard.py
│   └── test_benchmark.py
├── results/                      # JSON reports, metrics.jsonl, dashboard.html land here
└── NOTES.md                      # write-up (fill in after running)
```

## Design notes / trade-offs

- `do_sample=False` in generation: deterministic output, so fp16-vs-4bit
  comparison isn't confounded by sampling randomness.
