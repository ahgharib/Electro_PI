# Section 3 — Quantization: Write-up

## Setup
- Model: Qwen2.5-3B-Instruct (fp16 baseline), bitsandbytes NF4 4-bit (quantized)
- Hardware: Asus TUF Gaming F16 (2025), Intel i7, RTX 5050 Laptop GPU (8GB VRAM), 16GB RAM, Windows 11
- Ran via: `python run_benchmark.py --precision fp16 --device gpu`, `--precision 4bit --device gpu`, and `--precision fp16 --device cpu` (portability check)

## Results

| Precision | Device | Avg tok/s | VRAM peak (GB) | RAM (GB) | Fallback? | Failed |
|---|---|---|---|---|---|---|
| fp16 | cuda | 23.24 | 5.784 | 2.078 | no | 0/5 |
| 4bit (bnb NF4) | cuda | 14.17 | 2.002 | 1.190 | no | 0/5 |
| fp16 | cpu | 1.44 | 0 (N/A) | 11.532 | no | 0/5 |

See `results/dashboard.html` for the charts.

**Sanity checks that confirm these numbers are real, not noise:**
- VRAM dropped 65% under 4-bit (5.78 → 2.00 GB) — matches the expected ~4x weight compression (some overhead from activations/cache keeps it from being a clean 4x).
- CPU RAM usage (11.53 GB) is almost exactly 3B params × 4 bytes (fp32, since no CUDA means no fp16 kernels) ≈ 12 GB — matches theory.
- CPU throughput is ~16x slower than GPU fp16 (23.24 / 1.44 ≈ 16.1x) — consistent with expected CPU-vs-GPU gap for a 3B model.

## Qualitative comparison (5 fixed prompts, 1-5 rubric: coherence / correctness / instruction-following)

| # | Prompt type | fp16 score | 4-bit score | Notes |
|---|---|---|---|---|
| 1 | Explanation (what is quantization) | 5 | 5 | Both correct, coherent, comparable depth. |
| 2 | Code generation (memoized Fibonacci) | 5 | 5 | Both produced correct, idiomatic memoized implementations; near-identical structure. |
| 3 | Reasoning/math (bags of apples/oranges) | 5 | 3 | Both reach the correct reasoning (4 apple bags, working toward 5 orange bags), but the 4-bit output has a stray non-English character injected mid-sentence ("24 apples and 野20 oranges") — a direct, observed quantization artifact. |
| 4 | Summarization | 5 | 5 | Both accurate, appropriately reworded single-sentence summaries. |
| 5 | Creative (2-line couplet) | 5 | 3 | fp16's couplet genuinely rhymes ("light"/"night"). 4-bit's does not ("shines"/"own") — a real, if minor, quality regression on the creative task specifically. |

**Pattern observed:** quantization-induced quality loss showed up specifically on the two tasks requiring more precise token-level control (exact character generation in the math answer, exact rhyme in the couplet) — not on the more "average-case" tasks like explanation, code, and summarization. This tracks with what's generally reported about low-bit quantization: fluency and factual recall degrade less than precise, low-entropy output requirements.

## Write-up: when would you pick GPTQ/AWQ over bitsandbytes, or GGUF over both, for production?

- **The measured result that anchors this answer:** in this benchmark, 4-bit (bitsandbytes NF4) was *slower* than fp16 (14.17 vs 23.24 tok/s) despite using 65% less VRAM.
bitsandbytes quantizes weights for storage but **dequantizes them back to fp16/bf16 on every forward pass**, adding compute overhead that isn't offset by memory-bandwidth savings for a model this small on a GPU this capable. bitsandbytes optimizes for *memory footprint*, not throughput.
- **bitsandbytes** is still the right choice for fast iteration and fine-tuning (QLoRA) — no separate calibration/conversion step, one flag, and it let me benchmark fp16 vs quantized in the same script. That's exactly why I used it here, in a 5-7 day window.
- **GPTQ/AWQ** produce a static, pre-quantized checkpoint with dedicated low-bit inference kernels (ExLlama/vLLM) that skip the dequantize-then-matmul pattern — this is where you'd actually expect a *speed* win, which is what a production serving workload needs. AWQ tends to preserve accuracy slightly better than GPTQ at the same bit-width because it protects activation-salient weight channels instead of quantizing everything uniformly — directly relevant given the quality artifacts observed above at 4-bit.
- **GGUF/llama.cpp** is the right call when the deployment target doesn't have a reliable CUDA GPU (edge devices, CPU-only servers, offline/local apps) or when you need a single portable binary without a Python/CUDA dependency chain, specifically for its portability on machines.
- **Bottom line:** bitsandbytes = fastest to experiment with, best for dev/fine-tuning; GPTQ/AWQ = best for GPU-serving throughput in production; GGUF = best for CPU/edge/portable deployment. The measured throughput regression here is the clearest evidence for why you wouldn't ship bitsandbytes 4-bit as-is behind a high-throughput API.

## Known limitations / debugging notes

- No fallback mechanisms were triggered during the final runs (`auto_fallback_triggered: false` on all three reports) — the OOM→4-bit and model-fallback-chain logic exist as safety nets but weren't needed once the model/precision combination was sized correctly for the 8GB card.
- Not an easy Setup for Different Devices, Due to Cuda requirements Changing depending on the device.
- 4-bit is not usable without a GPU.
- Whlie 4-bit is better for VRAM it is slower than a normal f16 which is a trad-off between Storage and Speed.
