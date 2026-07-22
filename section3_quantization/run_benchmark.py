"""
CLI entry point for Section 3 benchmarking.

Examples
--------
fp16 baseline, auto device selection:
    python run_benchmark.py --model Qwen/Qwen2.5-3B-Instruct --precision fp16 --device auto --output results/fp16.json

Force GPU explicitly (fails loudly if no CUDA GPU, rather than silently using CPU):
    python run_benchmark.py --model Qwen/Qwen2.5-3B-Instruct --precision fp16 --device gpu --output results/fp16_gpu.json

4-bit quantized (bitsandbytes, requires a CUDA GPU):
    python run_benchmark.py --model Qwen/Qwen2.5-3B-Instruct --precision 4bit --device gpu --output results/bnb_4bit.json

CPU-only run (works on any machine, including an examiner's with no GPU):
    python run_benchmark.py --model Qwen/Qwen2.5-3B-Instruct --precision fp16 --device cpu --output results/fp16_cpu.json

Compare saved reports + generate a viewable HTML dashboard:
    python run_benchmark.py --compare results/fp16.json results/bnb_4bit.json --dashboard
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Section 3 quantization benchmark runner")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--fallback-model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Used automatically if the primary model fails to load. Pass '' to disable.",
    )
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "4bit", "8bit"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--prompts", default="prompts/fixed_prompts.json")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--output", default="results/report.json")
    parser.add_argument("--compare", nargs="+", help="Skip running; print (and optionally chart) a comparison of saved reports")
    parser.add_argument("--dashboard", action="store_true", help="Generate an HTML dashboard (use with --compare)")
    parser.add_argument(
        "--hf-cache-dir",
        default=None,
        help="Redirect the Hugging Face model cache here (e.g. D:\\hf_cache) instead of the "
             "default C:\\Users\\<you>\\.cache\\huggingface. Must be set before any download happens.",
    )
    args = parser.parse_args()

    # Must happen BEFORE importing anything that touches huggingface_hub/transformers,
    # since they read HF_HOME at import time.
    if args.hf_cache_dir:
        os.environ["HF_HOME"] = args.hf_cache_dir
        logging.info("HF cache redirected to: %s", args.hf_cache_dir)

    from src.benchmark import BenchmarkRunner
    from src.engines.transformers_engine import HFTransformersEngine
    from src.exceptions import QuantizationBenchmarkError
    from src.monitoring import generate_dashboard

    if args.compare:
        BenchmarkRunner.print_comparison_table(args.compare)
        if args.dashboard:
            path = generate_dashboard(args.compare)
            print(f"\nDashboard written to: {path} (open it in a browser)")
        return

    prompts = json.loads(Path(args.prompts).read_text())
    engine = HFTransformersEngine(
        model_id=args.model,
        precision=args.precision,
        device_mode=args.device,
        fallback_model_id=(args.fallback_model or None),
    )
    runner = BenchmarkRunner(engine=engine, prompts=prompts, max_new_tokens=args.max_new_tokens)

    try:
        report = runner.run()
    except QuantizationBenchmarkError as exc:
        print(f"\nBenchmark failed: {exc}")
        sys.exit(1)

    runner.save_report(report, args.output)

    print(
        f"\nDone. precision={report['precision_label']} device={report['device']} "
        f"avg_tok/s={report['avg_tokens_per_second']} vram_peak={report['vram_peak_gb']}GB "
        f"fallback_triggered={report['auto_fallback_triggered']} "
        f"failed={report['n_failed']}/{report['n_prompts']}"
    )


if __name__ == "__main__":
    main()
