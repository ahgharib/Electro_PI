"""
Pre-downloads and caches the model(s) used by the benchmark, with retries on
transient network failures, so the actual benchmark run doesn't spend its
first minutes on a cold download.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --model Qwen/Qwen2.5-3B-Instruct
"""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# primary model used throughout Section 3 and (reused) Section 4, plus the
# smaller fallback model the engine auto-switches to if the primary fails
DEFAULT_MODELS = ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]


def download(model_id: str) -> None:
    from src.utils import retry
    from huggingface_hub import snapshot_download

    @retry(times=3, delay_s=5.0, exceptions=(Exception,))
    def _do_download():
        logger.info("Downloading %s ...", model_id)
        path = snapshot_download(repo_id=model_id)
        logger.info("Cached at: %s", path)

    _do_download()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument(
        "--hf-cache-dir",
        default=None,
        help="Redirect the Hugging Face model cache here (e.g. D:\\hf_cache) instead of the "
             "default C:\\Users\\<you>\\.cache\\huggingface.",
    )
    args = parser.parse_args()

    if args.hf_cache_dir:
        os.environ["HF_HOME"] = args.hf_cache_dir
        logger.info("HF cache redirected to: %s", args.hf_cache_dir)

    failures = []
    for model_id in args.model:
        try:
            download(model_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to download %s after retries: %s", model_id, exc)
            failures.append(model_id)

    if failures:
        print(f"\nFailed to download: {failures}. Check your internet connection / Hugging Face status.")
        sys.exit(1)

    print("\nAll models cached. You're ready to run run_benchmark.py.")


if __name__ == "__main__":
    main()
