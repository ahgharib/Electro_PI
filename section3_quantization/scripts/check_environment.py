"""
Run this first: python scripts/check_environment.py

Checks Python version, whether required packages are importable, whether a
CUDA GPU is visible and actually usable, and prints a plain verdict on which
--device / --precision combinations will work on this machine.
"""
import os
import platform
import sys


def print_cache_location():
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        print(f"HF_HOME is set -> model cache will go to: {hf_home}")
    elif platform.system() == "Windows":
        print(r"HF_HOME is NOT set -> model cache defaults to C:\Users\<you>\.cache\huggingface")
        print('  If C: is low on space, redirect it: $env:HF_HOME = "D:\\hf_cache"')
        print("  Or pass --hf-cache-dir D:\\hf_cache to run_benchmark.py / download_model.py")
    else:
        print("HF_HOME is NOT set -> model cache defaults to ~/.cache/huggingface")


def print_nightly_hint():
    print(
        "\nFallback install for very new NVIDIA GPUs (e.g. RTX 50-series) not yet\n"
        "supported by the stable PyTorch build:\n"
        "  pip install torch torchvision torchaudio --pre "
        "--extra-index-url https://download.pytorch.org/whl/nightly/cu128\n"
        "See README 'Troubleshooting: RTX 50-series GPUs' for details."
    )


def check() -> None:
    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print("WARNING: Python 3.10+ is expected. You may hit syntax errors (e.g. 'X | None' type hints).")

    print_cache_location()

    missing = []
    for pkg in ("torch", "transformers", "accelerate", "bitsandbytes", "huggingface_hub", "psutil"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"MISSING packages: {missing}")
        print("  -> Install PyTorch first (see README step 1), then: pip install -r requirements.txt")
    else:
        print("All required packages are importable.")

    try:
        import torch

        print(f"torch version: {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        print(f"CUDA available: {cuda_ok}")

        if cuda_ok:
            name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"GPU: {name} | compute capability: {major}.{minor} | VRAM: {vram_gb:.1f} GB")
            try:
                a = torch.zeros(2, 2, device="cuda")
                _ = a @ a  # sanity op -- catches "no kernel image is available" on very new architectures
                print("CUDA sanity check: OK")
            except RuntimeError as exc:
                print(f"CUDA sanity check FAILED: {exc}")
                print_nightly_hint()
        else:
            print("No CUDA GPU detected by PyTorch.")
            print("If you have an RTX 50-series (Blackwell) GPU, stable PyTorch may not yet support it.")
            print_nightly_hint()
    except ImportError:
        print("torch not installed -- install it first (see README step 1).")

    print("\nRecommended --device values on this machine:")
    print("  cpu   : always works (slow for fp16/bf16, unavailable for 4bit/8bit)")
    print("  gpu   : only if 'CUDA sanity check: OK' printed above")
    print("  auto  : picks the best available automatically (safe default)")


if __name__ == "__main__":
    check()
