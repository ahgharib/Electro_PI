"""
Resolves the user-facing --device mode (auto / cpu / gpu) into an actual
torch device string.

Semantics (deliberate, not arbitrary):
  auto -> "cuda" if a GPU is visible, else "cpu". Never fails -- this is the
          convenience default for anyone just trying the code out.
  cpu  -> always "cpu", even if a GPU is present. Useful for controlled
          CPU-only benchmarking, or for an examiner who wants to verify the
          code runs without touching their GPU.
  gpu  -> "cuda" if available, else raises DeviceUnavailableError. This does
          NOT silently fall back to CPU: if you explicitly asked for GPU
          numbers, a silent CPU fallback would quietly invalidate the
          benchmark without telling you.

The resolution logic itself (_resolve) takes cuda_available as a plain bool
so it can be unit tested with no torch import and no GPU required -- see
tests/test_device.py.
"""
import logging
from enum import Enum

from .exceptions import DeviceUnavailableError

logger = logging.getLogger(__name__)


class DeviceMode(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


def _resolve(mode: str, cuda_available: bool) -> str:
    """Pure logic, no I/O -- fully unit-testable."""
    mode = DeviceMode(mode)

    if mode == DeviceMode.CPU:
        return "cpu"

    if mode == DeviceMode.GPU:
        if not cuda_available:
            raise DeviceUnavailableError(
                "Requested --device gpu but no CUDA GPU was detected by PyTorch.\n"
                "  1. Run `nvidia-smi` -- is your GPU listed?\n"
                "  2. Run `python scripts/check_environment.py` for a full diagnostic.\n"
                "  3. If you have an RTX 50-series (Blackwell) GPU and stable PyTorch "
                "doesn't detect it, see README 'Troubleshooting: RTX 50-series GPUs' "
                "for the CUDA 12.8 nightly install command."
            )
        return "cuda"

    # auto
    if cuda_available:
        return "cuda"
    logger.warning(
        "No CUDA GPU detected -- auto mode is falling back to CPU. "
        "fp16/bf16 will be slow; 4bit/8bit precisions are unavailable on CPU."
    )
    return "cpu"


def resolve_device(mode: str) -> str:
    """Real entry point: detects CUDA via torch, then delegates to the pure resolver."""
    import torch

    return _resolve(mode, cuda_available=torch.cuda.is_available())
