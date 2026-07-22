import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.device import _resolve
from src.exceptions import DeviceUnavailableError


def test_auto_prefers_cuda_when_available():
    assert _resolve("auto", cuda_available=True) == "cuda"


def test_auto_falls_back_to_cpu_when_no_cuda():
    assert _resolve("auto", cuda_available=False) == "cpu"


def test_cpu_mode_ignores_gpu_presence():
    assert _resolve("cpu", cuda_available=True) == "cpu"
    assert _resolve("cpu", cuda_available=False) == "cpu"


def test_gpu_mode_returns_cuda_when_available():
    assert _resolve("gpu", cuda_available=True) == "cuda"


def test_gpu_mode_raises_clear_error_when_unavailable():
    with pytest.raises(DeviceUnavailableError):
        _resolve("gpu", cuda_available=False)


def test_invalid_mode_raises_value_error():
    with pytest.raises(ValueError):
        _resolve("quantum", cuda_available=True)
