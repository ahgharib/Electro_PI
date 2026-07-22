"""
Lightweight resource monitor: RAM always, VRAM if a CUDA GPU is present.
No external monitoring stack needed for MVP scope -- gives real, loggable
numbers per run, which is what the task explicitly asks for.
"""
from dataclasses import dataclass, asdict
from time import time

import psutil


@dataclass
class ResourceSnapshot:
    timestamp: float
    ram_used_gb: float
    vram_used_gb: float | None  # None if no CUDA GPU available
    vram_peak_gb: float | None

    def to_dict(self) -> dict:
        return asdict(self)


class ResourceMonitor:
    def __init__(self):
        self._cuda_available = self._check_cuda()

    @staticmethod
    def _check_cuda() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def snapshot(self) -> ResourceSnapshot:
        ram_gb = psutil.Process().memory_info().rss / (1024 ** 3)
        vram_used, vram_peak = None, None
        if self._cuda_available:
            import torch

            vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
            vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        return ResourceSnapshot(
            timestamp=time(),
            ram_used_gb=round(ram_gb, 3),
            vram_used_gb=round(vram_used, 3) if vram_used is not None else None,
            vram_peak_gb=round(vram_peak, 3) if vram_peak is not None else None,
        )

    def reset_peak(self) -> None:
        if self._cuda_available:
            import torch

            torch.cuda.reset_peak_memory_stats()
