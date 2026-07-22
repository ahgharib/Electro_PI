"""
Settings read from environment variables. Deliberately NOT using
pydantic-settings here -- after the transformers 5.x API surprise in
Section 3, adding another fast-moving dependency for something this simple
isn't worth the version-skew risk. Plain env-var reads with defaults cover
everything this service needs.
"""
import os
from dataclasses import dataclass, field


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass
class Settings:
    model_id: str = field(default_factory=lambda: _env("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct"))
    fallback_model_id: str = field(default_factory=lambda: _env("FALLBACK_MODEL_ID", ""))
    precision: str = field(default_factory=lambda: _env("PRECISION", "fp16"))
    device_mode: str = field(default_factory=lambda: _env("DEVICE_MODE", "auto"))
    max_concurrent_generations: int = field(default_factory=lambda: int(_env("MAX_CONCURRENT_GENERATIONS", "1")))
    metrics_log_path: str = field(default_factory=lambda: _env("METRICS_LOG_PATH", "results/api_metrics.jsonl"))
    hf_cache_dir: str = field(default_factory=lambda: _env("HF_CACHE_DIR", ""))
