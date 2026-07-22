"""Centralized configuration.

Every tunable in the system lives here, loaded from environment variables
(via a .env file for local dev). This is the ONLY place that should ever
contain a model name, threshold, or file path -- adapters and the pipeline
receive their settings through this object rather than hardcoding them.

Swapping providers, tuning the relevance threshold, changing chunk size,
etc. should always be a .env edit, never a code edit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    return int(val) if val not in (None, "") else default


def _env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    return float(val) if val not in (None, "") else default


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None or val == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# Rough $ per 1K tokens (input, output). Gemini free tier is $0 in practice,
# but the pricing table is kept real/updatable so cost tracking is not
# hardcoded to zero -- swapping to a paid provider or paid tier makes the
# cost field immediately meaningful with no code change. Update as pricing
# changes; treat these as approximate and override via .env if needed.
DEFAULT_PRICING_PER_1K: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.0, 0.0),          # free tier
    "gemini-2.5-flash-lite": (0.0, 0.0),     # free tier
    "gemini-embedding-001": (0.0, 0.0),      # free tier
    "gpt-4o-mini": (0.00015, 0.0006),
    "text-embedding-3-small": (0.00002, 0.0),
}


@dataclass
class Config:
    # --- providers ---
    embedding_provider: str = field(default_factory=lambda: _env_str("EMBEDDING_PROVIDER", "gemini"))
    generation_provider: str = field(default_factory=lambda: _env_str("GENERATION_PROVIDER", "gemini"))
    fallback_embedding_provider: str = field(
        default_factory=lambda: _env_str("FALLBACK_EMBEDDING_PROVIDER", "")
    )
    fallback_generation_provider: str = field(
        default_factory=lambda: _env_str("FALLBACK_GENERATION_PROVIDER", "")
    )

    # --- API keys ---
    google_api_key: str = field(default_factory=lambda: _env_str("GOOGLE_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: _env_str("OPENAI_API_KEY", ""))

    # --- model names (override freely as providers deprecate/rename models) ---
    gemini_chat_model: str = field(default_factory=lambda: _env_str("GEMINI_CHAT_MODEL", "gemini-2.5-flash"))
    gemini_embedding_model: str = field(
        default_factory=lambda: _env_str("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    )
    openai_chat_model: str = field(default_factory=lambda: _env_str("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    openai_embedding_model: str = field(
        default_factory=lambda: _env_str("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # --- chunking ---
    chunk_size_tokens: int = field(default_factory=lambda: _env_int("CHUNK_SIZE_TOKENS", 600))
    chunk_overlap_tokens: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP_TOKENS", 80))

    # --- ingestion / embedding request pacing ---
    # These exist because free-tier embedding quotas (RPM/RPD) are easy to
    # exceed on larger document sets, especially across repeated ingest runs
    # during development. embedding_batch_size caps how many chunks go into
    # one embed_documents() call (one physical API request each, roughly);
    # embedding_rpm_limit, if set above 0, proactively paces those calls
    # instead of only reacting to a 429 after the fact. 0 = no pacing.
    embedding_batch_size: int = field(default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 50))
    embedding_rpm_limit: int = field(default_factory=lambda: _env_int("EMBEDDING_RPM_LIMIT", 0))
    skip_unchanged_chunks_on_ingest: bool = field(
        default_factory=lambda: _env_bool("SKIP_UNCHANGED_CHUNKS_ON_INGEST", True)
    )

    # --- retrieval ---
    retrieval_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_K", 4))
    relevance_threshold: float = field(default_factory=lambda: _env_float("RELEVANCE_THRESHOLD", 0.55))
    enable_neighbor_expansion: bool = field(
        default_factory=lambda: _env_bool("ENABLE_NEIGHBOR_EXPANSION", False)
    )

    # --- token budgets / guardrails ---
    question_token_limit: int = field(default_factory=lambda: _env_int("QUESTION_TOKEN_LIMIT", 300))
    context_token_budget: int = field(default_factory=lambda: _env_int("CONTEXT_TOKEN_BUDGET", 6000))

    # --- vector store ---
    persist_directory: str = field(default_factory=lambda: _env_str("CHROMA_PERSIST_DIR", "./chroma_db"))
    collection_name: str = field(default_factory=lambda: _env_str("CHROMA_COLLECTION", "rag_docs"))

    # --- reliability ---
    retry_max_attempts: int = field(default_factory=lambda: _env_int("RETRY_MAX_ATTEMPTS", 2))
    retry_backoff_seconds: float = field(default_factory=lambda: _env_float("RETRY_BACKOFF_SECONDS", 1.0))
    circuit_breaker_failure_threshold: int = field(
        default_factory=lambda: _env_int("CIRCUIT_BREAKER_FAILURE_THRESHOLD", 3)
    )
    circuit_breaker_reset_seconds: float = field(
        default_factory=lambda: _env_float("CIRCUIT_BREAKER_RESET_SECONDS", 30.0)
    )

    # --- monitoring ---
    log_dir: str = field(default_factory=lambda: _env_str("LOG_DIR", "./logs"))
    verbose_console: bool = field(default_factory=lambda: _env_bool("VERBOSE_CONSOLE", True))
    debug_retrieval: bool = field(default_factory=lambda: _env_bool("DEBUG_RETRIEVAL", False))

    # --- docs ---
    docs_dir: str = field(default_factory=lambda: _env_str("DOCS_DIR", "./docs"))

    pricing_per_1k: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_PRICING_PER_1K)
    )

    def cost_for(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        input_price, output_price = self.pricing_per_1k.get(model, (0.0, 0.0))
        return (prompt_tokens / 1000.0) * input_price + (completion_tokens / 1000.0) * output_price
