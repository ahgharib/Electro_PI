"""Environment configuration and LLM provider wiring.

Design decision (see NOTES.md for the full write-up): the LLM is the one
piece of the pipeline that must be real per the task instructions, so unlike
STT/TTS it is never mocked. We use two free-tier, OpenAI-compatible
providers -- Groq as primary (fast, generous free tier, real tool-calling)
and Cerebras as an automatic fallback -- wired together with LiveKit's own
`llm.FallbackAdapter` rather than hand-rolled retry logic. Both providers
are reachable through `livekit-plugins-openai`, since their APIs are
OpenAI-compatible, so we don't need a plugin per vendor.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from livekit.agents import llm
from livekit.plugins import openai as lk_openai

logger = logging.getLogger("voice_agent.config")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
# Cerebras narrowed its self-serve/free-tier catalog in 2026; llama-4-scout
# and llama-3.3-70b were removed/deprecated from it (confirmed via a 404
# "model_not_found" from a real run -- see NOTES.md). gpt-oss-120b is the
# current production, free-tier, tool-calling-capable model on Cerebras.
DEFAULT_CEREBRAS_MODEL = "gpt-oss-120b"


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None
    groq_model: str
    cerebras_api_key: str | None
    cerebras_model: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY") or None,
            groq_model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            cerebras_api_key=os.getenv("CEREBRAS_API_KEY") or None,
            cerebras_model=os.getenv("CEREBRAS_MODEL", DEFAULT_CEREBRAS_MODEL),
        )


class ConfigurationError(RuntimeError):
    """Raised when no usable LLM provider is configured."""


def build_llm(settings: Settings | None = None) -> llm.LLM:
    """Build the LLM the agent will use, with automatic provider fallback.

    At least one of GROQ_API_KEY / CEREBRAS_API_KEY must be set. If both are
    set, Groq is tried first and the session transparently fails over to
    Cerebras if Groq errors or times out (see `llm.FallbackAdapter`). If
    only one is set, we use that one directly with no fallback wrapper --
    no point wrapping a single provider.
    """
    settings = settings or Settings.from_env()

    candidates: list[llm.LLM] = []

    if settings.groq_api_key:
        candidates.append(
            lk_openai.LLM(
                model=settings.groq_model,
                api_key=settings.groq_api_key,
                base_url=GROQ_BASE_URL,
            )
        )
        logger.info("llm_provider_configured", extra={"provider": "groq", "model": settings.groq_model})

    if settings.cerebras_api_key:
        candidates.append(
            lk_openai.LLM.with_cerebras(
                model=settings.cerebras_model,
                api_key=settings.cerebras_api_key,
            )
        )
        logger.info(
            "llm_provider_configured",
            extra={"provider": "cerebras", "model": settings.cerebras_model},
        )

    if not candidates:
        raise ConfigurationError(
            "No LLM provider configured. Set GROQ_API_KEY and/or CEREBRAS_API_KEY "
            "in your environment (see .env.example). Both have free tiers that "
            "need no credit card -- console.groq.com/keys and cloud.cerebras.ai."
        )

    if len(candidates) == 1:
        return candidates[0]

    logger.info("llm_fallback_chain_built", extra={"providers": len(candidates)})
    return llm.FallbackAdapter(
        candidates,
        attempt_timeout=10.0,
        # A real run surfaced this: Groq can stream a partial text chunk and
        # then fail while generating a tool call. The adapter's default
        # (retry_on_chunk_sent=False) refuses to retry once any output has
        # already been sent downstream, to avoid duplicated/garbled speech --
        # sensible for a live audio stream, but it means the whole request
        # errors out unrecoverably instead of failing over. For this MVP
        # (text-modality, no live audio to garble) reliability matters more
        # than that edge case, so we opt in to retrying anyway.
        retry_on_chunk_sent=True,
    )
