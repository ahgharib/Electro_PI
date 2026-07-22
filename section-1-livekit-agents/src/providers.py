"""STT/TTS provider selection.

This is what turns Task 1.2 (swap a pipeline component) from "here's a
code snippet of what it would look like" into an actual, working switch:
`AgentSession` was already provider-agnostic (it just takes any `stt.STT`
/ `tts.TTS` instance -- that part needed no changes at all). What this
module adds is a small **registry** of builder functions per component,
selected by an env var: each provider is one isolated function plus one
line registering it, so adding provider #3 never means editing the logic
for providers #1 and #2 (open/closed, not a growing if/elif chain).
Mirrors `config.py`'s `build_llm()` in spirit.

See SWAPPING_PROVIDERS.md for a full walkthrough -- both swapping the
provider behind an existing slot and adding a brand new one -- with real,
verified constructor signatures for four different providers.

Honesty note (see NOTES.md): this makes the swap mechanically real and
testable-for-construction -- these factories are exercised by
tests/test_providers.py, proving they build the right object with the
right arguments. What's still NOT verified is a live conversation against
real provider audio, for the same reason build_llm() couldn't be verified
against a real LLM from this environment: no network route to those
providers and no API key available here. That step needs a human with a
(free/trial) key, same as the LLM evidence transcript did.
"""

from __future__ import annotations

import logging
import os
from typing import Callable

from livekit.agents import stt, tts

from config import ConfigurationError
from mock_providers import MockSTT, MockTTS

logger = logging.getLogger("voice_agent.providers")

DEFAULT_DEEPGRAM_MODEL = "nova-3"
DEFAULT_CARTESIA_VOICE = "f786b574-daa5-4673-aa0c-cbe3e8534c02"  # Cartesia's default demo voice


def _require_api_key(env_var: str, provider_env: str, provider_name: str, signup_url: str) -> str:
    api_key = os.getenv(env_var)
    if not api_key:
        raise ConfigurationError(
            f"{provider_env}={provider_name} needs {env_var} set in .env "
            f"(free trial credit, no card: {signup_url})."
        )
    return api_key


def _require_plugin(import_path: str, module_name: str):
    try:
        return __import__(import_path, fromlist=[module_name])
    except ImportError as e:
        raise ConfigurationError(
            "This provider needs the plugin installed: "
            "pip install -r requirements-optional-providers.txt"
        ) from e


# --- STT builders -----------------------------------------------------

def _mock_stt() -> stt.STT:
    return MockSTT()


def _deepgram_stt() -> stt.STT:
    api_key = _require_api_key(
        "DEEPGRAM_API_KEY", "STT_PROVIDER", "deepgram", "console.deepgram.com"
    )
    deepgram = _require_plugin("livekit.plugins.deepgram", "deepgram")
    logger.info("stt_provider_configured", extra={"provider": "deepgram"})
    return deepgram.STT(api_key=api_key, model=DEFAULT_DEEPGRAM_MODEL, language="en-US")


# Registry: provider name -> zero-arg builder. Add a new STT provider by
# writing one `_<name>_stt()` function above and registering it here --
# nothing else in this file changes. See SWAPPING_PROVIDERS.md.
_STT_BUILDERS: dict[str, Callable[[], stt.STT]] = {
    "mock": _mock_stt,
    "deepgram": _deepgram_stt,
}


def build_stt() -> stt.STT:
    """Build the STT the agent will use, selected by the STT_PROVIDER env
    var (default 'mock'). Supported providers: see _STT_BUILDERS above.
    """
    provider = os.getenv("STT_PROVIDER", "mock").lower()
    builder = _STT_BUILDERS.get(provider)
    if builder is None:
        raise ConfigurationError(
            f"Unknown STT_PROVIDER '{provider}'. Supported: {', '.join(_STT_BUILDERS)}."
        )
    return builder()


# --- TTS builders -----------------------------------------------------

def _mock_tts() -> tts.TTS:
    return MockTTS()


def _cartesia_tts() -> tts.TTS:
    api_key = _require_api_key(
        "CARTESIA_API_KEY", "TTS_PROVIDER", "cartesia", "play.cartesia.ai"
    )
    cartesia = _require_plugin("livekit.plugins.cartesia", "cartesia")
    logger.info("tts_provider_configured", extra={"provider": "cartesia"})
    return cartesia.TTS(api_key=api_key, voice=DEFAULT_CARTESIA_VOICE)


# Registry: provider name -> zero-arg builder. Add a new TTS provider by
# writing one `_<name>_tts()` function above and registering it here --
# nothing else in this file changes. See SWAPPING_PROVIDERS.md.
_TTS_BUILDERS: dict[str, Callable[[], tts.TTS]] = {
    "mock": _mock_tts,
    "cartesia": _cartesia_tts,
}


def build_tts() -> tts.TTS:
    """Build the TTS the agent will use, selected by the TTS_PROVIDER env
    var (default 'mock'). Supported providers: see _TTS_BUILDERS above.
    """
    provider = os.getenv("TTS_PROVIDER", "mock").lower()
    builder = _TTS_BUILDERS.get(provider)
    if builder is None:
        raise ConfigurationError(
            f"Unknown TTS_PROVIDER '{provider}'. Supported: {', '.join(_TTS_BUILDERS)}."
        )
    return builder()
