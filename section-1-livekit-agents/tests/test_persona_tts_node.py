"""Tests for SupportAgent.tts_node -- the hard-guarantee sanitization layer.

`Agent.default.tts_node` (the SDK's real synthesis delegate) requires an
active `AgentActivity`, which only exists inside a running `AgentSession`
with a real audio output attached. These tests stub that delegate out so
we can verify OUR override's logic (sanitize each chunk, then delegate) in
isolation, without needing a full audio/room pipeline. What this does NOT
test: whether `Agent.default.tts_node` itself behaves correctly -- that's
the SDK's own code, already covered by LiveKit's own test suite, not ours.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from livekit.agents import Agent  # noqa: E402
from livekit.agents.voice import ModelSettings  # noqa: E402
from persona import SupportAgent  # noqa: E402


async def _text_stream(*chunks: str):
    for chunk in chunks:
        yield chunk


@pytest.fixture
def captured_delegate_input(monkeypatch):
    """Stubs Agent.default.tts_node and records exactly what text it
    received, so tests can assert on SupportAgent.tts_node's sanitization
    without needing a running AgentActivity."""
    captured: list[str] = []

    async def fake_default_tts_node(agent, text, model_settings):
        async for chunk in text:
            captured.append(chunk)
        return
        yield  # pragma: no cover -- makes this a valid async generator

    monkeypatch.setattr(Agent.default, "tts_node", fake_default_tts_node)
    return captured


class TestSupportAgentTTSNode:
    @pytest.mark.asyncio
    async def test_strips_emoji_and_markdown_before_delegating(
        self, captured_delegate_input
    ) -> None:
        agent = SupportAgent()
        async for _ in agent.tts_node(
            _text_stream("Sure! 🎉 Your **order** is on its way!", " #done ✅"),
            ModelSettings(tool_choice="auto"),
        ):
            pass

        assert captured_delegate_input == ["Sure! Your order is on its way!", "done"]

    @pytest.mark.asyncio
    async def test_chunk_that_sanitizes_to_empty_is_dropped_not_forwarded(
        self, captured_delegate_input
    ) -> None:
        agent = SupportAgent()
        async for _ in agent.tts_node(
            _text_stream("🎉", "Order A100 is preparing"),
            ModelSettings(tool_choice="auto"),
        ):
            pass

        # the emoji-only chunk sanitizes to "" and must not be forwarded as
        # an empty synthesis request
        assert captured_delegate_input == ["Order A100 is preparing"]

    @pytest.mark.asyncio
    async def test_clean_text_passes_through_unchanged(self, captured_delegate_input) -> None:
        agent = SupportAgent()
        text = "Order A100 is currently preparing, ETA 25 minutes."
        async for _ in agent.tts_node(_text_stream(text), ModelSettings(tool_choice="auto")):
            pass

        assert captured_delegate_input == [text]
