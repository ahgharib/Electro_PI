"""End-to-end integration test: real `AgentSession` + real `SupportAgent` +
real `MockSTT`/`MockTTS`, with `ScriptedLLM` standing in for a network LLM.

This is deliberately NOT the "real LLM invoking a tool" evidence the task
requires -- see `transcripts/README.md` for how to generate that against
Groq/Cerebras. What this proves is narrower but still valuable: the
session/agent/tool-calling/error-handling wiring is correct, exercised
through the exact same code path (`AgentSession.run`) the live demo uses,
with no network dependency -- so it can run in CI on every commit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from livekit.agents import AgentSession  # noqa: E402
from fakes import ScriptedLLM  # noqa: E402
from mock_providers import MockSTT, MockTTS  # noqa: E402
from persona import SupportAgent  # noqa: E402


@pytest.fixture
async def session():
    s = AgentSession(stt=MockSTT(), tts=MockTTS(), llm=ScriptedLLM())
    await s.start(agent=SupportAgent())
    yield s
    await s.aclose()


def _event_types(result) -> list[str]:
    return [type(ev).__name__ for ev in result.events]


class TestToolCallingHappyPath:
    @pytest.mark.asyncio
    async def test_status_question_triggers_get_order_status_tool(self, session) -> None:
        result = await session.run(
            user_input="Hi, what's the status of order A100?", input_modality="text"
        )

        assert _event_types(result) == [
            "FunctionCallEvent",
            "FunctionCallOutputEvent",
            "ChatMessageEvent",
        ]

        call_event, output_event, message_event = result.events
        assert call_event.item.name == "get_order_status"
        assert output_event.item.is_error is False
        assert "preparing" in output_event.item.output
        assert "preparing" in message_event.item.content[0]

    @pytest.mark.asyncio
    async def test_cancel_request_triggers_cancel_order_tool(self, session) -> None:
        result = await session.run(user_input="please cancel order A100", input_modality="text")

        call_event, output_event, _ = result.events
        assert call_event.item.name == "cancel_order"
        assert output_event.item.is_error is False
        assert "cancelled" in output_event.item.output


class TestToolCallingErrorPath:
    @pytest.mark.asyncio
    async def test_non_cancellable_order_surfaces_tool_error_gracefully(self, session) -> None:
        # A101 is "out_for_delivery" in the mock DB -> not cancellable.
        result = await session.run(user_input="cancel order A101 now", input_modality="text")

        call_event, output_event, message_event = result.events
        assert call_event.item.name == "cancel_order"
        assert output_event.item.is_error is True
        assert "no longer be cancelled" in output_event.item.output
        # the session must not crash -- a normal assistant message follows
        assert message_event.item.role == "assistant"
        assert "issue" in message_event.item.content[0].lower()


class TestNoToolNeeded:
    @pytest.mark.asyncio
    async def test_message_without_order_id_does_not_call_a_tool(self, session) -> None:
        result = await session.run(user_input="thanks, bye", input_modality="text")

        assert _event_types(result) == ["ChatMessageEvent"]


class TestPipelineIsWiredWithMockedComponents:
    @pytest.mark.asyncio
    async def test_final_message_metrics_report_mock_tts(self, session) -> None:
        """Confirms MockTTS (not a no-op) actually runs for every assistant turn,
        even under a text-modality run -- proving the pipeline is STT/LLM/TTS
        shaped, not just an LLM chatbot with tools bolted on."""
        result = await session.run(user_input="hello", input_modality="text")

        message_event = result.events[-1]
        assert message_event.item.metrics["tts_metadata"]["model_provider"] == "mock"
        assert message_event.item.metrics["tts_metadata"]["model_name"] == "text-io-stub"
