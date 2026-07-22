"""A deterministic, offline stand-in for a real LLM.

This is NOT the "real LLM" evidence the task asks for -- it never talks to
Groq/Cerebras/any network, and its "decisions" are simple keyword matching,
not model reasoning. Its purpose is narrower: prove that the wiring between
`AgentSession`, `SupportAgent`, and the tool-calling machinery is correct
end-to-end (tool gets invoked, arguments get parsed, `ToolError` gets
surfaced back into the conversation) without needing an API key in CI.

See NOTES.md / the README for how to run the same conversation against a
real Groq/Cerebras model to get the actual required evidence transcript.
"""

from __future__ import annotations

import json
import re
from typing import ClassVar

from livekit.agents import llm
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, NotGivenOr, NOT_GIVEN
from livekit.agents.utils import shortuuid

_ORDER_ID_RE = re.compile(r"\b[A-Za-z]\d{3}\b")


class ScriptedLLM(llm.LLM):
    """Keyword-driven fake LLM used only in tests, never in the live demo."""

    label: ClassVar[str] = "test.ScriptedLLM"

    @property
    def model(self) -> str:
        return "scripted-test-double"

    @property
    def provider(self) -> str:
        return "test"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict] = NOT_GIVEN,
    ) -> llm.LLMStream:
        return _ScriptedLLMStream(
            self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options
        )


class _ScriptedLLMStream(llm.LLMStream):
    async def _run(self) -> None:
        items = self._chat_ctx.items
        last = items[-1] if items else None

        if last is not None and last.type == "function_call_output":
            text = self._reply_after_tool_result(last)
            self._send_text(text)
            return

        last_user_text = self._last_user_text()
        order_id_match = _ORDER_ID_RE.search(last_user_text or "")

        if last_user_text and "cancel" in last_user_text.lower() and order_id_match:
            self._send_tool_call(
                name="cancel_order",
                arguments={"order_id": order_id_match.group(0), "reason": "customer request"},
            )
            return

        if last_user_text and order_id_match and any(
            kw in last_user_text.lower() for kw in ("status", "where", "track")
        ):
            self._send_tool_call(
                name="get_order_status", arguments={"order_id": order_id_match.group(0)}
            )
            return

        self._send_text(
            "Hi, I'm Sam from Foodie support. Could you give me your order number "
            "so I can look into that for you?"
        )

    def _last_user_text(self) -> str | None:
        for item in reversed(self._chat_ctx.items):
            if item.type == "message" and item.role == "user":
                return item.text_content
        return None

    def _reply_after_tool_result(self, tool_output: llm.FunctionCallOutput) -> str:
        if tool_output.is_error:
            return f"I ran into an issue: {tool_output.output}"
        return f"Here's what I found: {tool_output.output}"

    def _send_text(self, text: str) -> None:
        self._event_ch.send_nowait(
            llm.ChatChunk(
                id=shortuuid("scripted_"),
                delta=llm.ChoiceDelta(role="assistant", content=text),
            )
        )

    def _send_tool_call(self, *, name: str, arguments: dict) -> None:
        self._event_ch.send_nowait(
            llm.ChatChunk(
                id=shortuuid("scripted_"),
                delta=llm.ChoiceDelta(
                    role="assistant",
                    tool_calls=[
                        llm.FunctionToolCall(
                            name=name,
                            arguments=json.dumps(arguments),
                            call_id=shortuuid("call_"),
                        )
                    ],
                ),
            )
        )
