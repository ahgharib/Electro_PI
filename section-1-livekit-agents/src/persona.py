"""The support agent: persona + tools.

Two tools are implemented (not just described in the write-up):

1. `get_order_status` -- the example from the task spec. Straightforward
   lookup, demonstrates the happy path.
2. `cancel_order` -- deliberately designed to also exercise the *unhappy*
   path: unknown order IDs and orders that are no longer cancellable both
   raise `ToolError`, which LiveKit surfaces back to the LLM as a tool
   result it can react to in natural language, instead of raising an
   exception that would crash the turn. This is the safe-error-handling
   pattern described in the write-up (NOTES.md), demonstrated in code
   rather than left as a hypothetical.

The mock order data lives in mock_orders_data.py, deliberately separate
from this file -- see that module to add, remove, or edit test orders.
Swapping it for a real orders API is a one-line change inside each tool
below; the Agent/tool-calling contract doesn't change.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterable

from livekit import rtc
from livekit.agents import Agent, function_tool
from livekit.agents.llm import ToolError
from livekit.agents.voice import ModelSettings

from mock_orders_data import Order, seed_orders
from speech_sanitizer import sanitize_for_speech

logger = logging.getLogger("voice_agent.persona")


SYSTEM_INSTRUCTIONS = """\
You are Sam, a support assistant for Foodie, a food-delivery app.

You help customers check their order status and cancel orders when that's
still possible. Be warm, concise, and to the point -- this is a voice
conversation, so avoid long lists or anything that's awkward to say out
loud. Always use the tools to look up real order information instead of
guessing; never invent an order status. If a tool reports an error (for
example, an order can't be found or can't be cancelled anymore), explain
that plainly to the customer and offer a next step instead of repeating
the raw error.

Before calling a tool, check what's already happened earlier in this
conversation: if you already called it with the exact same order ID and
got a result -- found or not found, succeeded or failed -- don't call it
again with that same ID. Just refer back to what you already found out
instead of repeating the lookup. Only call it again if the customer gives
you a different order ID, or clearly asks you to check again (for example,
because they think something may have changed). If you already learned
that an order ID doesn't exist at all (from either tool), treat that as
settled -- don't call either tool again with that same ID; just tell the
customer it doesn't exist, no matter which action they're now asking for.

When you refer to a tool call you already made, or explain that you won't
repeat one, describe it in a plain, natural sentence only. Never write
tool-call syntax such as <function=...>...</function>, JSON, or any other
code-like text as part of what you say -- that syntax is only valid when
you actually invoke a tool through the real mechanism, never as words in
your reply.

This is a voice conversation: everything you say gets read aloud. Never
use emojis, markdown formatting (asterisks, bullet points, headers,
backticks), or symbols without a natural spoken form. Say numbers the way
you'd say them out loud in conversation.
"""


class SupportAgent(Agent):
    """Support assistant persona for a fictional food-delivery app."""

    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_INSTRUCTIONS)
        self._orders: dict[str, Order] = seed_orders()

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        """Hard guarantee that nothing unspeakable reaches the TTS engine.

        SYSTEM_INSTRUCTIONS already asks the model not to produce emojis or
        markdown, but a prompt instruction is probabilistic, not a
        contract. This overrides the default synthesis pipeline to strip
        each text segment (see speech_sanitizer.py) before it's handed to
        whichever TTS is configured -- this runs regardless of which
        provider that is (MockTTS today, a real one after the 1.2 swap),
        since it sits in the Agent layer, above the TTS interface.
        """

        async def _sanitized_text() -> AsyncIterable[str]:
            async for chunk in text:
                cleaned = sanitize_for_speech(chunk)
                if cleaned:
                    yield cleaned

        async for frame in Agent.default.tts_node(self, _sanitized_text(), model_settings):
            yield frame

    @function_tool
    async def get_order_status(self, order_id: str) -> str:
        """Look up the current status of a customer's order.

        Args:
            order_id: The order identifier the customer gave you, e.g. "A100".
        """
        order = self._orders.get(order_id.strip().upper())
        logger.info("tool_get_order_status", extra={"order_id": order_id, "found": order is not None})

        if order is None:
            raise ToolError(
                f"No order found with ID '{order_id}'. Ask the customer to double-check the "
                "order number -- it's in their confirmation email or the app's order history."
            )

        if order.eta_minutes is not None:
            return f"Order {order.order_id} is currently '{order.status}', ETA {order.eta_minutes} minutes."
        return f"Order {order.order_id} is currently '{order.status}'."

    @function_tool
    async def cancel_order(self, order_id: str, reason: str) -> str:
        """Cancel a customer's order, if it's still in a cancellable state.

        Args:
            order_id: The order identifier to cancel, e.g. "A100".
            reason: A short reason for the cancellation, for the support log.
        """
        key = order_id.strip().upper()
        order = self._orders.get(key)
        logger.info(
            "tool_cancel_order",
            extra={"order_id": order_id, "reason": reason, "found": order is not None},
        )

        if order is None:
            raise ToolError(f"No order found with ID '{order_id}'.")

        if not order.cancellable:
            raise ToolError(
                f"Order {order.order_id} can no longer be cancelled (current status: "
                f"'{order.status}'). It's too late to stop this one."
            )

        self._orders[key] = Order(order.order_id, "cancelled", eta_minutes=None, cancellable=False)
        logger.info("order_cancelled", extra={"order_id": key, "reason": reason})
        return f"Order {order.order_id} has been cancelled."
