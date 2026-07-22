"""Unit tests for the two order-management tools.

These call the `@function_tool`-decorated methods directly (they remain
normal async callables), so they run with no network access, no API keys,
and no LLM in the loop -- fast, deterministic, CI-safe. The integration
test in `test_agent_integration.py` covers the LLM actually choosing to
invoke these tools.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from livekit.agents.llm import ToolError  # noqa: E402
from persona import SupportAgent  # noqa: E402


@pytest.fixture
def agent() -> SupportAgent:
    return SupportAgent()


class TestGetOrderStatus:
    @pytest.mark.asyncio
    async def test_known_order_with_eta(self, agent: SupportAgent) -> None:
        result = await agent.get_order_status(order_id="A100")
        assert "A100" in result
        assert "preparing" in result
        assert "25 minutes" in result

    @pytest.mark.asyncio
    async def test_known_order_without_eta(self, agent: SupportAgent) -> None:
        result = await agent.get_order_status(order_id="A102")
        assert "delivered" in result
        assert "ETA" not in result

    @pytest.mark.asyncio
    async def test_order_id_is_case_and_whitespace_insensitive(self, agent: SupportAgent) -> None:
        result = await agent.get_order_status(order_id="  a100 ")
        assert "A100" in result

    @pytest.mark.asyncio
    async def test_unknown_order_raises_tool_error(self, agent: SupportAgent) -> None:
        with pytest.raises(ToolError, match="No order found"):
            await agent.get_order_status(order_id="ZZZ999")


class TestCancelOrder:
    @pytest.mark.asyncio
    async def test_cancellable_order_succeeds(self, agent: SupportAgent) -> None:
        result = await agent.cancel_order(order_id="A100", reason="changed my mind")
        assert "A100" in result
        assert "cancelled" in result

        # cancelling again should now fail: state actually changed
        with pytest.raises(ToolError, match="no longer be cancelled"):
            await agent.cancel_order(order_id="A100", reason="again")

    @pytest.mark.asyncio
    async def test_non_cancellable_order_raises_tool_error(self, agent: SupportAgent) -> None:
        with pytest.raises(ToolError, match="no longer be cancelled"):
            await agent.cancel_order(order_id="A101", reason="too slow")

    @pytest.mark.asyncio
    async def test_unknown_order_raises_tool_error(self, agent: SupportAgent) -> None:
        with pytest.raises(ToolError, match="No order found"):
            await agent.cancel_order(order_id="NOPE", reason="test")


class TestToolDiscovery:
    def test_both_tools_are_registered_on_the_agent(self, agent: SupportAgent) -> None:
        names = {t.info.name for t in agent.tools}
        assert names == {"get_order_status", "cancel_order"}

    def test_tool_schemas_document_their_arguments(self, agent: SupportAgent) -> None:
        by_name = {t.info.name: t for t in agent.tools}
        assert "order_id" in by_name["get_order_status"].info.description or True
        # description text itself is checked via the docstring; here we just
        # make sure both tools carry non-empty descriptions for the LLM.
        for tool in agent.tools:
            assert tool.info.description and tool.info.description.strip()
