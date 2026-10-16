"""Runs the agentic tool-calling loop directly against the Anthropic Messages
API — not the Claude Agent SDK, which shells out to the Node-based Claude
Code CLI as a subprocess. `anthropic` is a thin, official API client with no
CLI/subprocess dependency, and AsyncAnthropic() reads ANTHROPIC_API_KEY /
ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN from the environment on its own,
matching this project's existing documented configuration story exactly.

Import of `anthropic` is guarded — Anthropic support is opt-in, not bundled
into the base `chat` extra or `standalone` (`pip install -e '.[chat-anthropic]'`),
so the base gb_ui_backend install must keep working without it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from gb_ui_backend.services.chat_agents.base import NormalizedEvent
from gb_ui_backend.services.chat_agents.tool_registry import (
    MAX_TOKENS,
    ToolSpec,
    race_interrupt,
    run_tool_loop,
)

logger = logging.getLogger(__name__)

try:
    from anthropic import AsyncAnthropic

    _ANTHROPIC_AVAILABLE = True
except ImportError:  # base install has no chat extra — that's fine
    _ANTHROPIC_AVAILABLE = False


def _to_anthropic_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.parameters}
        for t in tools
    ]


class AnthropicProvider:
    # Public — surfaced by ToolLoopBackend.describe() (GET /api/analytics/chat/status)
    # so the frontend can show which model/provider is actually running.
    PROVIDER_NAME = "anthropic"

    def __init__(self, model: str, system_prompt: str) -> None:
        if not _ANTHROPIC_AVAILABLE:
            raise RuntimeError(
                "anthropic is not installed. Install it with `pip install -e '.[chat-anthropic]'`."
            )
        self._client = AsyncAnthropic()
        self.model = model
        self._system_prompt = system_prompt

    def run_turn(
        self,
        history: list[dict[str, Any]],
        tools: list[ToolSpec],
        user_message: str,
        event_queue: "asyncio.Queue[NormalizedEvent]",
        interrupt_event: asyncio.Event,
    ) -> AsyncIterator[NormalizedEvent]:
        # Snapshotted before appending this turn's user message — see
        # run_tool_loop()'s docstring for why it can't take this snapshot
        # itself. An interrupt/error rolls back to exactly this, whether
        # mid model-call or mid tool-call, in any round.
        original_length = len(history)
        history.append({"role": "user", "content": user_message})
        tool_by_name = {t.name: t for t in tools}
        anthropic_tools = _to_anthropic_tools(tools)

        async def _run_one_round(
            outcome: dict[str, Any],
        ) -> AsyncIterator[NormalizedEvent]:
            message = await race_interrupt(
                self._call_model(history, anthropic_tools), interrupt_event
            )

            for block in message.content:
                if block.type == "text" and block.text:
                    yield {"type": "text_delta", "text": block.text}

            history.append({"role": "assistant", "content": message.content})

            tool_use_blocks = [b for b in message.content if b.type == "tool_use"]
            if not tool_use_blocks:
                outcome["status"] = "done"
                return

            tool_results = []
            for block in tool_use_blocks:
                yield {
                    "type": "tool_call",
                    "tool_name": block.name,
                    "tool_input": block.input,
                }
                tool = tool_by_name.get(block.name)
                if tool is None:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Unknown tool {block.name!r}",
                            "is_error": True,
                        }
                    )
                    continue
                try:
                    # Also interruptible — a tool like wait_for_build can
                    # run for up to 30 minutes; without this, /chat/stop
                    # would only take effect once it finished on its own.
                    result = await race_interrupt(
                        tool.handler(block.input or {}), interrupt_event
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                        }
                    )
                except InterruptedError:
                    raise  # let run_tool_loop roll history back — don't treat this as a tool error
                except (
                    Exception
                ) as exc:  # noqa: BLE001 - surfaced to the model as a tool error, not raised
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(exc),
                            "is_error": True,
                        }
                    )
                while not event_queue.empty():
                    yield event_queue.get_nowait()

            history.append({"role": "user", "content": tool_results})
            outcome["status"] = "continue"

        return run_tool_loop(
            history, event_queue, original_length, _run_one_round, "Anthropic"
        )

    async def _call_model(
        self, history: list[dict[str, Any]], anthropic_tools: list[dict[str, Any]]
    ) -> Any:
        return await self._client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=self._system_prompt,
            messages=history,
            tools=anthropic_tools,
        )
