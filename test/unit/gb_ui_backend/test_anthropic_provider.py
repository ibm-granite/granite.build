#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for AnthropicProvider.run_turn's history-rollback paths.

Covers a code-review finding: exhausting MAX_TOOL_ROUNDS, or any non-
InterruptedError exception mid-turn, used to leave `history` ending in an
assistant tool_use block with no matching tool_result — which the Anthropic
API rejects on the session's next turn. Constructs AnthropicProvider via
__new__ (bypassing __init__'s `AsyncAnthropic()` call) so these tests don't
require the optional `chat-anthropic` extra to be installed.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gb_ui_backend.services.chat_agents.providers.anthropic_provider import (
    AnthropicProvider,
)
from gb_ui_backend.services.chat_agents.tool_registry import MAX_TOOL_ROUNDS, ToolSpec


def _make_provider() -> AnthropicProvider:
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.model = "fake-model"
    provider._system_prompt = "sys"
    return provider


def _tool_use_message(block_id: str, name: str, tool_input: dict) -> SimpleNamespace:
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="tool_use", id=block_id, name=name, input=tool_input)
        ]
    )


@pytest.mark.asyncio
class TestMaxToolRoundsRollsBackHistory:
    async def test_exhausting_max_rounds_rolls_back_history(self, monkeypatch):
        provider = _make_provider()

        async def always_tool_call(history, tools):
            return _tool_use_message("call_1", "loop_tool", {})

        monkeypatch.setattr(provider, "_call_model", always_tool_call)

        async def handler(args):
            return "ok"

        tool = ToolSpec(
            name="loop_tool", description="", parameters={}, handler=handler
        )

        history: list = []
        events = [
            e
            async for e in provider.run_turn(
                history, [tool], "hello", asyncio.Queue(), asyncio.Event()
            )
        ]

        assert events[-1] == {
            "type": "error",
            "message": f"Stopped after {MAX_TOOL_ROUNDS} tool-calling rounds without a final answer.",
        }
        # No dangling assistant tool_use block left for the next turn to choke on.
        assert history == []


@pytest.mark.asyncio
class TestNonInterruptExceptionRollsBackHistory:
    async def test_model_call_exception_rolls_back_history_and_propagates(
        self, monkeypatch
    ):
        provider = _make_provider()

        async def failing_call_model(history, tools):
            raise RuntimeError("boom")

        monkeypatch.setattr(provider, "_call_model", failing_call_model)

        history: list = []
        with pytest.raises(RuntimeError, match="boom"):
            async for _ in provider.run_turn(
                history, [], "hello", asyncio.Queue(), asyncio.Event()
            ):
                pass

        assert history == []

    async def test_a_stale_queued_event_from_the_rolled_back_turn_is_discarded(
        self, monkeypatch
    ):
        """A confirmable tool's handler puts a confirm_action event on
        event_queue as a side effect before the round that ultimately fails
        — that event must not survive to be drained by a later, unrelated
        turn once this turn's history is erased."""
        provider = _make_provider()

        async def call_then_fail(history, tools):
            if len(history) == 1:  # first round: propose an action
                return _tool_use_message("call_1", "confirmable_tool", {})
            raise RuntimeError("boom")

        monkeypatch.setattr(provider, "_call_model", call_then_fail)

        event_queue: asyncio.Queue = asyncio.Queue()

        async def handler(args):
            await event_queue.put({"type": "confirm_action", "confirmation_id": "c1"})
            return "proposed"

        tool = ToolSpec(
            name="confirmable_tool", description="", parameters={}, handler=handler
        )

        history: list = []
        with pytest.raises(RuntimeError, match="boom"):
            async for _ in provider.run_turn(
                history, [tool], "hello", event_queue, asyncio.Event()
            ):
                pass

        assert history == []
        assert event_queue.empty()

    async def test_cancellation_still_rolls_back_history_and_propagates(
        self, monkeypatch
    ):
        """asyncio.CancelledError is a BaseException, not an Exception
        subclass — narrower `except Exception` coverage here used to let a
        client-disconnect cancellation skip the rollback, corrupting history
        for the rest of the session."""
        provider = _make_provider()

        async def cancelled_call_model(history, tools):
            raise asyncio.CancelledError()

        monkeypatch.setattr(provider, "_call_model", cancelled_call_model)

        history: list = []
        with pytest.raises(asyncio.CancelledError):
            async for _ in provider.run_turn(
                history, [], "hello", asyncio.Queue(), asyncio.Event()
            ):
                pass

        assert history == []
