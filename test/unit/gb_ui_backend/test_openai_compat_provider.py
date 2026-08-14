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

"""Regression tests for OpenAICompatProvider.run_turn's interrupt handling.

Covers a code-review finding: /chat/stop only ever raced the model call, not
tool execution, so stopping mid-wait_for_build (up to 30 minutes) did nothing
until the tool finished on its own. The fix makes tool execution
interruptible too — but interrupting mid-turn (whether during the model call
or a tool call) must leave `history` exactly as it was before the turn
started, not with a dangling unanswered user/assistant/tool entry that would
corrupt every later turn in the session.
"""

from __future__ import annotations

import asyncio

import pytest

from gb_ui_backend.services.chat_agents.providers.openai_compat_provider import OpenAICompatProvider
from gb_ui_backend.services.chat_agents.tool_registry import ToolSpec


def _make_provider() -> OpenAICompatProvider:
    return OpenAICompatProvider(base_url="http://fake", api_key="x", model="fake-model", system_prompt="sys")


@pytest.mark.asyncio
class TestInterruptRollsBackHistory:
    async def test_interrupt_during_model_call_leaves_history_at_pre_turn_state(self, monkeypatch):
        provider = _make_provider()
        interrupt_event = asyncio.Event()

        async def slow_chat_completion(**kwargs):
            interrupt_event.set()  # simulate POST /chat/stop firing while the model call is in flight
            await asyncio.sleep(10)  # would hang the test if race_interrupt didn't cancel this
            raise AssertionError("should have been cancelled before this ever resolved")

        monkeypatch.setattr(provider._client, "chat_completion", slow_chat_completion)

        history: list = []
        events = [e async for e in provider.run_turn(history, [], "hello", asyncio.Queue(), interrupt_event)]

        # Only the bootstrap system message survives — the user message this
        # turn added gets rolled back, not left dangling with no reply.
        assert history == [{"role": "system", "content": "sys"}]
        assert events == []

    async def test_interrupt_during_tool_call_leaves_history_at_pre_turn_state(self, monkeypatch):
        provider = _make_provider()
        interrupt_event = asyncio.Event()

        async def fake_chat_completion(**kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{"id": "call_1", "function": {"name": "slow_tool", "arguments": "{}"}}],
                        }
                    }
                ]
            }

        monkeypatch.setattr(provider._client, "chat_completion", fake_chat_completion)

        async def slow_handler(args):
            interrupt_event.set()  # simulate POST /chat/stop firing mid-tool-call (e.g. wait_for_build)
            await asyncio.sleep(10)
            raise AssertionError("should have been cancelled before this ever resolved")

        tool = ToolSpec(name="slow_tool", description="", parameters={}, handler=slow_handler)

        history: list = []
        events = [
            e async for e in provider.run_turn(history, [tool], "hello", asyncio.Queue(), interrupt_event)
        ]

        # Rolled back past the assistant's tool_use message too — not just
        # the user message — so there's no orphaned tool_call with no
        # matching tool result left behind.
        assert history == [{"role": "system", "content": "sys"}]
        assert events == [{"type": "tool_call", "tool_name": "slow_tool", "tool_input": {}}]

    async def test_malformed_tool_call_missing_id_and_name_does_not_crash(self, monkeypatch):
        """A non-compliant OpenAI-compatible endpoint (this provider's whole
        point is supporting less-standardized ones) can return a tool_call
        missing `id`/`function.name` — that must degrade to a clean tool
        error, not an unhandled KeyError killing the turn."""
        provider = _make_provider()
        call_count = 0

        async def fake_chat_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "choices": [
                        {"message": {"role": "assistant", "content": None, "tool_calls": [{"function": {}}]}}
                    ]
                }
            # Second round: no further tool calls, so the loop ends naturally
            # instead of resubmitting the same malformed tool_call 12 times.
            return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

        monkeypatch.setattr(provider._client, "chat_completion", fake_chat_completion)

        history: list = []
        events = [
            e
            async for e in provider.run_turn(history, [], "hello", asyncio.Queue(), asyncio.Event())
        ]

        assert events == [
            {"type": "tool_call", "tool_name": "<missing tool name>", "tool_input": {}},
            {"type": "text_delta", "text": "done"},
        ]
        assert {"role": "tool", "tool_call_id": "", "content": "Unknown tool '<missing tool name>'"} in history

    async def test_invalid_json_arguments_still_yields_a_tool_call_event(self, monkeypatch):
        """Regression: the invalid-JSON branch used to append straight to
        history and `continue`, skipping the tool_call event every other
        failure mode (unknown tool, handler exception) yields. That left a
        gap in the event stream's shape with no current symptom (nothing
        consumes tool_call yet — see ChatWidget.tsx) but would silently break
        any future consumer of it."""
        provider = _make_provider()
        call_count = 0

        async def fake_chat_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {"id": "call_1", "function": {"name": "build_status", "arguments": "{bad json"}}
                                ],
                            }
                        }
                    ]
                }
            # Second round: no further tool calls, so the loop ends naturally.
            return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

        monkeypatch.setattr(provider._client, "chat_completion", fake_chat_completion)

        history: list = []
        events = [e async for e in provider.run_turn(history, [], "hello", asyncio.Queue(), asyncio.Event())]

        assert events == [
            {"type": "tool_call", "tool_name": "build_status", "tool_input": {}},
            {"type": "text_delta", "text": "done"},
        ]
        assert any(
            m.get("role") == "tool"
            and m.get("tool_call_id") == "call_1"
            and "Invalid JSON arguments" in m.get("content", "")
            for m in history
        )
