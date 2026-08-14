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

"""Tests for the propose-then-confirm gate on build_start/gbserver_stop
(gbmcp_policy.py's CONFIRMABLE_GBMCP_TOOLS).

Covers the correction to an earlier, over-scoped fix: these two tools must
be genuinely *available* to the model — proposing them, not declining them —
with real execution gated behind ToolLoopBackend.confirm_action(), which is
only ever reached via a separate POST /chat/confirm call, never from inside
a model turn.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest

from gb_ui_backend.config import Config
from gb_ui_backend.services.chat_agents import tool_loop_backend
from gb_ui_backend.services.chat_agents.tool_loop_backend import ToolLoopBackend


class _StubProvider:
    """When the model calls build_start/gbserver_stop, run_turn() invokes
    the ToolSpec's handler directly (mirroring what a real provider does)
    and yields the resulting tool_call/confirm_action events, without any
    real model round-trip."""

    PROVIDER_NAME = "stub"

    def __init__(self) -> None:
        self.model = "stub-model"

    async def run_turn(
        self,
        history: list[Any],
        tools: list[Any],
        user_message: str,
        event_queue: Any,
        interrupt_event: Any,
    ) -> AsyncIterator[dict]:
        history.append({"role": "user", "content": user_message})
        for tool in tools:
            if tool.name in ("build_start", "gbserver_stop"):
                yield {"type": "tool_call", "tool_name": tool.name, "tool_input": {}}
                await tool.handler({})
                while not event_queue.empty():
                    yield event_queue.get_nowait()


class _FakeStdioClient:
    async def __aenter__(self):
        return (None, None)

    async def __aexit__(self, *exc_info):
        return False


class _FakeMcpSession:
    """Unlike test_chat_page_context.py's fake, this one implements
    list_tools()/call_tool() for real — confirm_action()'s execute-on-approve
    path calls call_tool() directly, so this suite needs it to actually work,
    not just no-op."""

    def __init__(self, *_args):
        self.call_tool_calls: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def initialize(self):
        pass

    async def list_tools(self):
        tool = lambda name: SimpleNamespace(  # noqa: E731
            name=name,
            description=f"{name} tool",
            inputSchema={"type": "object", "properties": {}},
        )
        return SimpleNamespace(tools=[tool("build_start"), tool("gbserver_stop")])

    async def call_tool(self, name: str, args: dict):
        self.call_tool_calls.append((name, args))
        if name == "gbserver_stop":
            return SimpleNamespace(
                content=[SimpleNamespace(text="gbserver stopped")], isError=False
            )
        return SimpleNamespace(
            content=[SimpleNamespace(text="ERROR: no such space")], isError=True
        )


@pytest.fixture
def backend(monkeypatch):
    stub_provider = _StubProvider()
    monkeypatch.setattr(tool_loop_backend, "_resolve_gbmcp_bin", lambda: "fake-gbmcp")
    monkeypatch.setattr(
        tool_loop_backend, "stdio_client", lambda params: _FakeStdioClient()
    )
    monkeypatch.setattr(tool_loop_backend, "ClientSession", _FakeMcpSession)
    monkeypatch.setattr(
        tool_loop_backend, "_build_provider", lambda config, prompt: stub_provider
    )
    return ToolLoopBackend(Config(_env_file=None))


async def _propose(backend, session_id: str, action: str) -> str:
    """Drains one stream_turn() call and returns the confirmation_id from
    the resulting confirm_action event."""
    confirmation_id = None
    async for event in backend.stream_turn(session_id, f"please {action}"):
        if event.get("type") == "confirm_action" and event.get("tool_name") == action:
            confirmation_id = event["confirmation_id"]
    assert confirmation_id, f"no confirm_action event for {action}"
    return confirmation_id


@pytest.mark.asyncio
class TestConfirmableToolsAreProposedNotExecuted:
    async def test_calling_build_start_never_touches_gbmcp(self, backend):
        confirmation_id = await _propose(backend, "s1", "build_start")
        session = backend._sessions["s1"]
        assert session.mcp_session.call_tool_calls == []
        assert confirmation_id in session.pending_confirmations
        assert session.pending_confirmations[confirmation_id] == {
            "action": "build_start",
            "args": {},
        }


@pytest.mark.asyncio
class TestConfirmAction:
    async def test_approve_executes_and_records_success_in_history(self, backend):
        confirmation_id = await _propose(backend, "s1", "gbserver_stop")
        session = backend._sessions["s1"]

        result = await backend.confirm_action("s1", confirmation_id, approved=True)

        assert result == {
            "found": True,
            "approved": True,
            "result": "gbserver stopped",
            "is_error": False,
        }
        assert session.mcp_session.call_tool_calls == [("gbserver_stop", {})]
        assert confirmation_id not in session.pending_confirmations
        assert "approved the proposed gbserver_stop action" in str(session.history[-1])
        assert "gbserver stopped" in str(session.history[-1])

    async def test_approve_records_failure_without_raising(self, backend):
        confirmation_id = await _propose(backend, "s1", "build_start")

        result = await backend.confirm_action("s1", confirmation_id, approved=True)

        assert result["approved"] is True
        assert result["is_error"] is True
        assert "no such space" in result["result"]

    async def test_decline_never_calls_gbmcp(self, backend):
        confirmation_id = await _propose(backend, "s1", "build_start")
        session = backend._sessions["s1"]

        result = await backend.confirm_action("s1", confirmation_id, approved=False)

        assert result == {"found": True, "approved": False}
        assert session.mcp_session.call_tool_calls == []
        assert confirmation_id not in session.pending_confirmations
        assert "declined the proposed build_start action" in str(session.history[-1])

    async def test_unknown_confirmation_id_returns_found_false_without_raising(
        self, backend
    ):
        await _propose(backend, "s1", "build_start")  # ensures the session exists

        result = await backend.confirm_action("s1", "not-a-real-id", approved=True)

        assert result == {"found": False}

    async def test_unknown_session_id_returns_found_false_without_raising(
        self, backend
    ):
        result = await backend.confirm_action(
            "no-such-session", "not-a-real-id", approved=True
        )

        assert result == {"found": False}

    async def test_resolving_twice_only_executes_once(self, backend):
        """The pending entry is popped, not just read — a second confirm
        call for the same id (e.g. a double-click) must not re-execute."""
        confirmation_id = await _propose(backend, "s1", "gbserver_stop")

        first = await backend.confirm_action("s1", confirmation_id, approved=True)
        second = await backend.confirm_action("s1", confirmation_id, approved=True)

        assert first["found"] is True
        assert second == {"found": False}
        assert backend._sessions["s1"].mcp_session.call_tool_calls == [
            ("gbserver_stop", {})
        ]
