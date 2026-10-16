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

"""Regression tests for ToolLoopBackend's session lifecycle — two bugs found
in code review:

  - Idle eviction used to key only on `last_used` staleness, with no check
    on whether the session's turn currently in flight — a concurrent
    request's eviction sweep could tear down a session's gbmcp subprocess
    out from under an active, possibly long-running turn.
  - Closing a session's AsyncExitStack (which wraps stdio_client/
    ClientSession, both anyio-based) used to happen from whichever task
    happened to trigger eviction/close_session — a *different* task than
    the one that originally entered the stack. anyio cancel scopes can only
    be exited by the exact task that entered them, so this raised and
    leaked the gbmcp subprocess. _run_session_owner() now owns the stack in
    one long-lived task for the life of the session, and is the only thing
    that ever closes it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest

from gb_ui_backend.config import Config
from gb_ui_backend.services.chat_agents import tool_loop_backend
from gb_ui_backend.services.chat_agents.tool_loop_backend import ToolLoopBackend


class _BlockingStubProvider:
    """run_turn() appends the user message, then blocks on resume_event —
    standing in for a long-running tool call (e.g. wait_for_build) that
    holds turn_lock well past any short IDLE_EVICTION_SECONDS window."""

    PROVIDER_NAME = "stub"

    def __init__(self) -> None:
        self.model = "stub-model"
        self.resume_event = asyncio.Event()

    async def run_turn(
        self,
        history: list[Any],
        tools: list[Any],
        user_message: str,
        event_queue: Any,
        interrupt_event: Any,
    ) -> AsyncIterator[dict]:
        history.append({"role": "user", "content": user_message})
        await self.resume_event.wait()
        reply = f"echo: {user_message}"
        history.append({"role": "assistant", "content": reply})
        yield {"type": "text_delta", "text": reply}


class _FakeStdioClient:
    async def __aenter__(self):
        return (None, None)

    async def __aexit__(self, *exc_info):
        return False


class _FakeMcpSession:
    def __init__(self, *_args):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def initialize(self):
        pass

    async def list_tools(self):
        return SimpleNamespace(tools=[])


def _fake_build_gbmcp_tools(_mcp_session, _listed_tools):
    return []


def _fake_build_confirmable_gbmcp_tools(_listed_tools, _event_queue, _pending):
    return []


@pytest.fixture
def backend(monkeypatch):
    stub_provider = _BlockingStubProvider()
    monkeypatch.setattr(tool_loop_backend, "_resolve_gbmcp_bin", lambda: "fake-gbmcp")
    monkeypatch.setattr(
        tool_loop_backend, "stdio_client", lambda params: _FakeStdioClient()
    )
    monkeypatch.setattr(tool_loop_backend, "ClientSession", _FakeMcpSession)
    monkeypatch.setattr(tool_loop_backend, "build_gbmcp_tools", _fake_build_gbmcp_tools)
    monkeypatch.setattr(
        tool_loop_backend,
        "build_confirmable_gbmcp_tools",
        _fake_build_confirmable_gbmcp_tools,
    )
    monkeypatch.setattr(
        tool_loop_backend, "_build_provider", lambda config, prompt: stub_provider
    )

    b = ToolLoopBackend(Config(_env_file=None))
    b._stub_provider = stub_provider  # type: ignore[attr-defined]
    return b


@pytest.mark.asyncio
class TestEvictionNeverInterruptsALiveTurn:
    async def test_session_with_in_flight_turn_is_not_evicted(
        self, backend, monkeypatch
    ):
        """Force every session to look maximally idle/stale, then trigger an
        eviction sweep (via a second session's _get_or_create_session) while
        session-a's turn is still blocked inside run_turn. Without the
        turn_lock check, this used to pop and close session-a mid-turn."""
        monkeypatch.setattr(tool_loop_backend, "IDLE_EVICTION_SECONDS", -1)
        provider: _BlockingStubProvider = backend._stub_provider  # type: ignore[attr-defined]

        async def _drain():
            events = []
            async for event in backend.stream_turn("session-a", "hello", None, None):
                events.append(event)
            return events

        turn_task = asyncio.create_task(_drain())
        # Let stream_turn create the session, acquire turn_lock, and reach
        # run_turn's await on resume_event.
        for _ in range(10):
            await asyncio.sleep(0)

        session_a = backend._sessions["session-a"]
        assert session_a.turn_lock.locked()

        # A concurrent request for a different session runs the eviction
        # sweep — with IDLE_EVICTION_SECONDS=-1 every session (including
        # session-a) looks arbitrarily stale.
        await backend._get_or_create_session("session-b")

        assert backend._sessions.get("session-a") is session_a
        assert not session_a.owner_task.done()

        provider.resume_event.set()
        events = await turn_task

        assert any(e.get("type") == "text_delta" for e in events)
        assert backend._sessions["session-a"].history == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "echo: hello"},
        ]


class _TaskAffineAsyncCM:
    """Mimics the real anyio invariant stdio_client/ClientSession rely on:
    a context manager entered by task T can only be exited by task T. Using
    the plain fakes above (which don't enforce this) would let a regression
    of the cross-task-close bug pass silently."""

    def __init__(self, value):
        self._value = value
        self._enter_task: "asyncio.Task | None" = None

    async def __aenter__(self):
        self._enter_task = asyncio.current_task()
        return self._value

    async def __aexit__(self, *exc_info):
        if asyncio.current_task() is not self._enter_task:
            raise RuntimeError(
                "Attempted to exit cancel scope in a different task than it "
                "was entered in"
            )
        return False


class _TaskAffineMcpSession(_TaskAffineAsyncCM):
    def __init__(self, *_args):
        super().__init__(self)

    async def initialize(self):
        pass

    async def list_tools(self):
        return SimpleNamespace(tools=[])


@pytest.fixture
def task_affine_backend(monkeypatch):
    monkeypatch.setattr(tool_loop_backend, "_resolve_gbmcp_bin", lambda: "fake-gbmcp")
    monkeypatch.setattr(
        tool_loop_backend,
        "stdio_client",
        lambda params: _TaskAffineAsyncCM((None, None)),
    )
    monkeypatch.setattr(tool_loop_backend, "ClientSession", _TaskAffineMcpSession)
    monkeypatch.setattr(tool_loop_backend, "build_gbmcp_tools", _fake_build_gbmcp_tools)
    monkeypatch.setattr(
        tool_loop_backend,
        "build_confirmable_gbmcp_tools",
        _fake_build_confirmable_gbmcp_tools,
    )
    monkeypatch.setattr(
        tool_loop_backend,
        "_build_provider",
        lambda config, prompt: _BlockingStubProvider(),
    )
    return ToolLoopBackend(Config(_env_file=None))


@pytest.mark.asyncio
class TestMaxSessionsCap:
    """Code-review finding: session creation had no cap at all — each
    distinct session_id spawns a gbmcp subprocess, so an unbounded caller
    could exhaust host resources. This is a backend-wide backstop
    independent of chat.py's per-identity rate limit, since it bounds total
    concurrent subprocesses across every identity combined."""

    async def test_creating_a_session_past_the_cap_raises(self, backend, monkeypatch):
        monkeypatch.setattr(tool_loop_backend, "MAX_SESSIONS", 2)
        await backend._get_or_create_session("s1")
        await backend._get_or_create_session("s2")

        with pytest.raises(RuntimeError, match="Too many concurrent chat sessions"):
            await backend._get_or_create_session("s3")

        assert set(backend._sessions) == {"s1", "s2"}

    async def test_reusing_an_existing_session_id_is_unaffected_by_the_cap(
        self, backend, monkeypatch
    ):
        monkeypatch.setattr(tool_loop_backend, "MAX_SESSIONS", 1)
        await backend._get_or_create_session("s1")

        await backend._get_or_create_session("s1")  # must not raise — not a new session


@pytest.mark.asyncio
class TestSessionOwnerClosesItsOwnStack:
    async def test_close_session_does_not_raise_across_task_boundary(
        self, task_affine_backend
    ):
        """close_session() runs in the test's own task — a different task
        than the one that entered the AsyncExitStack (the session's
        dedicated owner task). Before the fix, closing directly from here
        raised RuntimeError (swallowed elsewhere, but reproduced faithfully
        by this fake) and left the fake "subprocess" never torn down."""
        await task_affine_backend._get_or_create_session("s1")
        session = task_affine_backend._sessions["s1"]
        owner_task = session.owner_task

        await task_affine_backend.close_session("s1")

        assert "s1" not in task_affine_backend._sessions
        assert owner_task.done()
        assert owner_task.exception() is None

    async def test_eviction_also_closes_from_the_owning_task(
        self, task_affine_backend, monkeypatch
    ):
        monkeypatch.setattr(tool_loop_backend, "IDLE_EVICTION_SECONDS", -1)
        await task_affine_backend._get_or_create_session("s1")
        session = task_affine_backend._sessions["s1"]

        # Any subsequent _get_or_create_session call runs the eviction
        # sweep; s1 isn't mid-turn, so with IDLE_EVICTION_SECONDS=-1 it's
        # evicted here.
        await task_affine_backend._get_or_create_session("s2")

        assert "s1" not in task_affine_backend._sessions
        assert session.owner_task.done()
        assert session.owner_task.exception() is None
