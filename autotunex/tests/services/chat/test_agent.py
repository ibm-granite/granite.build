"""Tests for :mod:`autotunex.services.chat.agent`."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from autotunex.core.config import Settings
from autotunex.core.exceptions import LlmUnavailableError
from autotunex.models.auth import Principal
from autotunex.services.chat.agent import MAX_TOOL_RESULT_CHARS, _truncate_tool_result, run_agent
from autotunex.services.chat.context import ToolContext
from autotunex.services.llm.base import ChatDelta, ToolCallDelta


def _ctx(session_factory: async_sessionmaker[AsyncSession], principal: Principal) -> ToolContext:
    return ToolContext(
        principal=principal, settings=Settings(job_backend="none"), session_factory=session_factory
    )


class _ScriptedLlm:
    """Yields a tool call on the first turn, then a final answer."""

    def __init__(self) -> None:
        self._turn = 0

    async def complete(self, **_kw: object) -> str:  # unused Protocol member
        return ""

    async def stream_chat(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[ChatDelta]:
        self._turn += 1
        if self._turn == 1:
            yield ChatDelta(
                tool_calls=[ToolCallDelta(index=0, id="c1", name="list_jobs", arguments="{}")]
            )
            yield ChatDelta(finish_reason="tool_calls")
        else:
            yield ChatDelta(content="You have no jobs.")
            yield ChatDelta(finish_reason="stop")


class _AlwaysToolLlm:
    """Always asks for the same tool call — never produces a final answer."""

    async def complete(self, **_kw: object) -> str:
        return ""

    async def stream_chat(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[ChatDelta]:
        yield ChatDelta(
            tool_calls=[ToolCallDelta(index=0, id="c", name="list_jobs", arguments="{}")]
        )
        yield ChatDelta(finish_reason="tool_calls")


class _TalksThenCallsToolLlm:
    """Streams pre-tool text alongside a tool call, then a post-tool answer."""

    def __init__(self) -> None:
        self._turn = 0

    async def complete(self, **_kw: object) -> str:
        return ""

    async def stream_chat(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[ChatDelta]:
        self._turn += 1
        if self._turn == 1:
            yield ChatDelta(content="Sure, let me look.")
            yield ChatDelta(
                tool_calls=[ToolCallDelta(index=0, id="c1", name="list_jobs", arguments="{}")]
            )
            yield ChatDelta(finish_reason="tool_calls")
        else:
            yield ChatDelta(content="Done!")
            yield ChatDelta(finish_reason="stop")


class _PlainAnswerLlm:
    """Answers directly with plain text — never calls a tool at all."""

    async def complete(self, **_kw: object) -> str:
        return ""

    async def stream_chat(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[ChatDelta]:
        yield ChatDelta(content="Hello there, ")
        yield ChatDelta(content="how can I help?")
        yield ChatDelta(finish_reason="stop")


class _MalformedArgsLlm:
    """Calls a tool with unparseable JSON arguments, then answers."""

    def __init__(self) -> None:
        self._turn = 0

    async def complete(self, **_kw: object) -> str:
        return ""

    async def stream_chat(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[ChatDelta]:
        self._turn += 1
        if self._turn == 1:
            yield ChatDelta(
                tool_calls=[ToolCallDelta(index=0, id="c1", name="get_job", arguments="{not-json")]
            )
            yield ChatDelta(finish_reason="tool_calls")
        else:
            yield ChatDelta(content="ok")
            yield ChatDelta(finish_reason="stop")


class _UnavailableLlm:
    """Raises ``LlmUnavailableError`` on its first (and only expected) call."""

    def __init__(self) -> None:
        self._turn = 0

    async def complete(self, **_kw: object) -> str:
        return ""

    async def stream_chat(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[ChatDelta]:
        self._turn += 1
        if self._turn == 1:
            raise LlmUnavailableError()
        yield ChatDelta(finish_reason="stop")


class _BoomLlm:
    """Raises an exception unrelated to the LLM-unavailable seam."""

    def __init__(self) -> None:
        self._turn = 0

    async def complete(self, **_kw: object) -> str:
        return ""

    async def stream_chat(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[ChatDelta]:
        self._turn += 1
        if self._turn == 1:
            raise RuntimeError("boom")
        yield ChatDelta(finish_reason="stop")


async def test_agent_runs_tool_then_answers(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """A tool-call round runs, then the model's final answer streams as tokens."""
    ctx = _ctx(session_factory, provisioned_principal)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "list my jobs"}]

    events = [
        e async for e in run_agent(llm=_ScriptedLlm(), ctx=ctx, messages=messages, max_iterations=5)
    ]

    types = [e["type"] for e in events]
    assert "tool_start" in types and "tool_end" in types
    assert types[-1] == "done"
    assert any(e["type"] == "token" and "no jobs" in e["text"].lower() for e in events)


async def test_agent_emits_tool_start_before_tool_end_and_labels_the_tool(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """``tool_start`` carries the friendly ``TOOL_LABELS`` label and precedes ``tool_end``."""
    ctx = _ctx(session_factory, provisioned_principal)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "list my jobs"}]

    events = [
        e async for e in run_agent(llm=_ScriptedLlm(), ctx=ctx, messages=messages, max_iterations=5)
    ]

    start_index = next(i for i, e in enumerate(events) if e["type"] == "tool_start")
    end_index = next(i for i, e in enumerate(events) if e["type"] == "tool_end")
    assert start_index < end_index
    assert events[start_index]["name"] == "list_jobs"
    assert events[start_index]["label"] == "Looking up your jobs…"
    assert events[end_index]["name"] == "list_jobs"


async def test_agent_appends_assistant_and_tool_messages(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """The tool-call round mutates ``messages`` with an assistant turn and a tool result turn.

    ``_ScriptedLlm`` also answers with plain text on its second (final) round,
    which — per :func:`run_agent`'s plain-answer persistence — appends a
    trailing assistant message too; see
    ``test_agent_persists_a_plain_answer_to_thread_memory`` for that behavior
    in isolation.
    """
    ctx = _ctx(session_factory, provisioned_principal)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "list my jobs"}]

    [_ async for _ in run_agent(llm=_ScriptedLlm(), ctx=ctx, messages=messages, max_iterations=5)]

    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assistant_msg = messages[1]
    assert assistant_msg["content"] is None
    assert assistant_msg["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "list_jobs", "arguments": "{}"}}
    ]
    tool_msg = messages[2]
    assert tool_msg["tool_call_id"] == "c1"
    assert isinstance(tool_msg["content"], str)
    final_msg = messages[3]
    assert final_msg == {"role": "assistant", "content": "You have no jobs."}


async def test_agent_persists_a_plain_answer_to_thread_memory(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """A turn that never calls a tool still appends its own reply to ``messages``.

    ``ChatService`` persists ``messages`` as thread memory across turns, so a
    plain-answer round that is only streamed as tokens and never appended
    would make the model amnesiac about its own prior replies.
    """
    ctx = _ctx(session_factory, provisioned_principal)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]

    [
        _
        async for _ in run_agent(
            llm=_PlainAnswerLlm(), ctx=ctx, messages=messages, max_iterations=5
        )
    ]

    assert messages[-1] == {"role": "assistant", "content": "Hello there, how can I help?"}


async def test_agent_stops_at_iteration_cap(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """A model that always calls a tool is stopped after ``max_iterations`` rounds, not forever."""
    ctx = _ctx(session_factory, provisioned_principal)

    events = [
        e
        async for e in run_agent(
            llm=_AlwaysToolLlm(),
            ctx=ctx,
            messages=[{"role": "user", "content": "x"}],
            max_iterations=3,
        )
    ]

    assert sum(1 for e in events if e["type"] == "tool_start") == 3
    assert events[-1]["type"] in {"done", "error"}


async def test_agent_inserts_paragraph_break_after_tool_call(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """Post-tool text is prefixed with a paragraph break so it never fuses with pre-tool text."""
    ctx = _ctx(session_factory, provisioned_principal)

    events = [
        e
        async for e in run_agent(
            llm=_TalksThenCallsToolLlm(),
            ctx=ctx,
            messages=[{"role": "user", "content": "x"}],
            max_iterations=5,
        )
    ]

    tokens = [e["text"] for e in events if e["type"] == "token"]
    assert tokens == ["Sure, let me look.", "\n\nDone!"]


async def test_agent_falls_back_to_empty_args_on_malformed_tool_call_json(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """Unparseable ``arguments`` fall back to ``{}`` instead of crashing the loop."""
    ctx = _ctx(session_factory, provisioned_principal)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "x"}]

    events = [
        e
        async for e in run_agent(
            llm=_MalformedArgsLlm(), ctx=ctx, messages=messages, max_iterations=3
        )
    ]

    assert events[-1]["type"] == "done"
    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    # get_job requires job_id; falling back to {} makes run_tool report a
    # validation error rather than the agent raising one of its own.
    assert tool_messages[0]["content"].startswith("Error")


async def test_agent_emits_error_event_when_llm_is_unavailable(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """``LlmUnavailableError`` from the stream yields one clean ``error`` event, not a crash."""
    ctx = _ctx(session_factory, provisioned_principal)

    events = [
        e
        async for e in run_agent(
            llm=_UnavailableLlm(),
            ctx=ctx,
            messages=[{"role": "user", "content": "x"}],
            max_iterations=3,
        )
    ]

    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert isinstance(events[0]["message"], str) and events[0]["message"]


async def test_agent_emits_error_event_on_unexpected_stream_exception(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """Any unexpected exception from the stream is caught, not just ``LlmUnavailableError``."""
    ctx = _ctx(session_factory, provisioned_principal)

    events = [
        e
        async for e in run_agent(
            llm=_BoomLlm(), ctx=ctx, messages=[{"role": "user", "content": "x"}], max_iterations=3
        )
    ]

    assert len(events) == 1
    assert events[0]["type"] == "error"


def test_truncate_tool_result_leaves_short_results_untouched() -> None:
    """A result under the cap passes through unchanged."""
    assert _truncate_tool_result("short result") == "short result"


def test_truncate_tool_result_caps_oversized_results() -> None:
    """A result over the cap is cut to the limit and marked as truncated."""
    long_result = "a" * (MAX_TOOL_RESULT_CHARS + 100)

    truncated = _truncate_tool_result(long_result)

    assert truncated.startswith("a" * MAX_TOOL_RESULT_CHARS)
    assert truncated.endswith("[truncated — result too large]")
    assert len(truncated) < len(long_result)
