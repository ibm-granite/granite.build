"""Tests for :mod:`autotunex.services.chat.service`.

``ChatService.chat_stream`` builds a :class:`~autotunex.services.chat.context.ToolContext`
via ``ToolContext.for_principal``, which binds to the process-wide session factory
(``autotunex.db.session.get_session_factory``) rather than to this test module's
``session_factory``/``engine`` fixtures — there is no way to point that global at
the in-memory test database from here. Every fake LLM below therefore answers
directly, with ``finish_reason="stop"`` and no tool calls, so ``run_agent`` never
reaches ``run_tool`` / ``ctx.services()`` and the global factory is never touched.
This keeps the tests focused on what this module actually owns: message-list
construction, thread-memory reuse, event forwarding, and error surfacing — not
tool execution, which is already covered by ``test_agent.py`` and ``test_tools.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from autotunex.core.config import Settings
from autotunex.core.exceptions import LlmUnavailableError
from autotunex.models.auth import Principal
from autotunex.models.chat import ChatMessage
from autotunex.services.chat.memory import ConversationMemory
from autotunex.services.chat.service import SYSTEM_PROMPT, ChatService
from autotunex.services.llm.base import ChatDelta


class _AnswersLlm:
    """Answers immediately with scripted text; never asks for a tool call.

    ``turns`` is one list of content chunks per expected ``stream_chat`` call
    (each chunk streamed as its own ``ChatDelta``, letting a single-turn test
    verify that ``chat`` joins multiple token events). Records a snapshot of
    the ``messages`` list it was called with on every call, so tests can
    inspect exactly what ``ChatService`` built for each turn.
    """

    def __init__(self, turns: list[list[str]]) -> None:
        self._turns = list(turns)
        self.seen_messages: list[list[dict[str, Any]]] = []

    async def complete(self, **_kw: object) -> str:  # unused Protocol member
        return ""

    async def stream_chat(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[ChatDelta]:
        self.seen_messages.append([dict(m) for m in messages])
        chunks = self._turns.pop(0) if self._turns else ["..."]
        for chunk in chunks:
            yield ChatDelta(content=chunk)
        yield ChatDelta(finish_reason="stop")


class _ErrorLlm:
    """Raises ``LlmUnavailableError`` on its first (and only expected) call."""

    def __init__(self) -> None:
        self._turn = 0

    async def complete(self, **_kw: object) -> str:  # unused Protocol member
        return ""

    async def stream_chat(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[ChatDelta]:
        self._turn += 1
        if self._turn == 1:
            raise LlmUnavailableError()
        yield ChatDelta(finish_reason="stop")


def _memory() -> ConversationMemory:
    return ConversationMemory(max_threads=10, ttl_seconds=100.0)


def _settings() -> Settings:
    return Settings(job_backend="none")


async def test_chat_stream_emits_tokens_and_done(provisioned_principal: Principal) -> None:
    """A plain (no tool call) turn streams token events and ends with ``done``."""
    svc = ChatService(llm=_AnswersLlm([["Hello there."]]), memory=_memory(), settings=_settings())

    events = [
        e
        async for e in svc.chat_stream(
            messages=[ChatMessage(role="user", content="hi")],
            principal=provisioned_principal,
            thread_id="t1",
        )
    ]

    assert events[-1]["type"] == "done"
    assert any(e["type"] == "token" and e["text"] == "Hello there." for e in events)


async def test_chat_returns_joined_output(provisioned_principal: Principal) -> None:
    """The blocking ``chat`` path joins every streamed token's text into ``output``."""
    svc = ChatService(
        llm=_AnswersLlm([["Hello ", "there."]]), memory=_memory(), settings=_settings()
    )

    response = await svc.chat(
        messages=[ChatMessage(role="user", content="hi")],
        principal=provisioned_principal,
        thread_id=None,
    )

    assert response.output == "Hello there."
    assert response.context == {}


async def test_thread_memory_is_reused_across_turns(provisioned_principal: Principal) -> None:
    """A second turn on the same ``thread_id`` sees the first turn's full history, once.

    Without state, ``ChatService`` seeds ``[system] + client messages``. With
    state (this second call), it must reuse the stored list as-is — including
    the first turn's own plain-answer assistant reply, per ``run_agent``'s
    persistence of non-tool-call turns (see ``test_agent.py``'s
    ``test_agent_persists_a_plain_answer_to_thread_memory``) — and append only
    the newest user message, never prepending a second system message.
    """
    llm = _AnswersLlm([["First answer."], ["Second answer."]])
    svc = ChatService(llm=llm, memory=_memory(), settings=_settings())

    [
        _
        async for _ in svc.chat_stream(
            messages=[ChatMessage(role="user", content="turn one")],
            principal=provisioned_principal,
            thread_id="shared",
        )
    ]
    [
        _
        async for _ in svc.chat_stream(
            messages=[ChatMessage(role="user", content="turn two")],
            principal=provisioned_principal,
            thread_id="shared",
        )
    ]

    assert len(llm.seen_messages) == 2
    first_call, second_call = llm.seen_messages
    assert [m["role"] for m in first_call] == ["system", "user"]
    assert first_call[1]["content"] == "turn one"
    # Second call reuses the first call's exact system message (not a fresh
    # duplicate), carries the first turn's own assistant reply, and appends
    # only the new user turn on top of that.
    assert [m["role"] for m in second_call] == ["system", "user", "assistant", "user"]
    assert second_call[0] == first_call[0]
    assert second_call[1]["content"] == "turn one"
    assert second_call[2]["content"] == "First answer."
    assert second_call[3]["content"] == "turn two"


async def test_thread_memory_is_isolated_per_caller(provisioned_principal: Principal) -> None:
    """Two different callers who happen to share a ``thread_id`` never share memory.

    Regression test for the cross-user isolation gap: ``thread_id`` is
    arbitrary client input, so without binding it to the caller's identity
    (see ``ChatService._memory_key`` / ``service._memory_key``), a second
    caller who reused someone else's ``thread_id`` would have that other
    caller's stored conversation replayed straight into their own turn.
    """
    memory = _memory()
    principal_a = provisioned_principal
    principal_b = principal_a.model_copy(
        update={"user_id": uuid4(), "email": "someone-else@autotunex.local"}
    )
    shared_thread_id = "shared-thread-id"

    svc_a = ChatService(
        llm=_AnswersLlm([["Secret answer for A."]]), memory=memory, settings=_settings()
    )
    [
        _
        async for _ in svc_a.chat_stream(
            messages=[ChatMessage(role="user", content="A's private question")],
            principal=principal_a,
            thread_id=shared_thread_id,
        )
    ]
    stored_for_a = [dict(m) for m in memory.get(f"{principal_a.user_id}::{shared_thread_id}")]
    assert stored_for_a  # sanity: A's turn actually persisted something

    llm_b = _AnswersLlm([["B's own answer."]])
    svc_b = ChatService(llm=llm_b, memory=memory, settings=_settings())
    [
        _
        async for _ in svc_b.chat_stream(
            messages=[ChatMessage(role="user", content="B's own question")],
            principal=principal_b,
            thread_id=shared_thread_id,
        )
    ]

    # B's turn must have been seeded fresh — no trace of A's history.
    [b_call] = llm_b.seen_messages
    assert [m["role"] for m in b_call] == ["system", "user"]
    assert b_call[1]["content"] == "B's own question"
    assert all(m.get("content") != "A's private question" for m in b_call)
    assert all(m.get("content") != "Secret answer for A." for m in b_call)

    # A's stored memory is untouched by B's turn.
    assert [
        dict(m) for m in memory.get(f"{principal_a.user_id}::{shared_thread_id}")
    ] == stored_for_a


async def test_stream_surfaces_llm_error(provisioned_principal: Principal) -> None:
    """An ``LlmUnavailableError`` from the agent's stream surfaces as one ``error`` event."""
    svc = ChatService(llm=_ErrorLlm(), memory=_memory(), settings=_settings())

    events = [
        e
        async for e in svc.chat_stream(
            messages=[ChatMessage(role="user", content="hi")],
            principal=provisioned_principal,
            thread_id=None,
        )
    ]

    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert isinstance(events[0]["message"], str) and events[0]["message"]


async def test_chat_surfaces_error_message_as_output(provisioned_principal: Principal) -> None:
    """The blocking ``chat`` path returns the error's message as ``output`` rather than raising."""
    svc = ChatService(llm=_ErrorLlm(), memory=_memory(), settings=_settings())

    response = await svc.chat(
        messages=[ChatMessage(role="user", content="hi")],
        principal=provisioned_principal,
        thread_id=None,
    )

    assert response.output == str(LlmUnavailableError())


async def test_error_turn_is_not_persisted_to_thread_memory(
    provisioned_principal: Principal,
) -> None:
    """A turn that ends in ``error`` (no ``done``) leaves the thread's stored memory untouched."""
    memory = _memory()
    svc = ChatService(llm=_ErrorLlm(), memory=memory, settings=_settings())

    [
        _
        async for _ in svc.chat_stream(
            messages=[ChatMessage(role="user", content="hi")],
            principal=provisioned_principal,
            thread_id="fails",
        )
    ]

    assert memory.get(f"{provisioned_principal.user_id}::fails") == []


async def test_failing_turn_does_not_corrupt_existing_thread_memory(
    provisioned_principal: Principal,
) -> None:
    """A failing turn on a thread that ALREADY has history must not mutate that history.

    Regression test for an aliasing bug: ``ConversationMemory.get`` returns the
    same list object it stores internally (no copy), so building ``working``
    from it without copying let ``run_agent`` mutate the stored entry in place
    as the turn ran — before the turn's outcome was known. A second turn that
    then failed (no ``done``, so ``memory.put`` is never reached) had already
    leaked its user message into the thread's persisted memory despite that.
    """
    memory = _memory()
    successful_svc = ChatService(
        llm=_AnswersLlm([["First answer."]]), memory=memory, settings=_settings()
    )
    [
        _
        async for _ in successful_svc.chat_stream(
            messages=[ChatMessage(role="user", content="turn one")],
            principal=provisioned_principal,
            thread_id="t",
        )
    ]
    memory_key = f"{provisioned_principal.user_id}::t"
    stored_after_success = [dict(m) for m in memory.get(memory_key)]
    assert stored_after_success  # sanity: turn one actually persisted something

    failing_svc = ChatService(llm=_ErrorLlm(), memory=memory, settings=_settings())
    [
        _
        async for _ in failing_svc.chat_stream(
            messages=[ChatMessage(role="user", content="turn two")],
            principal=provisioned_principal,
            thread_id="t",
        )
    ]

    assert memory.get(memory_key) == stored_after_success
    assert all(m.get("content") != "turn two" for m in memory.get(memory_key))


def test_system_prompt_includes_user_email(provisioned_principal: Principal) -> None:
    """The formatted system prompt embeds the caller's email."""
    assert provisioned_principal.email is not None

    prompt = SYSTEM_PROMPT.format(user_email=provisioned_principal.email)

    assert provisioned_principal.email in prompt


def test_system_prompt_falls_back_to_unknown_placeholder() -> None:
    """Formatting with the ``"unknown"`` fallback (a principal with no email) still succeeds."""
    prompt = SYSTEM_PROMPT.format(user_email="unknown")

    assert "User email: unknown" in prompt


def test_system_prompt_directs_presenting_the_list_before_asking_a_choice() -> None:
    """The prompt tells the model to call the lookup tool and show the list in the same turn.

    Regression guard for the observed failure where, on "I want to fine-tune X",
    the assistant asked the user to pick a dataset/config without ever calling
    ``list_datasets``/``list_configs`` — leaving them nothing to pick from. The
    fix must keep instructing the model to present the list itself, so lock that
    intent here (the fake-LLM suite cannot exercise real model behavior).
    """
    prompt = SYSTEM_PROMPT.format(user_email="unknown")

    assert "PRESENT THE LIST" in prompt
    assert "list_datasets" in prompt
    assert "list_configs" in prompt


def test_system_prompt_forbids_referring_to_a_list_not_shown_in_the_conversation() -> None:
    """The prompt bans the "select from the list above" hallucination directly.

    The observed failure pointed the user at a list "above" that no tool call
    had produced and the UI never renders. The prompt must forbid referring to
    any list not actually shown in the current conversation.
    """
    prompt = SYSTEM_PROMPT.format(user_email="unknown")

    assert "have not shown" in prompt
    assert "'above'" in prompt
