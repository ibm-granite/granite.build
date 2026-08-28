# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""ChatService — orchestrates the agent loop, thread memory, and SSE event shaping.

This is the seam between the HTTP layer (``api/routers/chat.py``, Task 10 — not
yet written) and the native tool-calling agent (:mod:`autotunex.services.chat.agent`).
It owns exactly two responsibilities the agent itself does not: building the
working ``messages`` list for a turn (seeding it from either server-side thread
memory or the client's own history) and persisting that list back to memory once
the turn completes. It knows nothing about ``fastapi`` — both the streaming and
blocking chat endpoints call the same two methods here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from autotunex.core.config import Settings
from autotunex.models.auth import Principal
from autotunex.models.chat import ChatMessage, ChatResponse
from autotunex.services.chat.agent import run_agent
from autotunex.services.chat.context import ToolContext
from autotunex.services.chat.memory import ConversationMemory
from autotunex.services.llm.base import LlmClient

SYSTEM_PROMPT = (
    "You are AutoTuneX Assistant, a concise AI helper for IBM's automated "
    "LLM fine-tuning platform.\n\n"
    "User email: {user_email}\n\n"
    "RULES:\n"
    "1. Be CONVERSATIONAL. Keep responses short for explanations, but "
    "when the user asks for a list (datasets, configs, jobs, etc.) "
    "ALWAYS show the full list of names from the tool output — never "
    "summarize with just a count. Never dump long tutorials or code blocks.\n"
    "2. USE YOUR TOOLS to perform actions — do not explain how to call APIs "
    "or write sample code. You ARE the interface.\n"
    "3. GATHER INFO STEP BY STEP, AND SHOW WHAT YOU ASK ABOUT. If the user "
    "asks to do something but information is missing, resolve ONE missing "
    "piece at a time. Whenever the missing piece is a choice among the "
    "user's existing resources, CALL THE LOOKUP TOOL AND PRESENT THE LIST "
    "in the SAME turn, then ask which one to use — never ask the user to "
    "choose before you have shown the options. For example, to fine-tune a "
    "model: first call list_datasets and present the datasets, then ask "
    "which to use; after they choose, call list_configs and present the "
    "configurations, then ask which to use; then confirm and start. NEVER "
    "ask the user to choose from a list you have not shown them in this "
    "conversation, and NEVER refer to a list as 'above' or as already "
    "shown unless a tool call in THIS conversation actually produced it. "
    "IMPORTANT: if you have already called a "
    "lookup tool earlier in THIS conversation, reuse those results instead "
    "of re-calling the tool. Only re-fetch when the user asks for fresh "
    "data or references something you haven't looked up yet.\n"
    "4. Before any destructive or expensive action (starting a job, deleting "
    "something), state what you're about to do in one sentence and ask "
    "for confirmation.\n"
    "5. If a tool call fails, explain the error briefly and ask the user "
    "how to proceed.\n"
    "6. ACCURACY IS CRITICAL: When presenting data from tool results "
    "(names, IDs, counts, values), reproduce them EXACTLY as returned by "
    "the tool. NEVER invent, guess, extrapolate, or paraphrase names or "
    "IDs. If a list tool returns items, present only those exact items — "
    "do not continue patterns or fill in gaps.\n"
    "7. FORMATTING: Tool results are pre-formatted with markdown (bold "
    "names, numbered lists, inline code). Present the tool output "
    "directly — preserve the line breaks and formatting. Do NOT collapse "
    "a numbered list into a single comma-separated paragraph."
)
"""The assistant's system prompt, ``{user_email}`` filled in per turn.

Ported from the 2025 ``chat_service.py``'s ``SYSTEM_PROMPT`` (rules 1-5, 7-8
there, renumbered 1-7 here). Rule 6 there ("Always pass the user's email when
tools require user_email") is dropped: every 2026 tool is bound to the calling
``Principal`` through :class:`~autotunex.services.chat.context.ToolContext`
rather than taking an email argument, so no tool has a ``user_email`` parameter
to pass — see the module docstring in ``services/chat/tools.py``.
"""


def _memory_key(principal: Principal, thread_id: str | None) -> str | None:
    """Bind ``thread_id`` to the caller's identity, or return ``None`` when memory is off.

    ``thread_id`` is client-supplied and therefore untrusted: without this,
    two different authenticated callers who happened to pass the same
    ``thread_id`` would read and write the exact same
    :class:`ConversationMemory` bucket, letting one caller's prior turns leak
    into another caller's conversation. Namespacing every key by the caller's
    identity — ``user_id`` when resolved, else the caller's email, else the
    literal ``"standalone"`` for an unresolvable standalone caller — makes
    that cross-caller collision structurally impossible while leaving
    ``ConversationMemory`` itself generic (it just stores whatever key it's
    given).

    Returns ``None`` for a falsy ``thread_id`` so callers can keep using it
    as the "is memory enabled for this turn" flag, unchanged from before.
    """
    if not thread_id:
        return None
    ident = (
        str(principal.user_id)
        if principal.user_id is not None
        else (principal.email or "standalone")
    )
    return f"{ident}::{thread_id}"


class ChatService:
    """Runs one chat turn end to end: message state, the agent loop, memory.

    Stateless itself between calls — all continuity across turns lives in the
    injected :class:`ConversationMemory`, keyed by the caller-supplied
    ``thread_id``. A caller that never sends a ``thread_id`` gets the 2025
    "legacy" behavior: the full client-sent history is replayed every turn and
    nothing is persisted server-side.
    """

    def __init__(self, *, llm: LlmClient, memory: ConversationMemory, settings: Settings) -> None:
        """Wire the collaborators this service orchestrates.

        Args:
            llm: The provider-agnostic chat client the agent streams turns from.
            memory: The bounded per-thread store for server-side conversation state.
            settings: Used for ``chat_max_tool_iterations`` and to build the
                Principal-scoped :class:`ToolContext` for each turn.
        """
        self._llm = llm
        self._memory = memory
        self._settings = settings

    async def chat_stream(
        self,
        *,
        messages: list[ChatMessage],
        principal: Principal,
        thread_id: str | None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run one turn, yielding the agent's SSE-shaped events as they occur.

        Builds the working ``messages`` list, then delegates the tool-calling
        loop to :func:`~autotunex.services.chat.agent.run_agent`, forwarding
        every event it yields unchanged. When the turn ends cleanly (a
        ``"done"`` event) and ``thread_id`` was supplied, the mutated working
        list is persisted to :attr:`_memory` so the next turn on the same
        thread can resume from it — an ``"error"`` event, by contrast, is never
        followed by ``"done"`` (see ``run_agent``), so a failed turn is never
        persisted.

        Args:
            messages: The client's visible conversation, ending with the new
                user turn. Only actually read in full when ``thread_id`` is
                unset or names a thread with no stored state yet — see the
                class docstring.
            principal: The caller, used to scope every tool call and to fill
                ``{user_email}`` in :data:`SYSTEM_PROMPT`.
            thread_id: Stable id for server-side memory across turns, or
                ``None`` to always replay the client's own history. Namespaced
                by the caller's identity before touching memory (see
                :func:`_memory_key`) — ``thread_id`` alone is client-supplied
                and must never let one caller read another's stored thread.

        Yields:
            The same event dicts :func:`run_agent` yields — this method adds
            no event types of its own.
        """
        system_message: dict[str, Any] = {
            "role": "system",
            "content": SYSTEM_PROMPT.format(user_email=principal.email or "unknown"),
        }
        newest_user: dict[str, Any] = (
            {"role": messages[-1].role, "content": messages[-1].content}
            if messages
            else {"role": "user", "content": None}
        )

        memory_key = _memory_key(principal, thread_id)
        prior = self._memory.get(memory_key) if memory_key else []
        working: list[dict[str, Any]]
        if prior:
            # Existing thread state already holds its own system message plus
            # every earlier turn (including tool messages) — replaying the
            # client's echoed history on top would duplicate it, so only the
            # newest user turn is appended. Mirrors the 2025 LangGraph
            # checkpointer's "has_state" branch.
            #
            # Copy rather than alias: `memory.get` returns the SAME list object
            # it stores internally (no copy — see ConversationMemory.get), and
            # `run_agent` mutates `working` in place as the turn runs. Without
            # this copy, a turn that fails partway (no "done", so `memory.put`
            # below is never reached) would still have already corrupted the
            # thread's persisted history via that shared reference — violating
            # "persist on done, never on error" for turns that build on
            # existing state (a from-scratch turn has no such risk, since its
            # `working` list is freshly constructed below either way).
            working = list(prior)
            working.append(newest_user)
        else:
            working = [system_message]
            working.extend({"role": m.role, "content": m.content} for m in messages)

        ctx = ToolContext.for_principal(principal, self._settings)

        async for event in run_agent(
            llm=self._llm,
            ctx=ctx,
            messages=working,
            max_iterations=self._settings.chat_max_tool_iterations,
        ):
            if event["type"] == "done" and memory_key:
                self._memory.put(memory_key, working)
            yield event

    async def chat(
        self,
        *,
        messages: list[ChatMessage],
        principal: Principal,
        thread_id: str | None,
    ) -> ChatResponse:
        """Run one turn to completion and return the joined, non-streaming response.

        Consumes :meth:`chat_stream` fully, concatenating every ``"token"``
        event's text. A turn that ends in an ``"error"`` event instead of
        ``"done"`` still returns a normal :class:`ChatResponse` — its
        ``output`` is the error's message — rather than raising, so this
        method never surfaces an exception the agent already turned into a
        clean event.

        Args:
            messages: See :meth:`chat_stream`.
            principal: See :meth:`chat_stream`.
            thread_id: See :meth:`chat_stream`.

        Returns:
            The assistant's joined text as ``output``, with an empty ``context``.
        """
        output_parts: list[str] = []
        error_message: str | None = None

        async for event in self.chat_stream(
            messages=messages, principal=principal, thread_id=thread_id
        ):
            if event["type"] == "token":
                output_parts.append(event["text"])
            elif event["type"] == "error":
                error_message = event["message"]

        output = error_message if error_message is not None else "".join(output_parts)
        return ChatResponse(output=output, context={})
