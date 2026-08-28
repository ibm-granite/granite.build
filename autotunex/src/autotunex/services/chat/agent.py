# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The native OpenAI-style tool-calling agent loop for AutoTuneX chat.

Runs a standard tool-calling loop against a provider-agnostic
:class:`~autotunex.services.llm.base.LlmClient`: stream one assistant turn,
forward its text as ``token`` events, and — when the model asks for tool
calls — execute them through the shared tool registry
(:mod:`autotunex.services.chat.tools`) and feed the results back for another
round. Conversation state lives entirely in the ``messages`` list passed in,
which is mutated in place (assistant tool-call turns and tool result turns
are appended as the loop runs) so the caller can persist the final list once
the generator is exhausted.

Ported from the 2025 ``chat_service.py``'s ``chat_stream`` (a LangGraph
``astream_events`` consumer): the ~50k-char tool-result truncation guard and
the post-tool paragraph-break separator (so text streamed after a tool call
doesn't fuse onto the pre-tool text, e.g. "…right IDs!Got everything!") are
both preserved here, adapted to this native loop instead of LangGraph's event
stream.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from autotunex.core.exceptions import LlmUnavailableError
from autotunex.services.chat.context import ToolContext
from autotunex.services.chat.tools import (
    TOOL_LABELS,
    TOOL_REFRESH_TARGETS,
    openai_tool_specs,
    run_tool,
)
from autotunex.services.llm.base import LlmClient

logger = logging.getLogger(__name__)

AgentEvent = dict[str, Any]
"""One streamed agent event. ``type`` selects which other keys are present.

- ``{"type": "token", "text": str}`` — a fragment of assistant text.
- ``{"type": "tool_start", "name": str, "label": str}`` — a tool call began.
- ``{"type": "tool_end", "name": str}`` — that tool call finished.
- ``{"type": "refresh", "target": str}`` — a write tool succeeded; the UI
  should refresh the named view (see ``TOOL_REFRESH_TARGETS``).
- ``{"type": "error", "message": str}`` — the turn failed; no more events
  follow (``done`` is not emitted after an ``error``).
- ``{"type": "done"}`` — the turn is over; safe to persist ``messages``.
"""

MAX_TOOL_RESULT_CHARS = 50_000
"""Cap on one tool result's length before it is appended to ``messages``.

A training-log dump or a large trial-results listing can otherwise consume a
large fraction of the model's context window in a single tool turn. Ported
verbatim (same cutoff) from the 2025 ``chat_service.py``'s
``_prepare_messages`` guard.
"""

_TRUNCATION_SUFFIX = "\n\n... [truncated — result too large]"

_UNEXPECTED_ERROR_MESSAGE = (
    "I'm sorry, I encountered an error while processing your request. "
    "Please try again or rephrase your question."
)


@dataclass(slots=True)
class _AccumulatedToolCall:
    """One tool call being assembled across streamed ``ToolCallDelta`` fragments.

    The first fragment for a given ``index`` carries ``id``/``name``; every
    fragment (including the first) may carry an ``arguments`` piece to
    concatenate — see :class:`~autotunex.services.llm.base.ToolCallDelta`.
    """

    id: str | None = None
    name: str | None = None
    arguments: str = ""


def _truncate_tool_result(result: str) -> str:
    """Cap ``result`` at :data:`MAX_TOOL_RESULT_CHARS`, appending a marker if cut."""
    if len(result) <= MAX_TOOL_RESULT_CHARS:
        return result
    return result[:MAX_TOOL_RESULT_CHARS] + _TRUNCATION_SUFFIX


async def run_agent(
    *,
    llm: LlmClient,
    ctx: ToolContext,
    messages: list[dict[str, Any]],
    max_iterations: int,
) -> AsyncIterator[AgentEvent]:
    """Run the tool-calling loop, streaming events and mutating ``messages`` in place.

    Each of up to ``max_iterations`` rounds streams one assistant turn from
    ``llm``. A turn that asks for tool calls is executed and fed back for
    another round; a turn that does not ends the loop immediately (its own
    completion never counts against the cap — only tool-call rounds do, so a
    conversation that never calls a tool is never truncated by this limit).
    Exhausting the cap while every round has been a tool call simply stops
    the loop without a further model call, rather than looping forever.

    A failure while streaming — including
    :class:`~autotunex.core.exceptions.LlmUnavailableError` and any other
    unexpected exception the client raises — yields exactly one ``error``
    event and returns, without a following ``done``, so a caller can tell a
    clean stop from a failed one.

    Args:
        llm: The provider-agnostic chat client to stream turns from.
        ctx: Builds the Principal-scoped services each tool call runs against.
        messages: The running conversation, OpenAI ``role``/``content`` shape.
            Appended to in place with the assistant tool-call turn and each
            tool's result turn, or — for a round that ends in a plain answer —
            a single assistant turn carrying that answer's text, so the caller
            can persist it once this generator is exhausted.
        max_iterations: The maximum number of tool-call rounds to run before
            stopping without a further model call.

    Yields:
        :data:`AgentEvent` values describing the turn as it streams.
    """
    just_finished_tool = False
    streaming_answer = False

    for _round in range(max_iterations):
        tools = openai_tool_specs()
        fragments: dict[int, _AccumulatedToolCall] = {}
        finish_reason: str | None = None
        content_parts: list[str] = []

        try:
            async for delta in llm.stream_chat(messages=messages, tools=tools):
                if delta.content:
                    text = delta.content
                    content_parts.append(text)
                    if just_finished_tool:
                        # The model's first post-tool token carries no leading
                        # whitespace, so without this the pre- and post-tool
                        # text would fuse into one run-on sentence.
                        if streaming_answer:
                            text = "\n\n" + text.lstrip()
                        just_finished_tool = False
                    streaming_answer = True
                    yield {"type": "token", "text": text}
                if delta.tool_calls:
                    for fragment in delta.tool_calls:
                        acc = fragments.setdefault(fragment.index, _AccumulatedToolCall())
                        if fragment.id is not None:
                            acc.id = fragment.id
                        if fragment.name is not None:
                            acc.name = fragment.name
                        if fragment.arguments is not None:
                            acc.arguments += fragment.arguments
                if delta.finish_reason is not None:
                    finish_reason = delta.finish_reason
        except LlmUnavailableError as exc:
            yield {"type": "error", "message": str(exc)}
            return
        except Exception:
            logger.exception("Chat agent's LLM stream failed unexpectedly.")
            yield {"type": "error", "message": _UNEXPECTED_ERROR_MESSAGE}
            return

        if finish_reason != "tool_calls":
            # The model answered without calling a tool: this round's text is
            # the assistant's actual reply, so it must join `messages` like any
            # other turn — otherwise `ChatService`'s persisted thread memory
            # silently drops every plain-answer turn, breaking multi-turn
            # coherence (the model can't see what it told the user last time).
            final_text = "".join(content_parts)
            if final_text:
                messages.append({"role": "assistant", "content": final_text})
            break

        ordered = sorted(fragments.items(), key=lambda item: item[0])
        messages.append(
            {
                "role": "assistant",
                "content": "".join(content_parts) or None,
                "tool_calls": [
                    {
                        "id": acc.id or f"call_{index}",
                        "type": "function",
                        "function": {"name": acc.name or "", "arguments": acc.arguments},
                    }
                    for index, acc in ordered
                ],
            }
        )

        for index, acc in ordered:
            name = acc.name or ""
            call_id = acc.id or f"call_{index}"
            label = TOOL_LABELS.get(name, f"Running {name}…")
            yield {"type": "tool_start", "name": name, "label": label}

            try:
                arguments = json.loads(acc.arguments) if acc.arguments else {}
                if not isinstance(arguments, dict):
                    arguments = {}
            except json.JSONDecodeError:
                arguments = {}

            result = await run_tool(name, arguments, ctx)
            yield {"type": "tool_end", "name": name}

            if name in TOOL_REFRESH_TARGETS and not result.startswith("Error"):
                yield {"type": "refresh", "target": TOOL_REFRESH_TARGETS[name]}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _truncate_tool_result(result),
                }
            )
            just_finished_tool = True

    yield {"type": "done"}
