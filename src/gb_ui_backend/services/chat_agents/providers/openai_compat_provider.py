"""Runs the agentic tool-calling loop against any OpenAI-compatible chat
completions endpoint — RITS, Ollama, OpenAI itself, or any self-hosted
equivalent — via the existing LLMClient (services/llm_client.py), which
already builds correct URLs/headers for each of those (RITS_API_KEY header +
model-slug-in-path for RITS, Authorization: Bearer + plain /v1/chat/completions
otherwise). What's added here is handling `tool_calls` in the response and
looping — LLMClient.chat_completion() already threads a `tools` param into
the request payload.

Uses non-streaming chat_completion() once per loop round rather than
extending chat_completion_stream() for real SSE token streaming — that would
also require accumulating partial tool_calls[].function.arguments JSON
fragments across chunks, real added complexity that buys perceived
responsiveness, not correctness, since the frontend already treats one
text_delta as a whole finished message regardless of how it was produced.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

from gb_ui_backend.services.chat_agents.base import NormalizedEvent
from gb_ui_backend.services.chat_agents.tool_registry import (
    MAX_TOKENS,
    ToolSpec,
    race_interrupt,
    run_tool_loop,
)
from gb_ui_backend.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


def _to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


class OpenAICompatProvider:
    # Public — surfaced by ToolLoopBackend.describe() (GET /api/analytics/chat/status)
    # so the frontend can show which model/provider is actually running.
    PROVIDER_NAME = "openai_compatible"

    def __init__(
        self, base_url: str, api_key: str, model: str, system_prompt: str
    ) -> None:
        self._client = LLMClient(base_url=base_url, api_key=api_key, models=[model])
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
        if not history:
            history.append({"role": "system", "content": self._system_prompt})
        # Snapshotted AFTER the system-message bootstrap (that's not part
        # of "this turn") but BEFORE appending this turn's user message —
        # see run_tool_loop()'s docstring for why it can't take this
        # snapshot itself.
        original_length = len(history)
        history.append({"role": "user", "content": user_message})
        tool_by_name = {t.name: t for t in tools}
        openai_tools = _to_openai_tools(tools)

        async def _run_one_round(
            outcome: dict[str, Any],
        ) -> AsyncIterator[NormalizedEvent]:
            response = await race_interrupt(
                self._client.chat_completion(
                    messages=history, max_tokens=MAX_TOKENS, tools=openai_tools
                ),
                interrupt_event,
            )

            message = response["choices"][0]["message"]
            history.append(message)

            content = message.get("content")
            tool_calls = message.get("tool_calls") or []
            if not content and not tool_calls:
                # Some OpenAI-compatible endpoints return neither on a
                # stop/refusal — surface it instead of silently ending the
                # turn with nothing shown to the user; run_tool_loop rolls
                # back so the empty assistant turn doesn't linger in history.
                outcome["status"] = "empty"
                outcome["message"] = (
                    "Model returned an empty response with no content or tool calls."
                )
                return

            if content:
                yield {"type": "text_delta", "text": content}

            if not tool_calls:
                outcome["status"] = "done"
                return

            for call in tool_calls:
                # .get(), not [...] — some OpenAI-compatible endpoints
                # (this provider's whole point is supporting less
                # standardized ones — RITS, Ollama, self-hosted) can
                # return a malformed tool_call; degrade to a clean tool
                # error instead of an unhandled KeyError killing the turn.
                call_id = call.get("id", "")
                fn = call.get("function") or {}
                tool_name = fn.get("name") or "<missing tool name>"
                raw_arguments = fn.get("arguments") or "{}"
                try:
                    # Some OpenAI-compatible endpoints hand back
                    # `arguments` already parsed as an object rather
                    # than a JSON string — json.loads(dict) raises
                    # TypeError, not JSONDecodeError, which used to
                    # escape this handler entirely and roll back the
                    # whole turn instead of degrading to a clean tool
                    # error like the malformed-JSON-string case below.
                    args = (
                        raw_arguments
                        if isinstance(raw_arguments, dict)
                        else json.loads(raw_arguments)
                    )
                    content: str | None = None
                except (json.JSONDecodeError, TypeError):
                    # Still yield tool_call below with whatever we've got
                    # — every attempt gets exactly one, whether it fails
                    # here, on an unknown tool name, or inside the
                    # handler, so a future UI/consumer of this event
                    # stream sees a consistent shape for all three.
                    args = {}
                    content = f"Invalid JSON arguments: {raw_arguments!r}"

                yield {
                    "type": "tool_call",
                    "tool_name": tool_name,
                    "tool_input": args,
                }

                if content is None:
                    tool = tool_by_name.get(tool_name)
                    if tool is None:
                        content = f"Unknown tool {tool_name!r}"
                    else:
                        try:
                            # Also interruptible — a tool like wait_for_build
                            # can run for up to 30 minutes; without this,
                            # /chat/stop would only take effect once it
                            # finished on its own.
                            content = str(
                                await race_interrupt(
                                    tool.handler(args), interrupt_event
                                )
                            )
                        except InterruptedError:
                            raise  # let run_tool_loop roll history back — don't treat this as a tool error
                        except (
                            Exception
                        ) as exc:  # noqa: BLE001 - surfaced to the model as a tool error, not raised
                            content = str(exc)

                history.append(
                    {"role": "tool", "tool_call_id": call_id, "content": content}
                )
                while not event_queue.empty():
                    yield event_queue.get_nowait()

            outcome["status"] = "continue"

        return run_tool_loop(
            history, event_queue, original_length, _run_one_round, "OpenAI-compatible"
        )
