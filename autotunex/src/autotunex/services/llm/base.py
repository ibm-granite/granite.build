# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The provider-agnostic LLM client seam."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class ToolCallDelta:
    """One streamed fragment of a tool call, keyed by ``index``.

    OpenAI-compatible gateways stream a tool call across many chunks: the first
    carries ``id`` and ``name``; later chunks carry only ``arguments`` fragments
    to be concatenated. Consumers accumulate by ``index``.
    """

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str | None = None


@dataclass(slots=True)
class ChatDelta:
    """One streamed step: a text token, tool-call fragments, or a stop signal."""

    content: str | None = None
    tool_calls: list[ToolCallDelta] | None = None
    finish_reason: str | None = None


class LlmClient(Protocol):
    """A provider-agnostic single-shot chat completion."""

    async def complete(
        self, *, system: str, user: str, response_schema: dict[str, Any] | None = None
    ) -> str:
        """Return the model's text content for one system+user turn.

        When ``response_schema`` is supplied and the backend supports it, the
        adapter requests structured JSON output; callers must still parse and
        validate the returned text (the service is the single place that does).
        """
        ...

    def stream_chat(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[ChatDelta]:
        """Stream one assistant turn that may call tools.

        ``messages`` and ``tools`` follow the OpenAI ``/chat/completions`` shape.
        The agent owns the multi-round ReAct loop; this performs one streamed
        completion. Failures raise :class:`LlmUnavailableError`.
        """
        ...
