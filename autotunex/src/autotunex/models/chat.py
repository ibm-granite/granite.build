# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Chat API schemas — the request/response contract for the assistant endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """One conversation turn as sent by the client."""

    role: str = Field(description="Message role: 'user' or 'assistant'.")
    content: str | None = Field(default=None, description="Message text.")


class ChatRequest(BaseModel):
    """A chat turn: the visible history plus an optional server-memory thread id."""

    messages: list[ChatMessage] = Field(description="Conversation message history.")
    context: dict[str, Any] = Field(
        default_factory=dict, description="Opaque conversation context."
    )
    thread_id: str | None = Field(
        default=None,
        description="Stable conversation id enabling server-side tool-result memory across turns.",
    )


class ChatResponse(BaseModel):
    """The blocking (non-streaming) assistant response."""

    output: str = Field(description="Assistant response text.")
    context: dict[str, Any] = Field(
        default_factory=dict, description="Updated conversation context."
    )
