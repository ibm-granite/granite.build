# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Chat endpoints — the assistant's HTTP surface.

``POST /chat`` runs one turn to completion and returns the joined response;
``POST /chat/stream`` runs the same turn but streams it as server-sent events
so a client can render per-tool status and token-by-token output. Both bodies
delegate entirely to :class:`~autotunex.services.chat.service.ChatService`;
this module owns only the HTTP shape (SSE framing, headers, the trailing
``context`` event) — see that service's module docstring for the turn logic
itself. The router-level ``get_principal`` dependency (see ``main.py``) 401s
an unauthenticated caller before either body runs.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from http import HTTPStatus
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from autotunex.api.deps import ChatServiceDep, PrincipalDep
from autotunex.core.logging import get_logger
from autotunex.models.chat import ChatRequest, ChatResponse
from autotunex.models.common import ProblemDetail

logger = get_logger(__name__)

router = APIRouter(tags=["chat"])

_PROBLEM_RESPONSE = {"model": ProblemDetail, "content": {"application/problem+json": {}}}
_CHAT_RESPONSES: dict[int | str, dict[str, Any]] = {
    HTTPStatus.UNAUTHORIZED: _PROBLEM_RESPONSE,
    HTTPStatus.SERVICE_UNAVAILABLE: _PROBLEM_RESPONSE,
}


@router.post("/chat", summary="Send a chat turn", responses=_CHAT_RESPONSES)
async def chat_endpoint(
    body: ChatRequest, service: ChatServiceDep, principal: PrincipalDep
) -> ChatResponse:
    """Run one chat turn to completion and return the joined response."""
    return await service.chat(messages=body.messages, principal=principal, thread_id=body.thread_id)


@router.post(
    "/chat/stream",
    summary="Stream a chat turn as server-sent events",
    responses=_CHAT_RESPONSES,
)
async def chat_stream_endpoint(
    body: ChatRequest, service: ChatServiceDep, principal: PrincipalDep
) -> StreamingResponse:
    """Stream one chat turn as SSE frames: ``token``/``tool_start``/``tool_end``/``done``.

    Mirrors the 2025 ``chat_routes.py``'s ``event_source`` generator: any
    exception escaping the agent loop itself (as opposed to an ``"error"``
    event the agent already turned into a clean SSE frame) is caught, logged,
    and turned into one last ``error`` frame rather than propagating into a
    half-sent SSE response, which HTTP has no way to retroactively turn into a
    5xx. A trailing ``context`` frame always follows, echoing the caller's own
    ``context`` plus the last user message's text, whether the turn succeeded,
    failed cleanly, or raised.
    """
    last_input = body.messages[-1].content if body.messages else None

    async def event_source() -> AsyncIterator[str]:
        try:
            async for event in service.chat_stream(
                messages=body.messages, principal=principal, thread_id=body.thread_id
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception:
            logger.exception("Chat SSE generator failed")
            error_event = {"type": "error", "message": "Stream failed unexpectedly."}
            yield f"data: {json.dumps(error_event)}\n\n"
        finally:
            context_event = {
                "type": "context",
                "context": {**body.context, "last_input": last_input},
            }
            yield f"data: {json.dumps(context_event)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
