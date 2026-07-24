# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from auth import get_current_user
import models as api

router = APIRouter()


@router.post(
    "/api/chat",
    tags=["Chat"],
    summary="Chat with AutoTuneX Assistant",
    response_description="AI assistant response with optional tool call details",
)
async def chat_endpoint(
    request: api.ChatRequest,
    auth_user: api.AuthUser = Depends(get_current_user),
):
    """Send messages to the AutoTuneX AI assistant. The assistant can query and
    manage your jobs, configurations, datasets, and models through natural language."""
    from services import chat_service

    result = await chat_service.chat(
        messages=[m.model_dump(exclude_none=True) for m in request.messages],
        user_email=auth_user.email,
        context=request.context or {},
        thread_id=request.thread_id,
    )
    return result


@router.post(
    "/api/chat/stream",
    tags=["Chat"],
    summary="Stream AutoTuneX Assistant responses via SSE",
    response_description="Server-sent events: tool_start, tool_end, token, done, error",
)
async def chat_stream_endpoint(
    request: api.ChatRequest,
    auth_user: api.AuthUser = Depends(get_current_user),
):
    """Stream the AutoTuneX AI assistant turn as server-sent events so the UI
    can display per-tool status and token-by-token output."""
    from services import chat_service

    messages = [m.model_dump(exclude_none=True) for m in request.messages]
    user_email = auth_user.email
    user_context = request.context or {}
    user_input = messages[-1].get("content", "") if messages else ""
    thread_id = request.thread_id

    async def event_source():
        try:
            async for event in chat_service.chat_stream(
                messages=messages,
                user_email=user_email,
                context=user_context,
                thread_id=thread_id,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logging.getLogger("chat_stream").exception("SSE generator failed: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'message': 'Stream failed unexpectedly.'})}\n\n"
        finally:
            final_context = {**user_context, "last_input": user_input}
            yield f"data: {json.dumps({'type': 'context', 'context': final_context})}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
