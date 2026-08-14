"""Chat assistant API — streams ChatAgentBackend responses as SSE.

Mirrors plans.py's gating pattern (503 when unconfigured). The route only
ever talks to the ChatAgentBackend interface via get_backend() — it never
imports a concrete backend module directly.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from gb_ui_backend.config import get_config
from gb_ui_backend.services.chat_agents import get_backend
from gb_ui_backend.services.chat_agents.base import NormalizedEvent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat")

# Defensive upper bound on browser-supplied route info — these are always
# short (a dashboard pathname/query string), so a much longer value signals
# something wrong rather than a legitimate route worth accepting.
_MAX_PAGE_FIELD_LEN = 512


def _resolve_identity(request: Request) -> str:
    """Same precedence as analytics.py's get_current_author and ai.py's
    _rate_limit_analyze_logs: gbserver's AuthMiddleware-trusted user when
    this app is mounted inside gbserver, the X-User-Email header as a
    fallback for running standalone outside gbserver, "standalone" for
    apikey/localhost mode with no per-user identity at all."""
    user = getattr(request.state, "data", {}).get("user")
    if user is not None:
        return user.email
    return request.headers.get("x-user-email") or "standalone"


def _scoped_session_id(request: Request, session_id: str) -> str:
    """Namespaces the client-supplied session_id by the caller's trusted
    identity before it ever reaches ToolLoopBackend.

    ToolLoopBackend._sessions is a plain dict keyed on whatever string it's
    given — it has no concept of "who owns this session," so without this,
    any caller who learned another user's raw session_id could resolve or
    act on their session (including approving a pending build_start/
    gbserver_stop confirmation meant for someone else). session_id itself
    stays an opaque client-generated UUID; this only ensures two different
    authenticated identities can never collide on the same backend session
    key, even if they somehow end up holding the same raw id."""
    return f"{_resolve_identity(request)}:{session_id}"


class ChatRequest(BaseModel):
    session_id: str
    message: str
    # The frontend's current route when the message was sent — passive
    # browser-awareness context (see tool_loop_backend.py's
    # _build_augmented_message()), never treated as part of the user's own
    # words. Both optional: older/other frontends simply omit them.
    page_pathname: str | None = Field(default=None, max_length=_MAX_PAGE_FIELD_LEN)
    page_search: str | None = Field(default=None, max_length=_MAX_PAGE_FIELD_LEN)


class ChatStopRequest(BaseModel):
    session_id: str


class ChatConfirmRequest(BaseModel):
    session_id: str
    confirmation_id: str
    approved: bool


class ChatConfirmResponse(BaseModel):
    found: bool
    approved: bool | None = None
    result: str | None = None
    is_error: bool | None = None


class ChatStatusResponse(BaseModel):
    enabled: bool
    # Which harness/provider/model is actually running — shown in the chat
    # window's startup text. Absent (None) if disabled, or if the backend
    # can't actually be constructed yet (e.g. credentials are set but the
    # matching package extra isn't installed) — chat_stream's own error
    # surfaces the real problem when a message is sent; /status just omits
    # this rather than 500ing on a check that's supposed to be cheap.
    backend: str | None = None
    provider: str | None = None
    model: str | None = None


class ChatStopResponse(BaseModel):
    interrupted: bool


@router.get("/status", response_model=ChatStatusResponse)
async def chat_status() -> ChatStatusResponse:
    config = get_config()
    if not config.chat_enabled:
        return ChatStatusResponse(enabled=False)

    try:
        info = get_backend().describe()
    except Exception:  # noqa: BLE001 - see ChatStatusResponse's docstring
        logger.exception("Chat is enabled but the backend couldn't be constructed")
        return ChatStatusResponse(enabled=True)

    return ChatStatusResponse(enabled=True, **info)


async def _sse_encode(events: AsyncIterator[NormalizedEvent]) -> AsyncIterator[str]:
    async for event in events:
        yield f"data: {json.dumps(event)}\n\n"


@router.post("/stream")
async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
    config = get_config()
    if not config.chat_enabled:
        raise HTTPException(status_code=503, detail="Chat assistant is not configured")

    backend = get_backend()
    session_id = _scoped_session_id(request, body.session_id)
    return StreamingResponse(
        _sse_encode(
            backend.stream_turn(
                session_id, body.message, body.page_pathname, body.page_search
            )
        ),
        media_type="text/event-stream",
    )


@router.post("/stop", response_model=ChatStopResponse)
async def chat_stop(body: ChatStopRequest, request: Request) -> ChatStopResponse:
    config = get_config()
    if not config.chat_enabled:
        raise HTTPException(status_code=503, detail="Chat assistant is not configured")

    backend = get_backend()
    interrupted = await backend.interrupt_session(
        _scoped_session_id(request, body.session_id)
    )
    return ChatStopResponse(interrupted=interrupted)


@router.post("/confirm", response_model=ChatConfirmResponse)
async def chat_confirm(
    body: ChatConfirmRequest, request: Request
) -> ChatConfirmResponse:
    """Resolves a pending confirm_action proposal (see base.py's
    NormalizedEvent.confirmation_id) — approved executes the real gbmcp
    action outside the model loop; declined discards it. found=False is a
    normal outcome (already resolved, or the session was evicted), not an
    error — the frontend just stops showing the card as pending either way.

    session_id is scoped by the caller's trusted identity (see
    _scoped_session_id) before it reaches the backend — otherwise anyone who
    learned another user's raw session_id could approve/decline a
    confirmation meant for them.
    """
    config = get_config()
    if not config.chat_enabled:
        raise HTTPException(status_code=503, detail="Chat assistant is not configured")

    backend = get_backend()
    session_id = _scoped_session_id(request, body.session_id)
    result = await backend.confirm_action(
        session_id, body.confirmation_id, body.approved
    )
    return ChatConfirmResponse(**result)
