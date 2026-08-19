"""Chat assistant API — streams ChatAgentBackend responses as SSE.

Mirrors plans.py's gating pattern (503 when unconfigured). The route only
ever talks to the ChatAgentBackend interface via get_backend() — it never
imports a concrete backend module directly.
"""

from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from gb_ui_backend.config import get_config
from gb_ui_backend.services.chat_agents import get_backend
from gb_ui_backend.services.chat_agents.base import NormalizedEvent
from gb_ui_backend.services.request_identity import resolve_identity

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat")

# Defensive upper bound on browser-supplied route info — these are always
# short (a dashboard pathname/query string), so a much longer value signals
# something wrong rather than a legitimate route worth accepting.
_MAX_PAGE_FIELD_LEN = 512

# Defensive upper bound on the message itself — unlike page_pathname/
# page_search above, this one actually reaches the model and is retained in
# session history across every subsequent turn, so an unbounded value would
# both balloon that history and repeat a multi-MB payload to the model on
# every later turn in the same session.
_MAX_MESSAGE_LEN = 8_000

# Per-identity sliding-window rate limit on session creation/streaming — each
# call can spawn a new gbmcp subprocess (see ToolLoopBackend._sessions), so
# this bounds subprocess-spawn rate the same way ai.py's
# _rate_limit_analyze_logs bounds billable-LLM-call rate for that endpoint.
_CHAT_STREAM_RATE_LIMIT_WINDOW_SECONDS = 60
_CHAT_STREAM_RATE_LIMIT_MAX_CALLS = 30
_chat_stream_call_times: dict[str, list[float]] = {}


def _resolve_identity(request: Request) -> str:
    """See resolve_identity() — kept as a thin wrapper here since
    _scoped_session_id/_rate_limit_chat_stream (and this module's tests)
    already reference it by this name."""
    return resolve_identity(request)


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


def _rate_limit_chat_stream(request: Request) -> None:
    """Minimal per-identity sliding-window rate limit for chat_stream — see
    ai.py's _rate_limit_analyze_logs, which this mirrors."""
    identity = _resolve_identity(request)
    now = time.monotonic()
    recent = [
        t
        for t in _chat_stream_call_times.get(identity, [])
        if now - t < _CHAT_STREAM_RATE_LIMIT_WINDOW_SECONDS
    ]
    if len(recent) >= _CHAT_STREAM_RATE_LIMIT_MAX_CALLS:
        raise HTTPException(429, "Rate limit exceeded — try again later")
    recent.append(now)
    _chat_stream_call_times[identity] = recent


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(max_length=_MAX_MESSAGE_LEN)
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
    # True only if config says chat is configured AND the backend actually
    # constructed successfully — e.g. an API key env var is set but the
    # matching package extra isn't installed still reports False here,
    # rather than telling the frontend it's safe to render the widget and
    # then 500ing on the first real /chat/stream call.
    enabled: bool
    # Which harness/provider/model is actually running — shown in the chat
    # window's startup text. Always present when enabled is True, always
    # absent otherwise.
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
    except Exception:  # noqa: BLE001 - reported as disabled, not raised — see below
        # Config says chat is configured, but the backend itself couldn't be
        # built (e.g. ANTHROPIC_API_KEY is set but `anthropic` isn't
        # installed, or no chat model is configured). Reporting enabled=True
        # here would let the frontend render the widget and then have every
        # /chat/stream call 500 — report the real, checked-just-now outcome
        # instead so the frontend simply doesn't show the widget.
        logger.exception("Chat is enabled but the backend couldn't be constructed")
        return ChatStatusResponse(enabled=False)

    return ChatStatusResponse(enabled=True, **info)


async def _sse_encode(events: AsyncIterator[NormalizedEvent]) -> AsyncIterator[str]:
    async for event in events:
        yield f"data: {json.dumps(event)}\n\n"


@router.post("/stream")
async def chat_stream(
    body: ChatRequest,
    request: Request,
    _rate_limit: None = Depends(_rate_limit_chat_stream),
) -> StreamingResponse:
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
