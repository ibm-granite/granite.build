"""Framework-agnostic interface for chat agent backends.

api/chat.py and the frontend only ever see NormalizedEvent dicts — never a
provider's native message/event type (e.g. an Anthropic ContentBlock or an
OpenAI-compatible tool_call). This is what lets tool_loop_backend.py support
multiple model providers (and, in principle, a wholesale different backend
framework later) without touching the route, the SSE wire format, or
ChatWidget.tsx.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Literal, TypedDict

NormalizedEventType = Literal[
    "text_delta",
    "tool_call",
    "ui_action",
    "confirm_action",
    "done",
    "error",
]


class NormalizedEvent(TypedDict, total=False):
    type: NormalizedEventType
    text: str
    tool_name: str
    tool_input: dict[str, Any]
    route: str
    label: str
    message: str
    # confirm_action only — see gbmcp_policy.py's CONFIRMABLE_GBMCP_TOOLS and
    # tool_registry.py's build_confirmable_gbmcp_tools(). Identifies which
    # pending proposal a later POST /chat/confirm call resolves.
    confirmation_id: str


class ChatAgentBackend(ABC):
    """One implementation per agent framework. Only ToolLoopBackend
    (services/chat_agents/tool_loop_backend.py) exists today — a hand-rolled
    tool-calling loop that itself supports multiple model providers
    (Anthropic directly, or any OpenAI-compatible API) via ModelProvider
    (services/chat_agents/tool_registry.py)."""

    @abstractmethod
    async def create_session(self, session_id: str) -> None:
        """Ensure a session exists for session_id, creating it if needed."""

    @abstractmethod
    def stream_turn(
        self,
        session_id: str,
        user_message: str,
        page_pathname: str | None = None,
        page_search: str | None = None,
    ) -> AsyncIterator[NormalizedEvent]:
        """Send user_message into the session and stream back NormalizedEvents.

        page_pathname/page_search are the frontend's current route (e.g.
        "/dashboard/builds/_" + "?id=abc123") at the moment the message was
        sent — passive browser-awareness context, not part of the user's own
        words. Both optional and provider-agnostic: a backend that doesn't
        use them is free to ignore both."""

    @abstractmethod
    async def close_session(self, session_id: str) -> None:
        """Tear down a session's underlying resources (e.g. a subprocess)."""

    @abstractmethod
    async def interrupt_session(self, session_id: str) -> bool:
        """Abort whatever the session is currently doing (a running stream_turn
        call). Returns False if session_id has no active session — the route
        treats that as a no-op, not an error, since it's a benign race with
        the turn finishing on its own."""

    @abstractmethod
    def describe(self) -> dict[str, str]:
        """Static info about what's actually running — surfaced by
        GET /api/analytics/chat/status so the frontend can show it in the
        chat window's startup text. Keys are backend-defined; ToolLoopBackend
        returns {"backend": ..., "provider": ..., "model": ...}. A future
        backend that's a wholesale different framework is free to return a
        different key set — the frontend only shows what's present."""

    @abstractmethod
    async def confirm_action(
        self, session_id: str, confirmation_id: str, approved: bool
    ) -> dict[str, Any]:
        """Resolve a pending confirm_action proposal (see NormalizedEvent's
        confirmation_id) — approved executes the real action outside the
        model loop entirely; declined discards it. Either way, a note about
        the outcome is appended to the session's own history so the model
        finds out on its next turn. Returns {"found": False} if
        confirmation_id doesn't match anything pending for this session
        (already resolved, expired with the session, or never existed) —
        the route treats that as a normal, reportable outcome, not an
        error."""
