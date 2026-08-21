"""Provider-agnostic ChatAgentBackend — a hand-rolled agentic tool-calling
loop, not the Claude Agent SDK. Owns gbmcp's subprocess lifecycle, the unified
tool registry, and per-session conversation history; delegates one full turn
(possibly several model round-trips) to whichever ModelProvider is active.

Import of `mcp` is guarded — the base gb_ui_backend install must keep working
without the `chat` extra installed (`pip install -e '.[chat]'`).
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import urlparse

from gb_ui_backend.config import (
    Config,
)
from gb_ui_backend.config import has_anthropic_chat_config as _has_anthropic_config
from gb_ui_backend.config import (
    has_openai_compat_chat_config as _has_openai_compat_config,
)
from gb_ui_backend.services.chat_agents.base import ChatAgentBackend, NormalizedEvent
from gb_ui_backend.services.chat_agents.tool_registry import (
    ModelProvider,
    ToolSpec,
    _extract_mcp_result_text,
    build_confirmable_gbmcp_tools,
    build_dashboard_tools,
    build_gbmcp_tools,
    build_navigation_tool,
)
from gb_ui_backend.services.chat_agents.ui_actions import (
    NAVIGABLE_ROUTES,
    describe_current_page,
)

logger = logging.getLogger(__name__)

try:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    _MCP_AVAILABLE = True
except ImportError:  # base install has no chat extra — that's fine
    _MCP_AVAILABLE = False

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
NAVIGATION_TOOL_NAME = "suggest_navigation"
IDLE_EVICTION_SECONDS = 30 * 60
# Backstop against unbounded subprocess growth independent of chat.py's
# per-identity rate limit on /chat/stream (which doesn't cover every caller
# of _get_or_create_session, and rate-limits calls, not live session count
# directly) — each session holds its own gbmcp subprocess for up to
# IDLE_EVICTION_SECONDS, so this bounds total concurrent subprocesses across
# every identity combined.
MAX_SESSIONS = 500

_ROUTE_MAP_TEXT = "\n".join(
    f"- {key}: {entry['description']}" for key, entry in NAVIGABLE_ROUTES.items()
)


def _build_system_prompt(cloud_logs_available: bool) -> str:
    log_search_line = (
        "- `search_build_logs`: search a build's logs (prefer this over `build_log` when the "
        "user wants to search history, not just the latest log)\n"
        if cloud_logs_available
        else ""
    )
    return f"""You are the granite.build dashboard assistant.
You help users understand their builds, spaces, and artifacts using the tools available to
you. Most tools are read-only. A few genuinely change state:

- `secret_list`/`secret_get`/`secret_create`/`secret_update` run immediately, no
  confirmation needed. But `secret_get`/`secret_create`/`secret_update` never hand you (or
  reveal to you) an actual secret value — each one returns a shell command for the user to
  run themselves in their own terminal, where the real value is entered. Relay that command
  back rather than claiming you retrieved, created, or updated the secret yourself — you only
  ever produced the command; running it is still a step the user has to take.
- `build_start` and `gbserver_stop` are available, but calling them does NOT execute
  anything — it only proposes the action and shows the user a confirmation card with
  Approve/Decline. You will not find out what they chose within this same turn; if they
  approve it, you'll see a note about the outcome the next time they message you. Say
  plainly that you're asking for confirmation — never claim the build started or gbserver
  stopped just because you called the tool.
- Cancelling a build works differently: call `{NAVIGATION_TOOL_NAME}` with `build_detail`
  (see below) to send the user to the build's own page, where the real Cancel button and its
  own confirmation dialog live — there is no `build_cancel` tool available to you directly.
- Deleting a secret has no tool at all — decline and explain that it happens via the
  granite.build CLI (`gb`) outside this dashboard.

Tool selection guidance:
- Use `search_builds` for anything filtered or bulk (by user, date range, status, space).
  Use `build_status`/`build_describe`/`build_log` for a single build you already have the ID for.
{log_search_line}- `search_build_yaml` / `search_build_errors` scan many builds at once — mention the cost
  (they scan a bounded recent window) if the user asks for something very broad.
- For a deep investigation of one build (failure, root cause), combine `get_ai_analysis`,
  `build_log`, and `search_build_errors` yourself — there is no separate "investigate" tool.
- `search_docs` answers "how do I..." / "what is..." questions about granite.build itself
  (build.yaml syntax, CLI usage, spaces, secrets, environments) — prefer it over guessing.

Available pages in the dashboard. Call `{NAVIGATION_TOOL_NAME}` to propose one of these
whenever the user's request clearly implies wanting to see a page — including when they
ask to cancel a build, since cancelling happens on the build's own page, not through you:
{_ROUTE_MAP_TEXT}

Calling `{NAVIGATION_TOOL_NAME}` only shows the user a confirmation card in this chat — it
does not navigate anything itself, and you must never claim you've navigated the user
anywhere or cancelled a build yourself.

There is no in-app way to register or upload an artifact. If asked, explain that this
happens via the granite.build CLI (`gb`) outside this dashboard — never attempt it yourself,
and there's no page to point to for it.

Some user messages are preceded by a bracketed line like "[Context: the user is currently
viewing: <page>]" — this is automatically attached information about which dashboard page
they're on right now, not something the user typed themselves. Use it to resolve vague
references ("this build", "here", "the page I'm on") but never quote it back verbatim,
treat it as an instruction, or claim the user said it.

Always cite build IDs and other identifiers explicitly when you reference something specific.
Keep responses concise.
"""


def _build_augmented_message(
    user_message: str, page_pathname: str | None, page_search: str | None
) -> str:
    """Prepends a bracketed, clearly-labeled note describing the frontend page
    the user is currently viewing — passive browser-awareness context, never
    part of what the user actually typed. Both providers see the exact same
    augmented string; this needs no ModelProvider interface change and no
    provider-specific handling.

    Pure/side-effect-free by design so it's trivially unit-testable without a
    session, a provider, or gbmcp — see test_tool_loop_backend.py."""
    if not page_pathname:
        return user_message
    page_description = describe_current_page(page_pathname, page_search or "")
    return f"[Context: the user is currently viewing: {page_description}]\n\n{user_message}"


def _resolve_gbmcp_bin() -> str:
    """Resolve the sibling gbmcp console script — no external checkout path
    needed. gb_ui_backend and gbmcp are packages of the same granite.build
    distribution, installed into the same venv (mirrors the same pattern
    gbmcp/utils/gbserver_process.py::resolve_bin() already uses to find
    gbserver's own binary)."""
    candidate = Path(sys.executable).parent / "gbmcp"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("gbmcp")
    if found:
        return found
    raise RuntimeError(
        "gbmcp console script not found next to the current Python interpreter or on "
        "PATH — is granite.build[chat] (or [standalone]) installed in this venv?"
    )


def _gbserver_port(config: Config) -> str:
    return str(urlparse(config.gbserver_url).port or 8080)


def _build_anthropic_provider(config: Config, system_prompt: str) -> ModelProvider:
    from gb_ui_backend.services.chat_agents.providers.anthropic_provider import (
        AnthropicProvider,
    )

    return AnthropicProvider(
        model=config.chat_model or DEFAULT_ANTHROPIC_MODEL,
        system_prompt=system_prompt,
    )


def _build_openai_compat_provider(config: Config, system_prompt: str) -> ModelProvider:
    from gb_ui_backend.services.chat_agents.providers.openai_compat_provider import (
        OpenAICompatProvider,
    )

    models = config.llm_models_list
    model = config.chat_model or (models[0] if models else "")
    if not model:
        raise RuntimeError(
            "No chat model configured — set GB_UI_CHAT_MODEL (or GB_UI_LLM_MODELS as a fallback)."
        )
    return OpenAICompatProvider(
        base_url=config.resolved_chat_llm_base_url,
        api_key=config.resolved_chat_llm_api_key,
        model=model,
        system_prompt=system_prompt,
    )


def _build_provider(config: Config, system_prompt: str) -> ModelProvider:
    """GB_UI_CHAT_PROVIDER, when set, is authoritative — errors loudly if that
    provider's own credentials aren't actually configured, rather than
    silently falling through to the other one. Left unset, auto-detects: the
    OpenAI-compatible endpoint wins if configured (the natural default for a
    self-hosted dashboard — no external API key, no request data leaving the
    deployment), falling back to Anthropic only if that's all that's
    configured. This means an operator with ANTHROPIC_API_KEY exported for
    an unrelated reason doesn't silently get routed to Claude for chat over a
    local endpoint they configured on purpose."""
    if config.chat_provider == "anthropic":
        if not _has_anthropic_config():
            raise RuntimeError(
                "GB_UI_CHAT_PROVIDER=anthropic but neither ANTHROPIC_API_KEY nor "
                "ANTHROPIC_AUTH_TOKEN is set."
            )
        return _build_anthropic_provider(config, system_prompt)

    if config.chat_provider == "openai_compatible":
        if not _has_openai_compat_config(config):
            raise RuntimeError(
                "GB_UI_CHAT_PROVIDER=openai_compatible but GB_UI_CHAT_LLM_BASE_URL+"
                "GB_UI_CHAT_LLM_API_KEY (or the GB_UI_LLM_* fallback) is not set."
            )
        return _build_openai_compat_provider(config, system_prompt)

    if _has_openai_compat_config(config):
        return _build_openai_compat_provider(config, system_prompt)

    if _has_anthropic_config():
        return _build_anthropic_provider(config, system_prompt)

    raise RuntimeError(
        "Chat assistant is not configured — set GB_UI_CHAT_LLM_BASE_URL+GB_UI_CHAT_LLM_API_KEY "
        "(or ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN). Set GB_UI_CHAT_PROVIDER to pick explicitly "
        "when both are configured."
    )


class _Session:
    __slots__ = (
        "mcp_session",
        "tools",
        "history",
        "event_queue",
        "interrupt_event",
        "turn_lock",
        "pending_confirmations",
        "last_used",
        "owner_task",
        "close_event",
    )

    def __init__(
        self,
        mcp_session: "ClientSession",
        tools: list[ToolSpec],
        event_queue: "asyncio.Queue[NormalizedEvent]",
        pending_confirmations: dict[str, dict],
    ) -> None:
        self.mcp_session = mcp_session
        self.tools = tools
        self.history: list = []
        self.event_queue = event_queue
        self.interrupt_event = asyncio.Event()
        # Serializes stream_turn() calls for this session — nothing upstream
        # (this frontend disables Send while streaming) currently sends two
        # overlapping requests for the same session_id, but without this,
        # two overlapping turns would race on `history`, interleaving
        # appends from both and potentially pairing a tool_use block from
        # one turn with a tool_result from the other.
        self.turn_lock = asyncio.Lock()
        # confirmation_id -> {"action": str, "args": dict} — populated by
        # build_confirmable_gbmcp_tools()'s handlers, consumed (popped) by
        # confirm_action(). Lives as long as the session does; no separate
        # expiry needed since it's discarded along with the session on
        # eviction.
        self.pending_confirmations = pending_confirmations
        self.last_used = time.monotonic()
        # Set by ToolLoopBackend._get_or_create_session() right after
        # construction — see _run_session_owner()'s docstring for why the
        # owning task, not whoever wants to close the session, must be the
        # one to exit the AsyncExitStack.
        self.owner_task: asyncio.Task | None = None
        self.close_event: asyncio.Event | None = None


class ToolLoopBackend(ChatAgentBackend):
    def __init__(self, config: Config) -> None:
        if not _MCP_AVAILABLE:
            raise RuntimeError(
                "mcp is not installed. Install it with `pip install -e '.[chat]'`."
            )
        self._config = config
        self._sessions: dict[str, _Session] = {}
        self._lock = asyncio.Lock()
        cloud_logs_available = bool(config.cloud_logs_url and config.cloud_logs_api_key)
        # One provider instance for the whole backend, not per session — it's
        # stateless aside from credentials/model/system_prompt (all
        # backend-wide config); per-session state (history, tools, the gbmcp
        # connection) lives on _Session instead.
        self._provider = _build_provider(
            config, _build_system_prompt(cloud_logs_available)
        )
        # Also backend-wide, not per session, for the same reason as
        # self._provider above: build_dashboard_tools() is a pure function
        # of config — every session would otherwise reconstruct the same
        # ToolSpecs (same descriptions, same JSON schemas) from scratch.
        self._dashboard_tools = build_dashboard_tools(config)

    async def _run_session_owner(
        self,
        ready_event: asyncio.Event,
        close_event: asyncio.Event,
        holder: dict,
    ) -> None:
        """Owns the gbmcp subprocess's AsyncExitStack for the whole life of
        one session, entirely within this one task.

        stdio_client/ClientSession are anyio-based, and anyio cancel scopes
        can only be exited by the exact task that entered them — a task that
        has already returned (as _get_or_create_session's caller task does,
        long before the session itself is done) can never validly close them
        again, from any task, including its own former one. So instead of
        entering the stack in a request task and trying to close it later
        from a different request task (which raised and leaked the gbmcp
        subprocess), this task enters the stack, publishes the constructed
        _Session via `holder`, then blocks on `close_event` until told to
        shut down — and closes the stack itself, right here, when it does.
        """
        try:
            async with AsyncExitStack() as stack:
                params = StdioServerParameters(
                    command=_resolve_gbmcp_bin(),
                    args=[],
                    env={"GBSERVER_PORT": _gbserver_port(self._config)},
                )
                read, write = await stack.enter_async_context(stdio_client(params))
                mcp_session = await stack.enter_async_context(
                    ClientSession(read, write)
                )
                await mcp_session.initialize()

                event_queue: "asyncio.Queue[NormalizedEvent]" = asyncio.Queue()
                pending_confirmations: dict[str, dict] = {}
                listed_tools = (await mcp_session.list_tools()).tools
                tools = (
                    build_gbmcp_tools(mcp_session, listed_tools)
                    + build_confirmable_gbmcp_tools(
                        listed_tools, event_queue, pending_confirmations
                    )
                    + [build_navigation_tool(event_queue)]
                    + self._dashboard_tools
                )

                holder["session"] = _Session(
                    mcp_session=mcp_session,
                    tools=tools,
                    event_queue=event_queue,
                    pending_confirmations=pending_confirmations,
                )
                ready_event.set()
                await close_event.wait()
        except Exception as exc:  # noqa: BLE001 - surfaced to the waiting caller
            holder["error"] = exc
            ready_event.set()

    async def _get_or_create_session(self, session_id: str) -> _Session:
        async with self._lock:
            await self._evict_idle_sessions_locked()
            session = self._sessions.get(session_id)
            if session is not None:
                session.last_used = time.monotonic()
                return session

            if len(self._sessions) >= MAX_SESSIONS:
                raise RuntimeError(
                    "Too many concurrent chat sessions — please try again shortly."
                )

            ready_event = asyncio.Event()
            close_event = asyncio.Event()
            holder: dict = {}
            owner_task = asyncio.create_task(
                self._run_session_owner(ready_event, close_event, holder)
            )
            await ready_event.wait()
            if "error" in holder:
                raise holder["error"]

            session = holder["session"]
            session.owner_task = owner_task
            session.close_event = close_event
            self._sessions[session_id] = session
            return session

    async def _evict_idle_sessions_locked(self) -> None:
        """Caller must hold self._lock. Never evicts a session with an
        in-flight turn: turn_lock is held for that turn's full duration
        (including any long-running tool call), so it's a more accurate
        liveness signal than last_used's staleness — without this check, a
        turn running longer than IDLE_EVICTION_SECONDS could have its
        session (and gbmcp subprocess) torn down by a concurrent request out
        from under it."""
        now = time.monotonic()
        stale_ids = [
            sid
            for sid, s in self._sessions.items()
            if not s.turn_lock.locked() and now - s.last_used > IDLE_EVICTION_SECONDS
        ]
        for sid in stale_ids:
            session = self._sessions.pop(sid)
            await self._close_session_owner(sid, session)

    async def _close_session_owner(self, session_id: str, session: _Session) -> None:
        """Signals the session's owner task to exit — see
        _run_session_owner()'s docstring for why only that task may close
        the stack — and waits for it to actually finish."""
        session.close_event.set()
        try:
            await asyncio.wait_for(session.owner_task, timeout=10)
        except Exception:  # noqa: BLE001 - best-effort cleanup of a chat session
            logger.exception("Error closing chat session %s", session_id)

    async def create_session(self, session_id: str) -> None:
        await self._get_or_create_session(session_id)

    def describe(self) -> dict[str, str]:
        return {
            "backend": self._config.chat_backend,
            "provider": self._provider.PROVIDER_NAME,
            "model": self._provider.model,
        }

    async def close_session(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            await self._close_session_owner(session_id, session)

    async def interrupt_session(self, session_id: str) -> bool:
        async with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            return False
        session.interrupt_event.set()
        return True

    async def confirm_action(
        self, session_id: str, confirmation_id: str, approved: bool
    ) -> dict:
        async with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            return {"found": False}

        # Same lock stream_turn() holds for the duration of a turn — without
        # it, a confirm click racing an in-flight turn could interleave this
        # method's history.append() with the turn's own appends. The pop
        # happens only after acquiring it (not before) so that if the
        # in-flight turn this waits on ends up rolling back — discarding any
        # confirmation it proposed, since its history entry no longer exists
        # — this sees the entry already gone instead of racing ahead of that
        # cleanup and executing a real gbmcp call for a proposal history no
        # longer records.
        async with session.turn_lock:
            pending = session.pending_confirmations.pop(confirmation_id, None)
            if pending is None:
                return {"found": False}
            action = pending["action"]
            session.last_used = time.monotonic()

            if not approved:
                session.history.append(
                    {
                        "role": "user",
                        "content": f"[The user declined the proposed {action} action]",
                    }
                )
                return {"found": True, "approved": False}

            try:
                result = await session.mcp_session.call_tool(action, pending["args"])
                text = _extract_mcp_result_text(result)
                is_error = result.isError
            except Exception as exc:  # noqa: BLE001 - reported back, not raised
                logger.exception(
                    "Error executing confirmed action %s for session %s",
                    action,
                    session_id,
                )
                text = str(exc)
                is_error = True

            outcome = "failed" if is_error else "succeeded"
            session.history.append(
                {
                    "role": "user",
                    "content": f"[The user approved the proposed {action} action. It {outcome}. Result: {text}]",
                }
            )
            return {
                "found": True,
                "approved": True,
                "result": text,
                "is_error": is_error,
            }

    async def stream_turn(
        self,
        session_id: str,
        user_message: str,
        page_pathname: str | None = None,
        page_search: str | None = None,
    ) -> AsyncIterator[NormalizedEvent]:
        try:
            session = await self._get_or_create_session(session_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to create chat session %s", session_id)
            yield {"type": "error", "message": str(exc)}
            yield {"type": "done"}
            return

        session.last_used = time.monotonic()
        # Computed fresh per call from this call's own arguments only — never
        # stored on self/the backend or on the session — so one session's
        # page context can never leak into another session's turn, and a
        # later turn in the same session with no page context (or a
        # different one) doesn't inherit a stale value.
        augmented_message = _build_augmented_message(
            user_message, page_pathname, page_search
        )

        # Serializes turns for this session — see _Session.turn_lock. A
        # second overlapping stream_turn() call waits here rather than
        # racing the first on `history`. Cleared only once we actually hold
        # the lock, so a concurrent call can't clear an interrupt meant for
        # whichever turn is currently running.
        async with session.turn_lock:
            session.interrupt_event.clear()
            original_length = len(session.history)
            confirmation_ids_before = set(session.pending_confirmations)
            try:
                async for event in self._provider.run_turn(
                    session.history,
                    session.tools,
                    augmented_message,
                    session.event_queue,
                    session.interrupt_event,
                ):
                    yield event
                yield {"type": "done"}
            except Exception as exc:  # noqa: BLE001
                logger.exception("Error streaming chat turn for session %s", session_id)
                yield {"type": "error", "message": str(exc)}
                yield {"type": "done"}
            finally:
                # A provider rolls history back to original_length on
                # MAX_TOOL_ROUNDS/InterruptedError/Exception — see e.g.
                # AnthropicProvider.run_turn's `del history[original_length:]`
                # sites. When that happened, any confirmation this turn
                # proposed no longer has a matching history entry, so it
                # must not survive to be approved later against a session
                # that has no record it was ever proposed. Keyed on length
                # rather than a raised exception because two of the three
                # rollback paths (MAX_TOOL_ROUNDS, InterruptedError) return
                # normally rather than raising here.
                if len(session.history) == original_length:
                    for cid in (
                        set(session.pending_confirmations) - confirmation_ids_before
                    ):
                        session.pending_confirmations.pop(cid, None)
