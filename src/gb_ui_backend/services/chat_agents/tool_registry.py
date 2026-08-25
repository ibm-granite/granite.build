"""Shared types and tool assembly for tool_loop_backend.py's two model providers.

ToolSpec/ModelProvider are the seam between the provider-agnostic backend and
each provider's own native wire format — a provider only ever sees a
list[ToolSpec], never gbmcp, dashboard_tools.py, or ui_actions.py directly.

build_gbmcp_tools() is the actual security boundary for gbmcp: it filters
against ALLOWED_GBMCP_TOOLS before ever constructing a ToolSpec, so a
disallowed tool is never described to the model in the first place, not just
blocked after the fact.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from mcp import ClientSession

from gb_ui_backend.config import Config
from gb_ui_backend.services.chat_agents import dashboard_tools
from gb_ui_backend.services.chat_agents.base import NormalizedEvent
from gb_ui_backend.services.chat_agents.gbmcp_policy import (
    ALLOWED_GBMCP_TOOLS,
    CONFIRMABLE_GBMCP_TOOLS,
)
from gb_ui_backend.services.chat_agents.ui_actions import (
    NAVIGABLE_ROUTES,
    build_navigation_route,
)

logger = logging.getLogger(__name__)

# Shared by every ModelProvider (anthropic_provider.py, openai_compat_
# provider.py) — previously each declared its own identical copy of both,
# risking one getting tuned and the other silently left behind.
# MAX_TOOL_ROUNDS: safety net against a runaway tool-calling loop (e.g. the
# model repeatedly mis-calling a tool) — well above any legitimate
# multi-step investigation.
MAX_TOOL_ROUNDS = 12
MAX_TOKENS = 4096

ToolHandler = Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema
    handler: (
        ToolHandler  # raises on failure; callers turn that into a tool-error result
    )


class ModelProvider(Protocol):
    """One implementation per model API. Each owns its own native message
    history shape and its own inner tool-calling loop (possibly several
    round-trips against the model per call) — see AnthropicProvider /
    OpenAICompatProvider. history is mutated in place across calls so a
    session's multi-turn context survives between stream_turn() calls.

    model/PROVIDER_NAME are read by ToolLoopBackend.describe() to answer
    GET /api/analytics/chat/status — the frontend's startup text shows both,
    so a user glancing at the chat window can tell which model/harness is
    actually running."""

    model: str
    PROVIDER_NAME: str

    def run_turn(
        self,
        history: list[Any],
        tools: list[ToolSpec],
        user_message: str,
        event_queue: "asyncio.Queue[NormalizedEvent]",
        interrupt_event: asyncio.Event,
    ) -> (
        Any
    ):  # AsyncIterator[NormalizedEvent] — Protocol can't spell this precisely for a generator method
        ...


async def race_interrupt(coro: Awaitable[Any], interrupt_event: asyncio.Event) -> Any:
    """Await coro, but abandon it (raising InterruptedError) as soon as
    interrupt_event is set — the mechanism behind POST /chat/stop actually
    cutting off an in-flight model call rather than only stopping the loop
    from starting another round."""
    task: asyncio.Task = asyncio.ensure_future(coro)
    interrupt_task = asyncio.ensure_future(interrupt_event.wait())
    try:
        done, _pending = await asyncio.wait(
            {task, interrupt_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if task in done:
            return task.result()
        raise InterruptedError("Chat turn interrupted")
    finally:
        for t in (task, interrupt_task):
            if not t.done():
                t.cancel()


def drain_and_discard_events(event_queue: "asyncio.Queue[NormalizedEvent]") -> None:
    """Called by a provider's run_turn when rolling history back — a
    confirmable/navigation tool's handler may have already put a
    confirm_action/ui_action event on the queue for a turn that's now being
    erased, and this must not survive to be drained (and yielded, out of
    context) by a later, unrelated turn. Discards rather than yields: a
    generator can't safely yield again from an exception handler that's
    mid-GeneratorExit, and the event describes a proposal history no longer
    records anyway, so there's nothing left to show the user for it."""
    while not event_queue.empty():
        event_queue.get_nowait()


async def run_tool_loop(
    history: list[Any],
    event_queue: "asyncio.Queue[NormalizedEvent]",
    original_length: int,
    run_one_round: Callable[[dict[str, Any]], Any],
    provider_label: str,
) -> Any:  # AsyncIterator[NormalizedEvent]
    """Drives the round-by-round loop, rollback, and cleanup shared by every
    ModelProvider — previously ~40-50 near-identical lines duplicated
    between anthropic_provider.py and openai_compat_provider.py, with only
    wire-format conversion actually differing between them. That
    provider-specific part (extracting text/tool-calls from one model
    response, dispatching tool calls, appending the result back onto
    history in that provider's own shape) is `run_one_round` — an async
    generator function taking a mutable `outcome` dict, yielding
    NormalizedEvents for this round, and before finishing setting
    outcome["status"] to:
      - "continue": more tool-calling rounds needed
      - "done": a natural end to the turn (e.g. no further tool calls)
      - "empty": the model returned nothing usable at all — rolls back and
        yields outcome.get("message", ...) as an error, same as exhausting
        MAX_TOOL_ROUNDS below

    `original_length` must be the history length from *before* this turn's
    user message was appended, snapshotted synchronously by the caller —
    not computed in here. This function is an async generator, so nothing
    in its body runs until the caller's first `async for`/`asend()`; by
    then, run_turn()'s own synchronous prefix (appending the user message,
    and for OpenAI-compat, possibly a bootstrap system message first) has
    already run. Snapshotting `len(history)` in here would capture a
    length that already includes this turn's own user message, and rolling
    back to it on error/interrupt would leave that message behind instead
    of truly restoring pre-turn state.
    """
    try:
        for _ in range(MAX_TOOL_ROUNDS):
            outcome: dict[str, Any] = {"status": "continue"}
            async for event in run_one_round(outcome):
                yield event

            if outcome["status"] == "done":
                return
            if outcome["status"] == "empty":
                del history[original_length:]
                drain_and_discard_events(event_queue)
                yield {
                    "type": "error",
                    "message": outcome.get(
                        "message",
                        "Model returned an empty response with no content or tool calls.",
                    ),
                }
                return

        logger.warning(
            "%s tool-calling loop hit MAX_TOOL_ROUNDS=%d without finishing",
            provider_label,
            MAX_TOOL_ROUNDS,
        )
        # Roll back before yielding the error — otherwise history ends in a
        # dangling assistant tool-call entry with no matching result, which
        # every provider's API rejects on the session's next turn.
        del history[original_length:]
        drain_and_discard_events(event_queue)
        yield {
            "type": "error",
            "message": f"Stopped after {MAX_TOOL_ROUNDS} tool-calling rounds without a final answer.",
        }
    except InterruptedError:
        del history[original_length:]
        drain_and_discard_events(event_queue)
        return
    except BaseException:
        # BaseException, not Exception: a client disconnect cancels this
        # turn's task with asyncio.CancelledError, which is a BaseException
        # subclass, not an Exception subclass — narrower coverage here
        # would let cancellation skip the rollback, leaving a dangling
        # tool-call entry for the life of the session. Same corruption risk
        # applies to a network error, malformed response, etc. mid-turn.
        del history[original_length:]
        drain_and_discard_events(event_queue)
        raise


def _extract_mcp_result_text(result: Any) -> str:
    """Shared by the direct gbmcp handler and confirm_action()'s
    execute-on-approve path — both need the same text-content-blocks-joined
    shape gbmcp's CallToolResult comes back as."""
    return "\n".join(getattr(block, "text", "") for block in result.content).strip()


def _mcp_tool_handler(mcp_session: ClientSession, tool_name: str) -> ToolHandler:
    async def handler(args: dict[str, Any]) -> str:
        # Belt-and-suspenders: build_gbmcp_tools() below already never
        # constructs a ToolSpec for a disallowed tool, so this can't
        # actually be reached with a bad name today — cheap insurance
        # against a future refactor accidentally widening what's dispatched.
        if tool_name not in ALLOWED_GBMCP_TOOLS:
            raise RuntimeError(f"{tool_name} is not permitted for the chat assistant")
        result = await mcp_session.call_tool(tool_name, args)
        text = _extract_mcp_result_text(result)
        if result.isError:
            raise RuntimeError(text or f"{tool_name} failed")
        return text

    return handler


def build_gbmcp_tools(
    mcp_session: ClientSession, listed_tools: list[Any]
) -> list[ToolSpec]:
    """`listed_tools` is `(await mcp_session.list_tools()).tools` — fetched
    once by the caller and shared with build_confirmable_gbmcp_tools()
    rather than each independently calling list_tools() (a second real
    round trip to the gbmcp subprocess for the identical listing)."""
    return [
        ToolSpec(
            name=tool.name,
            description=tool.description or "",
            parameters=tool.inputSchema or {"type": "object", "properties": {}},
            handler=_mcp_tool_handler(mcp_session, tool.name),
        )
        for tool in listed_tools
        if tool.name in ALLOWED_GBMCP_TOOLS
    ]


def _confirmable_tool_handler(
    tool_name: str,
    event_queue: "asyncio.Queue[NormalizedEvent]",
    pending: dict[str, dict[str, Any]],
) -> ToolHandler:
    async def handler(args: dict[str, Any]) -> str:
        # Same belt-and-suspenders as _mcp_tool_handler — build_confirmable_
        # gbmcp_tools() below already never constructs a ToolSpec for
        # anything outside CONFIRMABLE_GBMCP_TOOLS.
        if tool_name not in CONFIRMABLE_GBMCP_TOOLS:
            raise RuntimeError(f"{tool_name} is not permitted for the chat assistant")
        confirmation_id = str(uuid.uuid4())
        pending[confirmation_id] = {"action": tool_name, "args": args}
        await event_queue.put(
            {
                "type": "confirm_action",
                "confirmation_id": confirmation_id,
                "tool_name": tool_name,
                "tool_input": args,
                "label": f"Proposed action: {tool_name}",
            }
        )
        # The model finds out what happened, if anything, on its next turn
        # (see ToolLoopBackend.confirm_action()'s history note) — same as it
        # never directly sees whether a suggest_navigation proposal was
        # clicked. This return value must not claim the action happened.
        return f"Proposed {tool_name} for user confirmation (id={confirmation_id}). Awaiting the user's decision."

    return handler


def build_confirmable_gbmcp_tools(
    listed_tools: list[Any],
    event_queue: "asyncio.Queue[NormalizedEvent]",
    pending: dict[str, dict[str, Any]],
) -> list[ToolSpec]:
    """Tools the model can call, but which only ever propose the action —
    see gbmcp_policy.py's module docstring and
    ToolLoopBackend.confirm_action() for the execute-on-approve half of
    this. `listed_tools` is the same `(await mcp_session.list_tools()).tools`
    passed to build_gbmcp_tools() — one real round trip to the gbmcp
    subprocess shared by both, not two independent ones for the identical
    listing — so the model still sees gbmcp's real schema for these (e.g.
    build_start's file_content/space/params/tags/description) with nothing
    hand-maintained to drift out of sync."""
    return [
        ToolSpec(
            name=tool.name,
            description=(
                f"{tool.description or tool.name} REQUIRES THE USER'S EXPLICIT APPROVAL — "
                "calling this only proposes the action and shows the user a confirmation "
                "card; it does not execute anything itself, and you will not know the "
                "outcome until the user's next message."
            ),
            parameters=tool.inputSchema or {"type": "object", "properties": {}},
            handler=_confirmable_tool_handler(tool.name, event_queue, pending),
        )
        for tool in listed_tools
        if tool.name in CONFIRMABLE_GBMCP_TOOLS
    ]


def build_navigation_tool(event_queue: "asyncio.Queue[NormalizedEvent]") -> ToolSpec:
    async def handler(args: dict[str, Any]) -> str:
        page = args["page"]
        reason = args["reason"]
        params = args.get("params") or {}
        # build_navigation_route raises UnknownPageError/MissingRouteParamsError
        # (both ValueError) on bad input — left uncaught here so the generic
        # tool-error handling in each provider's dispatch loop turns it into a
        # normal tool-error result the model can see and react to, using the
        # already-descriptive message build_navigation_route constructs.
        result = build_navigation_route(page, reason, **params)
        await event_queue.put(
            {"type": "ui_action", "route": result["route"], "label": result["label"]}
        )
        return f"Proposed navigating to {result['route']}: {reason}"

    return ToolSpec(
        name="suggest_navigation",
        description=(
            "Propose navigating the user to an existing page in the dashboard. This never "
            "navigates anything itself — it only shows the user a confirmation card they must "
            "explicitly click before their browser goes anywhere."
        ),
        parameters={
            "type": "object",
            "properties": {
                "page": {
                    "type": "string",
                    "enum": list(NAVIGABLE_ROUTES.keys()),
                    "description": "Which page to navigate to",
                },
                "reason": {
                    "type": "string",
                    "description": "Short, user-facing reason shown on the confirmation card",
                },
                "params": {
                    "type": "object",
                    "description": 'Dynamic route params, e.g. {"build_id": "abc123"}',
                },
            },
            "required": ["page", "reason"],
        },
        handler=handler,
    )


def _wrap(
    name: str, description: str, parameters: dict[str, Any], fn: Callable[..., Any]
) -> ToolSpec:
    async def handler(args: dict[str, Any]) -> Any:
        try:
            result = fn(**args)
            return await result if inspect.isawaitable(result) else result
        except dashboard_tools.DashboardToolError as exc:
            raise RuntimeError(str(exc)) from exc

    return ToolSpec(
        name=name, description=description, parameters=parameters, handler=handler
    )


def build_dashboard_tools(config: Config) -> list[ToolSpec]:
    """Ports the same tool descriptions/schemas used previously — already
    framework-agnostic, just re-declared as ToolSpecs instead of via a
    Claude-Agent-SDK-specific @tool decorator."""
    specs = [
        _wrap(
            "search_docs",
            "Look up a granite.build documentation page by topic — build.yaml syntax, CLI usage, "
            "spaces, secrets, environments, etc. Use this instead of guessing how granite.build works.",
            {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "enum": sorted(dashboard_tools.DOC_TOPICS.keys()),
                        "description": "Which doc topic to read",
                    },
                },
                "required": ["topic"],
            },
            dashboard_tools.search_docs,
        ),
        _wrap(
            "search_builds",
            "Search/filter builds by text, status, user, space, and/or date range. Use this for "
            "anything bulk or filtered; use build_status/build_describe for a single known build ID.",
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to match against build name",
                    },
                    "status": {
                        "type": "string",
                        "description": "e.g. running, success, failed",
                    },
                    "user": {"type": "string", "description": "Username to filter by"},
                    "space_name": {"type": "string"},
                    "days_back": {"type": "integer", "description": "Default 30"},
                    "date_from": {"type": "string", "description": "YYYY-MM-DD"},
                    "date_to": {"type": "string", "description": "YYYY-MM-DD"},
                    "limit": {"type": "integer", "description": "Default 50, max 200"},
                },
            },
            dashboard_tools.search_builds,
        ),
        _wrap(
            "search_build_yaml",
            "Regex search build.yaml content across recent builds. Scans a bounded recent window "
            "(fetch-and-unzip per build) — mention the cost for very broad requests.",
            {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for",
                    },
                    "days_back": {"type": "integer", "description": "Default 5"},
                    "limit": {"type": "integer", "description": "Default 200"},
                },
                "required": ["pattern"],
            },
            dashboard_tools.search_build_yaml,
        ),
        _wrap(
            "search_build_errors",
            "Free-text search over precomputed AI analysis (summary/root cause) across many builds.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "days_back": {"type": "integer", "description": "Default 7"},
                    "limit": {"type": "integer", "description": "Default 200"},
                },
                "required": ["query"],
            },
            dashboard_tools.search_build_errors,
        ),
        _wrap(
            "get_ai_analysis",
            "Fetch stored AI analysis (root cause, summary, suggested action) for one build.",
            {
                "type": "object",
                "properties": {"build_id": {"type": "string"}},
                "required": ["build_id"],
            },
            dashboard_tools.get_ai_analysis,
        ),
        _wrap(
            "compare_builds",
            "Compare two or more builds — diffs their build.yaml and key fields (status, space, user).",
            {
                "type": "object",
                "properties": {
                    "build_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                    },
                    "days_back": {
                        "type": "integer",
                        "description": "YAML lookup window, default 30",
                    },
                },
                "required": ["build_ids"],
            },
            dashboard_tools.compare_builds,
        ),
        _wrap(
            "wait_for_build",
            "Poll a build's status until it reaches a terminal state or times out (max 30 minutes).",
            {
                "type": "object",
                "properties": {
                    "build_id": {"type": "string"},
                    "timeout_minutes": {
                        "type": "integer",
                        "description": "Default 15, max 30",
                    },
                    "poll_interval_seconds": {
                        "type": "integer",
                        "description": "Default 15, min 10",
                    },
                },
                "required": ["build_id"],
            },
            dashboard_tools.wait_for_build,
        ),
        _wrap(
            "list_artifacts",
            "List registered artifacts, optionally filtered by build, space, tag, or username.",
            {
                "type": "object",
                "properties": {
                    "build_id": {"type": "string"},
                    "space_name": {"type": "string"},
                    "tag": {"type": "string"},
                    "username": {"type": "string"},
                },
            },
            dashboard_tools.list_artifacts,
        ),
        _wrap(
            "describe_artifact",
            "Fetch full metadata for one artifact by ID (URI, checksum, tags, lineage).",
            {
                "type": "object",
                "properties": {"artifact_id": {"type": "string"}},
                "required": ["artifact_id"],
            },
            dashboard_tools.describe_artifact,
        ),
    ]

    if config.cloud_logs_url and config.cloud_logs_api_key:
        specs.append(
            _wrap(
                "search_build_logs",
                "Search a build's logs for a substring across its recent history. Prefer this over "
                "build_log when the user wants to search, not just see the latest log.",
                {
                    "type": "object",
                    "properties": {
                        "build_id": {"type": "string"},
                        "search": {
                            "type": "string",
                            "description": "Substring to filter for",
                        },
                        "tail": {
                            "type": "integer",
                            "description": "Max lines to return, default 500",
                        },
                    },
                    "required": ["build_id"],
                },
                dashboard_tools.search_build_logs,
            )
        )

    return specs
