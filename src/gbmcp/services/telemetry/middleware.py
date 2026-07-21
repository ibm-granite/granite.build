"""Telemetry middleware — records every MCP tool call to PostgreSQL.

Intercepts on_call_tool via FastMCP's Middleware hook. All DB writes are
fire-and-forget (asyncio.create_task) so telemetry never blocks tool responses
and never surfaces errors to callers.
"""

import asyncio
import json
import os
import re
from datetime import datetime, timezone

from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_access_token
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.utilities.logging import get_logger

from gbmcp.services.telemetry.telemetry_db import get_telemetry_db

logger = get_logger(__name__)


def _get_server_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("gbmcp")
    except Exception:
        return None


def _result_length(result) -> int:
    """Sum of text content lengths across all content items in a ToolResult."""
    try:
        total = 0
        for item in result.content:
            text = getattr(item, "text", None)
            if text:
                total += len(text)
        return total
    except Exception:
        return 0


_SENSITIVE_PATTERNS = [
    # Connection strings: postgresql://user:pass@host, mysql://..., etc.
    (re.compile(r"://[^:/?#]+:[^@/?#]+@"), "://<REDACTED>@"),
    # Bearer / token auth headers
    (
        re.compile(r"(Bearer|Token|Authorization)\s+\S+", re.IGNORECASE),
        r"\1 <REDACTED>",
    ),
    # AWS-style access key IDs (AKIA...)
    (re.compile(r"AKIA[0-9A-Z]{16}"), "<REDACTED_AWS_KEY>"),
    # AWS secret keys (40-char base64 following common prefixes)
    (
        re.compile(
            r"(?i)(aws_secret_access_key|secret_access_key|secretaccesskey)[=: ]+\S+"
        ),
        r"\1=<REDACTED>",
    ),
    # Generic key=value secrets
    (
        re.compile(
            r"(?i)(password|passwd|secret|token|api_key|apikey|access_key|private_key)[=: ]+\S+"
        ),
        r"\1=<REDACTED>",
    ),
    # GitHub personal access tokens (ghp_, gho_, ghs_, ghu_, ghr_)
    (re.compile(r"gh[pousr]_[A-Za-z0-9_]{30,}"), "<REDACTED_GH_TOKEN>"),
]


def _sanitize_error_message(msg: str, max_length: int = 1000) -> str:
    """Truncate and redact sensitive patterns from an error message."""
    if not msg:
        return msg
    for pattern, replacement in _SENSITIVE_PATTERNS:
        msg = pattern.sub(replacement, msg)
    if len(msg) > max_length:
        msg = msg[:max_length] + "... [truncated]"
    return msg


async def _safe_insert(db, event: dict, session_data: dict) -> None:
    """Insert event and upsert session; swallow all exceptions."""
    try:
        await db.insert_tool_call_event(event)
    except Exception as e:
        logger.warning(f"Telemetry insert failed (swallowed): {e}")

    try:
        await db.upsert_session(session_data)
    except Exception as e:
        logger.warning(f"Telemetry session upsert failed (swallowed): {e}")


class TelemetryMiddleware(Middleware):
    """Records every tools/call invocation to gbmcp_tool_call_events."""

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        db = get_telemetry_db()

        started_at = datetime.now(timezone.utc)

        # Extract session/request context
        session_id: str | None = None
        mcp_request_id: str | None = None
        mcp_client_id: str | None = None
        try:
            ctx = context.fastmcp_context
            if ctx is not None:
                try:
                    session_id = ctx.session_id
                except Exception:
                    pass
                try:
                    mcp_request_id = ctx.request_id
                except Exception:
                    pass
                try:
                    mcp_client_id = ctx.client_id
                except Exception:
                    pass
        except Exception:
            pass

        # Extract GitHub identity from JWT claims
        github_username: str | None = None
        github_user_id: str | None = None
        github_email: str | None = None
        github_name: str | None = None
        try:
            access_token = get_access_token()
            if access_token is not None:
                claims = getattr(access_token, "claims", {}) or {}
                github_username = claims.get("login")
                github_user_id = str(claims.get("sub")) if claims.get("sub") else None
                github_email = claims.get("email")
                github_name = claims.get("name")
        except Exception:
            pass

        # Extract tool name and arguments from the message
        tool_name: str = ""
        arguments: dict = {}
        try:
            tool_name = context.message.name
            arguments = dict(context.message.arguments or {})
        except Exception:
            pass

        argument_keys = list(arguments.keys())
        argument_input_lengths = [len(json.dumps(v)) for v in arguments.values()]
        argument_count = len(arguments)

        # Call the tool
        success = False
        error_message: str | None = None
        error_type: str | None = None
        result = None
        exc_to_reraise = None

        try:
            result = await call_next(context)
            success = True
        except SystemExit as exc:
            # A tool (or gbcli) called sys.exit() — e.g. an operation
            # intentionally unsupported in standalone mode, or a malformed
            # ~/.gbcli/config. SystemExit is a BaseException and would otherwise
            # propagate through the ASGI task group and kill the whole server.
            # Convert it to a tool error so the client sees the message and the
            # server survives.
            if isinstance(exc.code, str) and exc.code:
                msg = exc.code  # gbcli's warning text (verbatim)
            elif exc.code not in (None, 0):
                msg = (
                    f"Tool exited with code {exc.code}. This operation may be "
                    f"unsupported in standalone mode; see server logs for details."
                )
            else:
                msg = "Tool exited unexpectedly."
            error_message = _sanitize_error_message(str(msg))
            error_type = "SystemExit"  # original type, for telemetry fidelity
            exc_to_reraise = ToolError(error_message)
        except Exception as exc:
            error_message = _sanitize_error_message(str(exc))
            error_type = type(exc).__name__
            exc_to_reraise = exc

        ended_at = datetime.now(timezone.utc)
        duration_ms = int((ended_at - started_at).total_seconds() * 1000)
        result_len = _result_length(result) if result is not None else None

        if db is not None:
            event = {
                "server_version": _get_server_version(),
                "session_id": session_id,
                "mcp_request_id": mcp_request_id,
                "mcp_client_id": mcp_client_id,
                "github_username": github_username,
                "github_user_id": github_user_id,
                "github_email": github_email,
                "github_name": github_name,
                "tool_name": tool_name,
                "tool_category": None,
                "tool_module": None,
                "started_at": started_at,
                "ended_at": ended_at,
                "duration_ms": duration_ms,
                "success": success,
                "error_message": error_message,
                "error_type": error_type,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "argument_keys": argument_keys,
                "argument_input_lengths": argument_input_lengths,
                "argument_count": argument_count,
                "result_length": result_len,
                "result_truncated": False,
                "server_instance_id": os.environ.get("SERVER_INSTANCE_ID"),
                "gb_environment": os.environ.get("GB_ENVIRONMENT"),
                "tool_version": None,
                "tool_git_commit": None,
                "mcp_server_commit": None,
                "mcp_image": None,
                "metadata": {},
            }
            session_data = {
                "session_id": session_id,
                "github_username": github_username,
                "github_user_id": github_user_id,
                "gb_environment": os.environ.get("GB_ENVIRONMENT"),
                "server_instance_id": os.environ.get("SERVER_INSTANCE_ID"),
                "success": success,
                "tool_name": tool_name,
            }
            asyncio.create_task(_safe_insert(db, event, session_data))

        if exc_to_reraise is not None:
            raise exc_to_reraise
        return result
