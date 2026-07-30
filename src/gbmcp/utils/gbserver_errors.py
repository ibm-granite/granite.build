"""Rewrite gbserver-unreachable failures from gbcli into actionable messages.

Several `gbcli` call paths (e.g. `get_remote_spaces()` in `gbcli/utils/gbserver.py`)
swallow the underlying `requests.exceptions.ConnectionError` inside
`make_gbserver_call()` and re-raise a bare, generic `Exception` (e.g. "Error
getting spaces from GBServer."). gbmcp cannot fix that string at its source
(it lives in the gbcli dependency), but a tool wrapper CAN catch it and, when
a live probe confirms gbserver itself is actually unreachable, rewrite the
message into one the agent can act on. Any other exception (bad args, a 4xx
from gbserver, etc.) is re-raised unchanged -- this never swallows unrelated
errors, it only relabels the specific "backend is down" case.

Usage: stack `@actionable_gbserver_errors` under `@tool(...)` on any gbcli-
backed tool function:

    @tool(description="...")
    @actionable_gbserver_errors
    def build_list(...) -> str:
        ...
"""

import functools

import httpx
from fastmcp.exceptions import ToolError

from gbcli.utils.gbconstants import GBSERVER_INSTANCE
from gbcommon.types.gbenvconfig import is_standalone


class GBServerUnreachableError(ToolError):
    """Raised when gbserver itself is unreachable at GBSERVER_INSTANCE.

    Subclasses ToolError so the message always reaches the client (regardless
    of FastMCP's mask_error_details setting) instead of a generic gbcli
    exception string with no actionable next step.
    """


def is_gbserver_reachable(timeout: float = 2.0) -> bool:
    """Probe whether GBSERVER_INSTANCE (the URL gbcli/GBClient calls) is up.

    Any HTTP response (even a 404/401) counts as reachable -- this checks
    connectivity, not application-level success. Only a transport-level
    failure (connection refused, DNS failure, timeout, ...) counts as down.
    """
    try:
        httpx.get(GBSERVER_INSTANCE, timeout=timeout)
        return True
    except httpx.RequestError:
        return False


def _actionable_message() -> str:
    if is_standalone():
        hint = (
            "gbmcp runs as a separate process from gbserver, so the backend is "
            "likely just not started — call gbserver_status, then gbserver_start "
            "if it isn't ready."
        )
    else:
        hint = (
            "this looks like a transient backend connectivity issue -- retry "
            "shortly, or call info_gb_version to re-check."
        )
    return f"gbserver is not reachable at {GBSERVER_INSTANCE} — {hint}"


def actionable_gbserver_errors(func):
    """Decorator: rewrite gbserver-unreachable exceptions into an actionable message.

    Only active in standalone mode. On any exception raised by `func`, probes
    gbserver's reachability. If gbserver is unreachable, raises
    `GBServerUnreachableError` with a message telling the agent what to do next
    (chained from the original exception via `from e` so it's still visible in
    tracebacks/logs). Otherwise re-raises the original exception untouched --
    unrelated errors are never swallowed.

    Outside standalone mode (prod/staging/dev) the wrapper is a pass-through:
    the original exception is re-raised untouched with no reachability probe,
    so error semantics and latency there are unchanged.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ToolError:
            raise
        except Exception as e:
            if not is_standalone():
                raise
            if not is_gbserver_reachable():
                raise GBServerUnreachableError(_actionable_message()) from e
            raise

    return wrapper
