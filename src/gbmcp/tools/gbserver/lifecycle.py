"""MCP tools to manage the local gbserver backend that the build tools call.

gbmcp runs as its own stdio process, so its tools are available even when
gbserver is down. Call ``gbserver_status`` before build tools if unsure, and
``gbserver_start`` to bring the backend up.
"""

import json

from fastmcp.tools import tool

from gbmcp.utils import gbserver_process as gp


@tool(description="Check whether the local gbserver backend is running and reachable.")
def gbserver_status() -> str:
    """Report gbserver process + HTTP reachability as JSON."""
    p = gp.port()
    running = gp.is_running(p)
    reachable = gp.is_reachable(p)
    result = {
        "port": p,
        "url": gp.base_url(p),
        "process_running": running,
        "reachable": reachable,
        "ready": running and reachable,
    }
    if not reachable and gp.foreign_on_port(p):
        result["note"] = (
            f"Port {p} is held by another process (not gbserver). Its port is fixed "
            "when this MCP server starts and can't change mid-session — set "
            "GBSERVER_PORT to a free port and restart the MCP server."
        )
    return json.dumps(result, indent=2)


@tool(description="Start the local gbserver backend if not running; waits until ready.")
def gbserver_start() -> str:
    """Idempotently start ``gbserver standalone`` and wait until it answers.

    Returns JSON; on failure to become ready, inlines the tail of the server log.
    """
    p = gp.port()
    if gp.is_running(p):
        return json.dumps(
            {"already_running": True, "ready": gp.is_reachable(p), "port": p}, indent=2
        )
    try:
        gp.start(p)
    except (RuntimeError, OSError) as exc:
        return json.dumps({"started": False, "port": p, "error": str(exc)}, indent=2)
    ready = gp.wait_for_ready(p)
    result = {"started": True, "ready": ready, "port": p, "url": gp.base_url(p)}
    if not ready:
        if gp.foreign_on_port(p):
            result["error"] = (
                f"Port {p} is held by another process, so gbserver could not bind it. "
                "Its port is fixed when this MCP server starts — set GBSERVER_PORT to "
                "a free port and restart the MCP server."
            )
        result["log_tail"] = gp.tail(gp.log_path(p), 40)
    return json.dumps(result, indent=2)


@tool(description="Stop the local gbserver backend for this port (idempotent).")
def gbserver_stop() -> str:
    """Stop the gbserver on GBSERVER_PORT. Returns JSON."""
    p = gp.port()
    return json.dumps({"was_running": gp.stop(p), "port": p}, indent=2)
