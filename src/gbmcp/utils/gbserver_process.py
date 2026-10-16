"""Locate, launch, probe, and stop a local ``gbserver standalone`` process.

gbmcp and gbserver ship in the same distribution, so the ``gbserver`` executable
sits next to the interpreter running this process — no checkout, clone, or venv
build is needed here (that install is a one-time prerequisite of running gbmcp
at all). Backs the ``gbserver_start`` / ``gbserver_stop`` / ``gbserver_status``
tools.
"""

import os
import shutil
import subprocess
import sys
import time

import httpx

DEFAULT_PORT = 8080
REACHABLE_TIMEOUT = 2.0
READY_TIMEOUT = 30.0

# A gbserver-specific route used to confirm it's really gbserver answering the
# port — not, say, another user's inference server on a shared box. A foreign
# HTTP server 404s it, so we don't false-positive on it.
READY_PATH = "/api/v1/spaces/spaces_for_user"


def port() -> int:
    return int(os.environ.get("GBSERVER_PORT", DEFAULT_PORT))


def base_url(p: int | None = None) -> str:
    return f"http://127.0.0.1:{p if p is not None else port()}"


def log_path(p: int | None = None) -> str:
    p = p if p is not None else port()
    return os.environ.get("GBSERVER_LOG_PATH", f"/tmp/gbserver-{p}.log")


# pgrep/pkill -f regex. The port is boundary-anchored (space or end) so a
# shorter port isn't a prefix-match of a longer one (e.g. 808 vs 8080).
def _match(p: int) -> str:
    return f"gbserver standalone --port {p}( |$)"


def resolve_bin() -> str:
    explicit = os.environ.get("GBSERVER_BIN")
    if explicit:
        if not (os.path.isfile(explicit) and os.access(explicit, os.X_OK)):
            raise RuntimeError(f"GBSERVER_BIN={explicit!r} is not an executable file")
        return explicit
    sibling = os.path.join(os.path.dirname(sys.executable), "gbserver")
    if os.path.isfile(sibling) and os.access(sibling, os.X_OK):
        return sibling
    found = shutil.which("gbserver")
    if found:
        return found
    raise RuntimeError(
        "No gbserver executable found next to this interpreter or on PATH. "
        "Install with `pip install 'granite.build[standalone]'`, or build a "
        "checkout with `make standalone-venv`, or set GBSERVER_BIN."
    )


def is_running(p: int | None = None) -> bool:
    p = p if p is not None else port()
    return (
        subprocess.run(["pgrep", "-f", _match(p)], capture_output=True).returncode == 0
    )


def is_reachable(p: int | None = None, timeout: float = REACHABLE_TIMEOUT) -> bool:
    """True only if *gbserver* answers on the port — not just any HTTP server.

    Probes a gbserver-specific route so a foreign process holding the port
    doesn't get mistaken for a ready gbserver.
    """
    p = p if p is not None else port()
    try:
        resp = httpx.get(f"{base_url(p)}{READY_PATH}", timeout=timeout)
    except httpx.RequestError:
        return False
    return resp.status_code == 200


def foreign_on_port(p: int | None = None, timeout: float = REACHABLE_TIMEOUT) -> bool:
    """True if something answers HTTP on the port but it isn't gbserver.

    Distinguishes "gbserver failed to start" from "another process already holds
    this port" (common on shared machines).
    """
    p = p if p is not None else port()
    try:
        httpx.get(base_url(p), timeout=timeout)
    except httpx.RequestError:
        return False
    return not is_reachable(p, timeout=timeout)


def start(p: int | None = None) -> None:
    """Launch gbserver detached so it outlives this tool call and the session."""
    p = p if p is not None else port()
    cmd = [resolve_bin(), "standalone", "--port", str(p)]
    space_dir = os.environ.get("GBSERVER_SPACE_DIR")
    if space_dir:
        cmd += ["--space-dir", space_dir]
    # start_new_session detaches gbserver so it outlives this process; the child
    # keeps its own dup of the log fd, so closing the parent's copy here is safe.
    with open(log_path(p), "ab") as log:
        subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )


def stop(p: int | None = None) -> bool:
    p = p if p is not None else port()
    if not is_running(p):
        return False
    subprocess.run(["pkill", "-f", _match(p)], capture_output=True)
    return True


def wait_for_ready(
    p: int | None = None, timeout: float = READY_TIMEOUT, interval: float = 0.5
) -> bool:
    p = p if p is not None else port()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_running(p) and is_reachable(p):
            return True
        time.sleep(interval)
    return False


def tail(path: str, lines: int) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read().splitlines()
    except FileNotFoundError:
        return ""
    return "\n".join(content[-lines:])
