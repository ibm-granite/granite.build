#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone mode: REST API + BuildWatcher in one process."""

import os
import shutil
import socket
import subprocess
import threading
import time
from typing import Callable, Optional
from urllib.parse import urlparse

import click
import uvicorn

from gbserver.commands.command_rest_server import _configure_analytics_env
from gbserver.commands.utils import check_and_init_for_standalone
from gbserver.types.constants import (
    CONFIGURATIONS_STANDALONE_SPACE_SUBPATH,
    ENV_VAR_CONFIGURATIONS_DIR,
    find_configurations_root,
)
from gbserver.types.context import CliEnvironment, pass_environment
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


def _default_space_dir() -> str:
    """Resolve the default standalone space directory from the packaged
    configurations tree, raising a clear error if it cannot be found."""
    configurations_root = find_configurations_root()
    if configurations_root is None:
        raise click.UsageError(
            "Could not locate the packaged 'configurations/' directory. Pass "
            "--space-dir explicitly (a directory containing a space.yaml), or "
            f"set {ENV_VAR_CONFIGURATIONS_DIR} to a configurations/ tree."
        )
    return str(configurations_root / CONFIGURATIONS_STANDALONE_SPACE_SUBPATH)


def _start_nats_server(
    space_dir: str,
    port: int = 4222,
    nats_url: str = "nats://localhost:4222",
) -> "subprocess.Popen | None":
    """Start an embedded nats-server with JetStream enabled.

    Returns the subprocess handle, or None if nats-server is not found.
    """
    binary = shutil.which("nats-server")
    if binary is None:
        logger.warning(
            "nats-server not found on PATH; NATS messaging disabled. "
            "Install from https://nats.io/download/"
        )
        return None

    # Parse port from nats_url if provided
    parsed = urlparse(nats_url)
    if parsed.port:
        port = parsed.port

    data_dir = os.path.join(space_dir, ".gbserver", "nats-data")
    os.makedirs(data_dir, exist_ok=True)

    cmd = [binary, "-js", "-sd", data_dir, "-p", str(port), "-a", "127.0.0.1"]
    logger.info("Starting embedded nats-server: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not _wait_for_nats(nats_url, timeout=10):
        logger.error("nats-server failed to start within 10 seconds")
        proc.terminate()
        proc.wait(timeout=5)
        return None

    logger.info("Embedded nats-server ready on port %d (pid=%d)", port, proc.pid)
    return proc


def _wait_for_nats(nats_url: str, timeout: int = 10) -> bool:
    """Wait for nats-server to accept connections."""
    parsed = urlparse(nats_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 4222

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def _stop_nats_server(proc: "subprocess.Popen | None") -> None:
    """Stop the embedded nats-server subprocess."""
    if proc is None:
        return
    if proc.poll() is None:
        logger.info("Stopping embedded nats-server (pid=%d)", proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
    logger.info("Embedded nats-server stopped")


def _run_standalone(
    port: int,
    space_dir: str,
    host: str = "127.0.0.1",
    on_started: Optional[Callable[[], None]] = None,
    on_server_created: Optional[Callable[["uvicorn.Server"], None]] = None,
) -> None:
    """Core standalone logic — usable from tests via *on_started* callback.

    1. Apply standalone-friendly env var defaults.
    2. Register the "standalone" space in SQLite storage.
    3. Start a BuildWatcher in a background daemon thread.
    4. Start the REST API via uvicorn (single worker, in-process).

    Args:
        port: TCP port for the REST API.
        space_dir: Path to the space directory (contains space.yaml, environments/, steps/).
        host: Bind address for the REST API (default: 127.0.0.1).
        on_started: Optional callback fired once the uvicorn server has finished startup.
        on_server_created: Optional callback fired with the ``uvicorn.Server`` as
            soon as it is constructed (before ``server.run()``).  Tests that run
            this function in a background thread use it to capture the server and
            later set ``server.should_exit = True`` for a graceful shutdown, which
            lets the ``finally`` below stop the BuildWatcher instead of leaking it.
    """
    # 1. Force GB_ENVIRONMENT to STANDALONE — this is not optional when
    #    running the standalone command, regardless of prior env settings.
    os.environ["GB_ENVIRONMENT"] = "STANDALONE"

    logger.info(
        "Starting gbserver standalone on %s:%d with space-dir %s", host, port, space_dir
    )

    # Apply standalone env defaults, reload constants, migrate any legacy db,
    # install the SQLite storage factory + standalone space access manager, and
    # register the standalone space (+ legacy aliases). Extracted to
    # commands.utils so it can be reused and unit-tested.
    check_and_init_for_standalone(space_dir)

    # 2.5. Start embedded nats-server if configured.
    from gbserver.types.constants import GBSERVER_NATS_EMBEDDED, GBSERVER_NATS_URL

    nats_proc = None
    if GBSERVER_NATS_EMBEDDED:
        nats_proc = _start_nats_server(space_dir, nats_url=GBSERVER_NATS_URL)

    # 3. Start a BuildWatcher in a background daemon thread.
    #    check_and_init_for_standalone() set GBSERVER_DEFAULT_BUILDRUNNER_TYPE to
    #    "thread", and BuildWatcherConfig reads it at instantiation, so the watcher
    #    uses the thread runner here (no k8s) without any explicit override.
    from gbserver.buildwatcher.buildwatcher import BuildWatcher

    build_watcher = BuildWatcher(
        config_path=None,
        watch_for_config_changes=False,
        gh_token="",
    )

    watcher_thread = threading.Thread(
        target=build_watcher.start_and_wait,
        name="standalone-build-watcher",
        daemon=True,
    )
    watcher_thread.start()
    logger.info("BuildWatcher started in background thread")

    # 4. Default the analytics service's env vars (if gb_ui_backend is installed).
    _configure_analytics_env(host=host, port=port)

    # 5. Start the REST API via uvicorn.
    #    Force the "asyncio" event loop (not uvloop) to avoid subprocess-in-thread
    #    issues on macOS: uvloop's SIGCHLD handling doesn't work in non-main threads,
    #    causing BuildRunner's process.communicate() to hang indefinitely.
    config = uvicorn.Config(
        "gbserver.api.root_api:root_api",
        port=port,
        host=host,
        workers=1,
        log_config=None,
        loop="asyncio",
    )
    server = uvicorn.Server(config)

    if on_server_created:
        on_server_created(server)

    if on_started:
        original_startup = server.startup

        async def _startup_with_callback(*args, **kwargs):
            await original_startup(*args, **kwargs)
            on_started()

        server.startup = _startup_with_callback  # type: ignore[assignment]

    try:
        logger.info("Starting REST server")
        server.run()
    finally:
        build_watcher.stop()
        _stop_nats_server(nats_proc)
        logger.warning("Standalone server stopped!")


@click.command()
@click.option(
    "--port",
    default=8080,
    type=int,
    help="Port for the REST API server.",
)
@click.option(
    "--host",
    default="127.0.0.1",
    help="Bind address (use 0.0.0.0 for all interfaces).",
)
@click.option(
    "--space-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to the space directory (a directory containing a space.yaml).  "
    "Defaults to the packaged standalone space (configurations/spaces/local), "
    "discovered relative to the install or the current repo checkout.  Set "
    "GBSERVER_CONFIGURATIONS_DIR to point discovery at a different "
    "configurations/ tree.",
)
@pass_environment
def cli(ctx: CliEnvironment, port: int, host: str, space_dir: Optional[str]):
    """Run gbserver standalone -- REST API + BuildWatcher in one process."""
    if space_dir is None:
        space_dir = _default_space_dir()
    logger.info("Using space directory: %s", space_dir)

    def _log_ready():
        browse_host = "127.0.0.1" if host == "0.0.0.0" else host
        # Bold just the URL, matching Uvicorn's own "Uvicorn running on <bold-url>"
        # startup banner so this line carries the same visual weight.
        url = f"http://{browse_host}:{port}"
        logger.info("Frontend + API available at \x1b[1m%s\x1b[0m", url)

    _run_standalone(port=port, host=host, space_dir=space_dir, on_started=_log_ready)
