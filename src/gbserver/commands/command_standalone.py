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

from gbserver.types.context import CliEnvironment, pass_environment
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# Standalone-friendly env var defaults — only set if not already defined.
_STANDALONE_ENV_DEFAULTS = {
    "GB_ENVIRONMENT": "STANDALONE",
    "GBSERVER_METADATA_STORAGE": "sqlite",
    "GBSERVER_DEFAULT_BUILDRUNNER_TYPE": "thread",
    "GBSERVER_AUTH_MODE": "apikey",
}


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

    cmd = [binary, "-js", "-sd", data_dir, "-p", str(port)]
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
    """
    # 1. Force GB_ENVIRONMENT to STANDALONE — this is not optional when
    #    running the standalone command, regardless of prior env settings.
    os.environ["GB_ENVIRONMENT"] = "STANDALONE"

    # Set remaining defaults only if not already set (user may override these).
    for key, value in _STANDALONE_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)

    # Re-evaluate constants that were captured at import time before our
    # env-var defaults were applied (e.g. GB_METADATA_STORAGE, GB_ENVIRONMENT).
    import importlib

    import gbserver.types.constants

    importlib.reload(gbserver.types.constants)

    logger.info(
        "Starting gbserver standalone on %s:%d with space-dir %s", host, port, space_dir
    )

    # 2. Force SQLite storage — standalone always uses SQLite.
    from gbserver.storage import singleton_storage
    from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory
    from gbserver.storage.stored_space import StoredSpace

    singleton_storage.set_storage_factory(SqliteStorageFactory())

    # Use standalone space access manager — bypasses Lakehouse authorization.
    from gbserver.spaces.space_access_manager import set_space_access_manager
    from gbserver.spaces.standalone_space_access_manager import (
        StandaloneSpaceAccessManager,
    )

    set_space_access_manager(StandaloneSpaceAccessManager())

    # Register both 'standalone' (legacy alias) and 'public' (matches the new
    # space.yaml name) pointing at the same directory.  The trailing-slash
    # variant on the second URI sidesteps the unique index on git_repo_uri
    # without changing the schema; both URIs resolve to the same dir on disk.
    storage = singleton_storage.get_admin_storage()
    abs_dir = os.path.abspath(space_dir)
    space_aliases = [
        ("standalone", f"file://{abs_dir}"),
        ("public", f"file://{abs_dir}/"),
    ]
    for name, uri in space_aliases:
        existing = storage.space_storage.get_by_name(name)
        if existing is None:
            storage.space_storage.add(
                StoredSpace(
                    name=name,
                    git_repo_uri=uri,
                    lakehouse_namespace="",
                )
            )
            logger.info("Created '%s' space with URI %s", name, uri)
        else:
            logger.info("'%s' space already exists (uuid=%s)", name, existing.uuid)

    # 2.5. Start embedded nats-server if configured.
    from gbserver.types.constants import GBSERVER_NATS_EMBEDDED, GBSERVER_NATS_URL

    nats_proc = None
    if GBSERVER_NATS_EMBEDDED:
        nats_proc = _start_nats_server(space_dir, nats_url=GBSERVER_NATS_URL)

    # 3. Start a BuildWatcher in a background daemon thread.
    #    Force thread runner — BuildWatcherConfig defaults to "job" (k8s)
    #    because DEFAULT_BUILDRUNNER_TYPE is evaluated at import time.
    from gbserver.buildwatcher.buildwatcher import BuildWatcher

    build_watcher = BuildWatcher(
        config_path=None,
        watch_for_config_changes=False,
        gh_token="",
    )
    build_watcher.config.buildrunner_type = "thread"

    watcher_thread = threading.Thread(
        target=build_watcher.start_and_wait,
        name="standalone-build-watcher",
        daemon=True,
    )
    watcher_thread.start()
    logger.info("BuildWatcher started in background thread")

    # 4. Start the REST API via uvicorn.
    config = uvicorn.Config(
        "gbserver.api.root_api:root_api",
        port=port,
        host=host,
        workers=1,
        log_config=None,
    )
    server = uvicorn.Server(config)

    if on_started:
        original_startup = server.startup

        async def _startup_with_callback(*args, **kwargs):
            await original_startup(*args, **kwargs)
            on_started()

        server.startup = _startup_with_callback  # type: ignore[assignment]

    try:
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
    default="configurations/spaces/standalone/public",
    show_default=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to the space directory.  Defaults to the in-repo standalone "
    "space at configurations/spaces/standalone/public; override to point at "
    "any directory containing a space.yaml.",
)
@pass_environment
def cli(ctx: CliEnvironment, port: int, host: str, space_dir: str):
    """Run gbserver standalone -- REST API + BuildWatcher in one process."""
    _run_standalone(port=port, host=host, space_dir=space_dir)
