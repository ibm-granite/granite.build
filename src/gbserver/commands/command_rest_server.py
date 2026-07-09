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


import atexit
import os
import subprocess
import sys

import click
import uvicorn

from gbcommon.types.constants import get_gb_home_dir
from gbserver.storage.sqlite.sqlite_storage import SQLITE_DB_FILE_NAME
from gbserver.types.constants import (
    ENV_VAR_METADATA_STORAGE,
    GBSERVER_REST_SERVER_TIMEOUT_KEEP_ALIVE,
    GBSERVER_REST_SERVER_WORKERS,
)
from gbserver.types.context import CliEnvironment, pass_environment
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

_sidecar_process: "subprocess.Popen[bytes] | None" = None


def _start_analytics_sidecar(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Start gb_ui_backend as a background subprocess if it is installed.

    If GB_UI_DATABASE_URL is not set, defaults to the sidecar's own SQLite file in
    the GB home directory (see SIDECAR_DB_FILENAME in gb_ui_backend/config.py).
    If GB_UI_GBSERVER_DB_URL is not set and gbserver is running in SQLite mode,
    defaults to gbserver's own SQLite file so standalone analytics work out of the box.
    If GB_UI_GBSERVER_URL is not set, defaults to the main server's own host/port so
    the sidecar can report the frontend URL at the end of its own startup log.

    Args:
        host: Bind address the main REST server (and frontend) is listening on.
        port: Port the main REST server (and frontend) is listening on.
    """
    import importlib.util

    if importlib.util.find_spec("gb_ui_backend") is None:
        return

    global _sidecar_process
    gb_home = get_gb_home_dir()

    if not os.environ.get("GB_UI_DATABASE_URL"):
        # mirrors SIDECAR_DB_FILENAME in gb_ui_backend/config.py — keep in sync
        os.environ["GB_UI_DATABASE_URL"] = (
            f"sqlite+aiosqlite:///{os.path.join(gb_home, 'dashboard-analytics.db')}"
        )

    if (
        not os.environ.get("GB_UI_GBSERVER_DB_URL")
        and os.environ.get(ENV_VAR_METADATA_STORAGE, "sql").lower() == "sqlite"
    ):
        os.environ["GB_UI_GBSERVER_DB_URL"] = (
            f"sqlite+aiosqlite:///{os.path.join(gb_home, SQLITE_DB_FILE_NAME)}"
        )

    if not os.environ.get("GB_UI_GBSERVER_URL"):
        browse_host = "127.0.0.1" if host == "0.0.0.0" else host
        os.environ["GB_UI_GBSERVER_URL"] = f"http://{browse_host}:{port}"

    _sidecar_process = subprocess.Popen(
        [sys.executable, "-m", "gb_ui_backend"],
        start_new_session=True,  # own process group — immune to gbserver SIGTERM/SIGHUP
    )
    logger.info(
        "Analytics sidecar started (pid=%d) — listening on :8090.",
        _sidecar_process.pid,
    )
    atexit.register(_stop_analytics_sidecar)


def _stop_analytics_sidecar() -> None:
    if _sidecar_process is not None and _sidecar_process.poll() is None:
        logger.info("Stopping analytics sidecar (pid=%d).", _sidecar_process.pid)
        _sidecar_process.terminate()
        try:
            _sidecar_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _sidecar_process.kill()


_IBMID_REQUIRED_VARS = [
    "GBSERVER_IBMID_CLIENT_ID",
    "GBSERVER_IBMID_CLIENT_SECRET",
    "GBSERVER_IBMID_CALLBACK_URL",
]


@click.command()
@click.option("--port", default=8080, type=int, help="Set the port to listen on.")
@pass_environment
def cli(
    ctx: CliEnvironment,
    port: int,
):
    """Start the REST API server."""
    auth_mode = os.getenv("GBSERVER_AUTH_MODE", "github")

    if auth_mode in ("ibmid", "multi"):
        missing = [v for v in _IBMID_REQUIRED_VARS if not os.getenv(v)]
        if missing:
            logger.error(
                "GBSERVER_AUTH_MODE=%s requires the following env vars: %s",
                auth_mode,
                ", ".join(missing),
            )
            sys.exit(1)

    _start_analytics_sidecar(host="0.0.0.0", port=port)

    try:
        logger.info(
            "Starting GB REST server on port %d (auth_mode=%s)", port, auth_mode
        )
        # inherit the logging configuration
        # "host" is needed to make the server listen outside localhost
        uvicorn.run(
            "gbserver.api.root_api:root_api",
            port=port,
            host="0.0.0.0",
            workers=GBSERVER_REST_SERVER_WORKERS,
            timeout_keep_alive=GBSERVER_REST_SERVER_TIMEOUT_KEEP_ALIVE,
            log_config=None,
        )
    finally:
        logger.warning("server stopped!")
