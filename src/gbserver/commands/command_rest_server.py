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


import json
import os
import sys

import click
import uvicorn

from gbcommon.types.constants import get_gb_home_dir
from gbserver.storage.sqlite.sqlite_storage import SQLITE_DB_FILE_NAME
from gbserver.types.constants import (
    ENV_VAR_METADATA_STORAGE,
    GBSERVER_REST_SERVER_TIMEOUT_KEEP_ALIVE,
    GBSERVER_REST_SERVER_WORKERS,
    analytics_backend_enabled,
    derive_analytics_database_url,
    derive_analytics_sql_connect_args,
)
from gbserver.types.context import CliEnvironment, pass_environment
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


def _configure_analytics_env(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Default the analytics service's env vars if gb_ui_backend is installed.

    The analytics routers are included directly into root_api (see
    gbserver/api/root_api.py) — this only sets sensible defaults for the env
    vars gb_ui_backend's Config reads, so standalone analytics work out of
    the box without requiring the caller to configure a database explicitly.

    Does nothing unless analytics is enabled here (analytics_backend_enabled);
    an API-only rest-server thus never gets the database default below, which
    would otherwise crash startup on an unwritable path. root_api re-resolves the
    same way per worker off the same inherited env.

    If GB_UI_DATABASE_URL is not set, it's derived from the main store's own backend
    config (see derive_analytics_database_url in gbserver.types.constants):
    GBSERVER_METADATA_STORAGE=sql inherits GBSERVER_SQL_* as a postgresql+asyncpg
    URL (TLS cert, if any, carried via GB_UI_DATABASE_CONNECT_ARGS below);
    GBSERVER_METADATA_STORAGE=sqlite defaults to the analytics service's own
    SQLite file under the GB home directory (see ANALYTICS_DB_FILENAME in
    gb_ui_backend/config.py).
    If GB_UI_GBSERVER_DB_URL is not set and gbserver is running in SQLite mode,
    defaults to gbserver's own SQLite file so standalone analytics work out of the box.
    If GB_UI_GBSERVER_URL is not set, defaults to the main server's own host/port.

    Args:
        host: Bind address the main REST server (and frontend) is listening on.
        port: Port the main REST server (and frontend) is listening on.
    """
    if not analytics_backend_enabled():
        logger.info(
            "Analytics not enabled — skipping analytics env defaulting "
            "(no database URL fallback for GB_UI_DATABASE_URL)"
        )
        return

    gb_home = get_gb_home_dir()

    if not os.environ.get("GB_UI_DATABASE_URL"):
        derived_url = derive_analytics_database_url()
        if derived_url is not None:
            os.environ["GB_UI_DATABASE_URL"] = derived_url
            if derived_url.startswith("postgresql+asyncpg://"):
                connect_args = derive_analytics_sql_connect_args()
                if connect_args:
                    os.environ["GB_UI_DATABASE_CONNECT_ARGS"] = json.dumps(connect_args)

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

    # gb_ui_backend.config.get_config() is lru_cached — the analytics_backend_enabled()
    # call at the top of this function already constructed and cached a Config read
    # from os.environ *before* the env vars above were set, so every later consumer
    # (init_analytics(), _get_engine()) would otherwise see a stale, pre-mutation
    # instance with database_url="" — analytics silently never gets a database despite
    # GB_UI_DATABASE_URL being set correctly in the process environment. Clear it now
    # that all GB_UI_* mutations in this function are done, so the next get_config()
    # call picks up the real values.
    from gb_ui_backend.config import get_config

    get_config.cache_clear()


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

    _configure_analytics_env(host="0.0.0.0", port=port)

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
