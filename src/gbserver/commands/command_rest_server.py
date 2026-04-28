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


import os
import sys

import click
import uvicorn

from gbserver.types.constants import (
    GBSERVER_REST_SERVER_TIMEOUT_KEEP_ALIVE,
    GBSERVER_REST_SERVER_WORKERS,
)
from gbserver.types.context import CliEnvironment, pass_environment
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

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
