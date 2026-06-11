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


import asyncio

from fastapi import FastAPI

from gbserver.api import (  # noqa: F401  registers routes on builds_api
    build_files as _build_files,
)
from gbserver.api.artifacts import artifacts_api
from gbserver.api.auth import AuthMiddleware
from gbserver.api.auth_routes import auth_api
from gbserver.api.builds import builds_api
from gbserver.api.event_subscribe import event_subscribe_router
from gbserver.api.lineage import lineage_api
from gbserver.api.logs import logs_api
from gbserver.api.node_health import node_health_api
from gbserver.api.secrets import secrets_api
from gbserver.api.spaces import spaces_api
from gbserver.types.constants import (
    API_BASE_PATH,
    GBSERVER_EVENT_PUBLISHING_ENABLED,
    GBSERVER_GIT_COMMIT,
)
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


def get_app() -> FastAPI:
    app = FastAPI()
    if True:
        # perform an empty CLI call here just to let the work process go through the same init code in cli.py
        from gbserver.cli import gbserver

        gbserver(["rest-server-worker"], standalone_mode=False)
    return app


root_api = get_app()

root_api.add_middleware(AuthMiddleware)  # type: ignore[arg-type]


@root_api.get(API_BASE_PATH)
def read_root():
    return {
        "message": "Welcome to the REST API!",
        "git_commit": GBSERVER_GIT_COMMIT,
    }


root_api.include_router(event_subscribe_router, prefix=API_BASE_PATH)
root_api.mount(f"{API_BASE_PATH}/auth", auth_api)
root_api.mount(f"{API_BASE_PATH}/artifacts", artifacts_api)
root_api.mount(f"{API_BASE_PATH}/builds", builds_api)
root_api.mount(f"{API_BASE_PATH}/lineage", lineage_api)
root_api.mount(f"{API_BASE_PATH}/logs", logs_api)
root_api.mount(f"{API_BASE_PATH}/node-health", node_health_api)
root_api.mount(f"{API_BASE_PATH}/secrets", secrets_api)
root_api.mount(f"{API_BASE_PATH}/spaces", spaces_api)


@root_api.on_event("startup")
async def _start_background_tasks():
    """Launch background tasks that run for the lifetime of the server."""
    if GBSERVER_EVENT_PUBLISHING_ENABLED:
        from gbserver.messaging.credential_cleanup import start_cleanup_loop

        logger.info("Event publishing enabled — starting credential cleanup task")
        asyncio.create_task(start_cleanup_loop())
