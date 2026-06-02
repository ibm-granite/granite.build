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

"""APIRouter for build event subscription (RabbitMQ streaming)."""

import os

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from gbserver.messaging.rabbitmq_admin import RabbitMQAdmin
from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.constants import (
    GBSERVER_BUILD_EVENTS_EXCHANGE,
    GBSERVER_EVENT_SUBSCRIBE_TTL,
    GBSERVER_RABBITMQ_MGMT_PASSWORD,
    GBSERVER_RABBITMQ_MGMT_URL,
    GBSERVER_RABBITMQ_MGMT_USER,
)
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

event_subscribe_router = APIRouter(prefix="/builds", tags=["events"])


# ── Response Model ────────────────────────────────────────────────────────


class SubscribeResponse(BaseModel):
    rabbitmq_host: str
    rabbitmq_port: int
    username: str
    password: str
    exchange: str
    routing_key: str
    queue: str
    expires_at: str


# ── Endpoint ──────────────────────────────────────────────────────────────


@event_subscribe_router.post(
    "/{build_id}/events/subscribe",
    response_model=SubscribeResponse,
    status_code=status.HTTP_200_OK,
)
async def subscribe_build_events(build_id: str, request: Request) -> SubscribeResponse:
    """Subscribe to real-time build events via RabbitMQ.

    Provisions scoped, time-limited RabbitMQ credentials that allow the
    caller to consume events for the specified build only.
    """
    # 1. Authenticate — the AuthMiddleware populates request.state.data["user"]
    user = getattr(request.state, "data", {}).get("user")
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization required.",
        )

    # 2. Verify build exists
    storage = get_admin_storage()
    build = storage.build_storage.get_by_uuid(build_id)
    if build is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Build {build_id} not found.",
        )
    assert isinstance(build, StoredBuild)

    # 3. Provision scoped RabbitMQ credentials
    admin = RabbitMQAdmin(
        management_url=GBSERVER_RABBITMQ_MGMT_URL,
        admin_user=GBSERVER_RABBITMQ_MGMT_USER,
        admin_password=GBSERVER_RABBITMQ_MGMT_PASSWORD,
    )

    credentials = await admin.create_scoped_user(
        build_id=build_id,
        exchange=GBSERVER_BUILD_EVENTS_EXCHANGE,
        ttl_seconds=GBSERVER_EVENT_SUBSCRIBE_TTL,
    )

    # 4. Build response with connection info
    rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
    rabbitmq_port = int(os.getenv("RABBITMQ_PORT", "5672"))
    username = credentials["username"]

    # Use last segment of username as suffix for queue naming
    username_suffix = username.rsplit("-", 1)[-1] if "-" in username else username

    return SubscribeResponse(
        rabbitmq_host=rabbitmq_host,
        rabbitmq_port=rabbitmq_port,
        username=credentials["username"],
        password=credentials["password"],
        exchange=GBSERVER_BUILD_EVENTS_EXCHANGE,
        routing_key=f"build.{build_id}.#",
        queue=f"events.{build_id}.{username_suffix}",
        expires_at=credentials["expires_at"],
    )
