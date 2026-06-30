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

"""APIRouter for build event subscription."""

import httpx
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from gbserver.messaging.rabbitmq_admin import RabbitMQAdminError
from gbserver.messaging.subscription_service import provision_subscription
from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.storage.stored_build import StoredBuild
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

event_subscribe_router = APIRouter(prefix="/builds", tags=["events"])


# ── Response Model ────────────────────────────────────────────────────────


class SubscribeResponse(BaseModel):
    delivery_type: str
    host: str
    port: int
    tls: bool = False
    username: str | None = None
    password: str | None = None
    exchange: str | None = None
    routing_key: str | None = None
    queue: str | None = None
    url: str | None = None
    subject: str | None = None
    expires_at: int


# ── Endpoint ──────────────────────────────────────────────────────────────


@event_subscribe_router.post(
    "/{build_id}/events/subscribe",
    response_model=SubscribeResponse,
    status_code=status.HTTP_200_OK,
)
async def subscribe_build_events(build_id: str, request: Request) -> SubscribeResponse:
    """Subscribe to real-time build events.

    Returns connection info for consuming events for the specified build.
    In RabbitMQ mode: provisions scoped, time-limited credentials.
    In standalone/NATS mode: returns URL and subject (no auth; single-tenant).
    """
    # 1. Authenticate
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

    # 3. Provision credentials via messaging layer
    try:
        result = await provision_subscription(build_id)
    except RabbitMQAdminError as exc:
        logger.error(
            "RabbitMQ admin API error while provisioning subscription "
            "for build %s: %s",
            build_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Event subscription service unavailable.",
        ) from exc
    except httpx.ConnectError as exc:
        logger.error(
            "Cannot reach RabbitMQ management API for build %s: %s",
            build_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Event subscription service unreachable.",
        ) from exc
    except httpx.TimeoutException as exc:
        logger.error(
            "Timeout contacting RabbitMQ management API for build %s: %s",
            build_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Event subscription service timed out.",
        ) from exc
    return SubscribeResponse(**result)
