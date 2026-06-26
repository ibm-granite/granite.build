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

"""Service for provisioning event subscription credentials."""

from __future__ import annotations

import os
import time
from typing import Any, Dict
from urllib.parse import urlparse

from gbserver.types.constants import (
    GBSERVER_BUILD_EVENTS_EXCHANGE,
    GBSERVER_EVENT_SUBSCRIBE_TTL,
    GBSERVER_NATS_URL,
    GBSERVER_RABBITMQ_MGMT_PASSWORD,
    GBSERVER_RABBITMQ_MGMT_URL,
    GBSERVER_RABBITMQ_MGMT_USER,
)
from gbserver.utils.logger import get_logger
from gbserver.utils.optional_imports import HAS_NATS

logger = get_logger(__name__)


from gbcommon.types.gbenvconfig import is_standalone as _is_standalone


async def provision_subscription(build_id: str) -> Dict[str, Any]:
    """Provision credentials/connection info for consuming build events.

    In standalone/NATS mode: returns NATS url + subject (no credentials).
    In RabbitMQ mode: provisions scoped temporary user via Management API.
    """
    if _is_standalone() and HAS_NATS and not os.getenv("RABBITMQ_HOST"):
        return _provision_nats(build_id)

    return await _provision_rabbitmq(build_id)


def _provision_nats(build_id: str) -> Dict[str, Any]:
    """Return NATS connection info for standalone mode."""
    parsed = urlparse(GBSERVER_NATS_URL)
    host = parsed.hostname or "localhost"
    port = parsed.port or 4222

    # No expiry in standalone — set far-future (1 year)
    expires_at = int(time.time()) + 86400 * 365

    return {
        "delivery_type": "nats",
        "host": host,
        "port": port,
        "username": None,
        "password": None,
        "exchange": None,
        "routing_key": None,
        "queue": None,
        "url": GBSERVER_NATS_URL,
        "subject": f"gbserver.build.{build_id}.>",
        "expires_at": expires_at,
    }


async def _provision_rabbitmq(build_id: str) -> Dict[str, Any]:
    """Provision scoped RabbitMQ credentials via Management API."""
    from gbserver.messaging.rabbitmq_admin import RabbitMQAdmin

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

    host = os.getenv("RABBITMQ_HOST", "localhost")
    port = int(os.getenv("RABBITMQ_PORT", "5672"))
    username = credentials["username"]
    username_suffix = username.rsplit("-", 1)[-1] if "-" in username else username

    return {
        "delivery_type": "rabbitmq",
        "host": host,
        "port": port,
        "username": credentials["username"],
        "password": credentials["password"],
        "exchange": GBSERVER_BUILD_EVENTS_EXCHANGE,
        "routing_key": f"build.{build_id}.#",
        "queue": f"events.{build_id}.{username_suffix}",
        "url": None,
        "subject": None,
        "expires_at": credentials["expires_at"],
    }
