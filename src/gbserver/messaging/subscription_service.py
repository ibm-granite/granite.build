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
from typing import Any, Dict

from gbserver.messaging.rabbitmq_admin import RabbitMQAdmin
from gbserver.types.constants import (
    GBSERVER_BUILD_EVENTS_EXCHANGE,
    GBSERVER_EVENT_SUBSCRIBE_TTL,
    GBSERVER_RABBITMQ_MGMT_PASSWORD,
    GBSERVER_RABBITMQ_MGMT_URL,
    GBSERVER_RABBITMQ_MGMT_USER,
)
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


async def provision_subscription(build_id: str) -> Dict[str, Any]:
    """Provision scoped, time-limited credentials for consuming build events.

    Returns a dict with keys: host, port, username, password, exchange,
    routing_key, queue, expires_at, delivery_type.
    """
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
        "expires_at": credentials["expires_at"],
    }
