#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""
Background cleanup task for expired RabbitMQ temporary users.

Periodically scans for and removes temporary build users whose
time-to-live has elapsed.
"""

import asyncio

from gbserver.messaging.rabbitmq_admin import RabbitMQAdmin
from gbserver.types.constants import (
    GBSERVER_RABBITMQ_MGMT_PASSWORD,
    GBSERVER_RABBITMQ_MGMT_URL,
    GBSERVER_RABBITMQ_MGMT_USER,
    GBSERVER_RABBITMQ_TLS_VERIFY,
)
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

CLEANUP_INTERVAL_SECONDS = 60


def _get_admin() -> RabbitMQAdmin:
    return RabbitMQAdmin(
        management_url=GBSERVER_RABBITMQ_MGMT_URL,
        admin_user=GBSERVER_RABBITMQ_MGMT_USER,
        admin_password=GBSERVER_RABBITMQ_MGMT_PASSWORD,
        tls_verify=GBSERVER_RABBITMQ_TLS_VERIFY,
    )


async def run_cleanup_once() -> int:
    """Run a single cleanup pass. Returns number of users deleted."""
    try:
        admin = _get_admin()
        deleted = await admin.cleanup_expired_users()
        if deleted > 0:
            logger.info("[CredentialCleanup] Deleted %d expired user(s)", deleted)
        return deleted
    except Exception as e:
        logger.warning("[CredentialCleanup] Cleanup failed (will retry): %s", e)
        return 0


async def start_cleanup_loop() -> None:
    """Run cleanup in a loop. Intended to be started as a background task."""
    logger.info(
        "[CredentialCleanup] Starting cleanup loop (interval=%ds)",
        CLEANUP_INTERVAL_SECONDS,
    )
    while True:
        await run_cleanup_once()
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
