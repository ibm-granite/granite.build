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

"""
RabbitMQ Management API client for dynamic user provisioning.

Provides temporary, scoped RabbitMQ credentials for build event consumers.
Users are created with time-limited access scoped to a single build's events
and are cleaned up after expiry.
"""

from __future__ import annotations

import secrets
import string
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from urllib.parse import quote as urlquote

import httpx

from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class RabbitMQAdminError(Exception):
    """Raised when a RabbitMQ Management API call fails."""


class RabbitMQAdmin:
    """
    Client for the RabbitMQ Management HTTP API.

    Used to dynamically provision and clean up temporary users scoped
    to specific builds for event streaming.
    """

    def __init__(
        self,
        management_url: str,
        admin_user: str,
        admin_password: str,
        vhost: str = "/",
    ) -> None:
        """
        Parameters
        ----------
        management_url : str
            Base URL of the RabbitMQ Management API (e.g. "http://localhost:15672").
        admin_user : str
            Administrator username for the Management API.
        admin_password : str
            Administrator password for the Management API.
        vhost : str
            RabbitMQ virtual host (default "/").
        """
        self.management_url = management_url.rstrip("/")
        self.admin_user = admin_user
        self.admin_password = admin_password
        self.vhost = vhost
        self._vhost_encoded = urlquote(vhost, safe="")

    def _auth(self) -> httpx.BasicAuth:
        return httpx.BasicAuth(self.admin_user, self.admin_password)

    @staticmethod
    def _random_string(length: int, alphabet: str | None = None) -> str:
        """Generate a cryptographically random string."""
        if alphabet is None:
            alphabet = string.ascii_letters + string.digits
        return "".join(secrets.choice(alphabet) for _ in range(length))

    async def create_scoped_user(
        self,
        build_id: str,
        exchange: str,
        ttl_seconds: int = 3600,
    ) -> Dict[str, Any]:
        """
        Create a temporary RabbitMQ user scoped to a specific build.

        The user gets read-only permissions on queues and topics matching
        the build ID pattern, allowing them to consume build events but
        not publish or access other builds' data.

        Parameters
        ----------
        build_id : str
            The build identifier to scope permissions to.
        exchange : str
            The exchange name to grant topic-permission read access on.
        ttl_seconds : int
            Time-to-live in seconds before the user expires (default 3600).

        Returns
        -------
        dict
            {"username": str, "password": str, "expires_at": int (epoch seconds)}
        """
        suffix = self._random_string(6)
        username = f"tmp-build-{build_id[:8]}-{suffix}"
        password = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        expires_epoch = int(expires_at.timestamp())

        async with httpx.AsyncClient(auth=self._auth()) as client:
            # 1. Create user
            user_url = f"{self.management_url}/api/users/{username}"
            user_body = {
                "password": password,
                "tags": f"tmp-build,expires:{expires_epoch}",
            }
            resp = await client.put(user_url, json=user_body)
            if resp.status_code not in (201, 204):
                raise RabbitMQAdminError(
                    f"Failed to create user {username}: "
                    f"{resp.status_code} {resp.text}"
                )
            logger.info(
                "Created RabbitMQ user %s (expires %s)", username, expires_epoch
            )

            # 2. Set queue permissions (read-only, scoped to build)
            perm_url = (
                f"{self.management_url}/api/permissions/"
                f"{self._vhost_encoded}/{username}"
            )
            exchange_escaped = exchange.replace(".", "\\.")
            perm_body = {
                "configure": f"events\\.{build_id}\\..*",
                "write": f"events\\.{build_id}\\..*",
                "read": f"({exchange_escaped}|events\\.{build_id}\\..*)",
            }
            resp = await client.put(perm_url, json=perm_body)
            if resp.status_code not in (201, 204):
                raise RabbitMQAdminError(
                    f"Failed to set permissions for {username}: "
                    f"{resp.status_code} {resp.text}"
                )

            # 3. Set topic permissions (read-only on build.<build_id>.* pattern)
            topic_perm_url = (
                f"{self.management_url}/api/topic-permissions/"
                f"{self._vhost_encoded}/{username}"
            )
            topic_perm_body = {
                "exchange": exchange,
                "write": "",
                "read": f"build\\.{build_id}\\..*",
            }
            resp = await client.put(topic_perm_url, json=topic_perm_body)
            if resp.status_code not in (201, 204):
                raise RabbitMQAdminError(
                    f"Failed to set topic permissions for {username}: "
                    f"{resp.status_code} {resp.text}"
                )

            logger.info(
                "Configured scoped permissions for user %s on build %s",
                username,
                build_id,
            )

        return {
            "username": username,
            "password": password,
            "expires_at": expires_epoch,
        }

    async def cleanup_expired_users(self) -> int:
        """
        Find and delete all temporary build users that have expired.

        Iterates through all RabbitMQ users, identifies those tagged with
        "tmp-build" and an "expires:{ISO8601}" timestamp in the past,
        and deletes them.

        Returns
        -------
        int
            Number of users deleted.
        """
        deleted = 0
        now = datetime.now(timezone.utc)

        async with httpx.AsyncClient(auth=self._auth()) as client:
            resp = await client.get(f"{self.management_url}/api/users")
            if resp.status_code != 200:
                raise RabbitMQAdminError(
                    f"Failed to list users: {resp.status_code} {resp.text}"
                )

            users: List[Dict[str, Any]] = resp.json()

            for user in users:
                tags = user.get("tags", "")
                if "tmp-build" not in tags:
                    continue

                # Parse expiry from tags (format: "tmp-build,expires:1748870400")
                expires_at = self._parse_expiry_from_tags(tags)
                if expires_at is None:
                    logger.warning(
                        "User %s has tmp-build tag but no parseable expiry: %s",
                        user.get("name"),
                        tags,
                    )
                    continue

                if expires_at <= now:
                    username = user["name"]
                    await self.delete_user(username, client=client)
                    deleted += 1

        logger.info("Cleaned up %d expired temporary users", deleted)
        return deleted

    async def delete_user(
        self, username: str, client: httpx.AsyncClient | None = None
    ) -> None:
        """
        Delete a specific RabbitMQ user.

        Parameters
        ----------
        username : str
            The username to delete.
        client : httpx.AsyncClient | None
            Optional pre-existing client to reuse (avoids opening a new connection).
        """

        async def _do_delete(c: httpx.AsyncClient) -> None:
            url = f"{self.management_url}/api/users/{username}"
            resp = await c.delete(url)
            if resp.status_code not in (200, 204):
                raise RabbitMQAdminError(
                    f"Failed to delete user {username}: "
                    f"{resp.status_code} {resp.text}"
                )
            logger.info("Deleted RabbitMQ user %s", username)

        if client is not None:
            await _do_delete(client)
        else:
            async with httpx.AsyncClient(auth=self._auth()) as c:
                await _do_delete(c)

    @staticmethod
    def _parse_expiry_from_tags(tags: str) -> datetime | None:
        """
        Extract the expiry datetime from a user's tags string.

        Tags format example: "tmp-build,expires:1748870400"
        """
        for part in tags.split(","):
            part = part.strip()
            if part.startswith("expires:"):
                value = part[len("expires:") :]
                try:
                    return datetime.fromtimestamp(int(value), tz=timezone.utc)
                except (ValueError, OSError):
                    return None
        return None
