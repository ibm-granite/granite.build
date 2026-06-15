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
Unit tests for RabbitMQAdmin Management API client.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from gbserver.messaging.rabbitmq_admin import RabbitMQAdmin, RabbitMQAdminError


@pytest.fixture
def admin():
    """Create a RabbitMQAdmin instance for testing."""
    return RabbitMQAdmin(
        management_url="http://localhost:15672",
        admin_user="admin",
        admin_password="secret",
        vhost="/",
    )


class TestRabbitMQAdminInit:
    """Tests for RabbitMQAdmin initialization."""

    def test_basic_init(self, admin):
        assert admin.management_url == "http://localhost:15672"
        assert admin.admin_user == "admin"
        assert admin.admin_password == "secret"
        assert admin.vhost == "/"
        assert admin._vhost_encoded == "%2F"

    def test_trailing_slash_stripped(self):
        a = RabbitMQAdmin(
            management_url="http://host:15672/",
            admin_user="u",
            admin_password="p",
        )
        assert a.management_url == "http://host:15672"

    def test_custom_vhost(self):
        a = RabbitMQAdmin(
            management_url="http://host:15672",
            admin_user="u",
            admin_password="p",
            vhost="/my-vhost",
        )
        assert a._vhost_encoded == "%2Fmy-vhost"


class TestCreateScopedUser:
    """Tests for create_scoped_user method."""

    @pytest.mark.asyncio
    async def test_create_scoped_user_success(self, admin):
        """Successful user creation returns username, password, and expiry."""
        build_id = "abcdef12-3456-7890-abcd-ef1234567890"

        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 201
        mock_response.text = ""

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.put = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await admin.create_scoped_user(
                build_id=build_id,
                exchange="build.events",
                ttl_seconds=7200,
            )

        assert "username" in result
        assert "password" in result
        assert "expires_at" in result

        # Username format: tmp-build-{first8chars}-{6char_suffix}
        assert result["username"].startswith(f"tmp-build-{build_id[:8]}-")
        assert len(result["username"]) == len("tmp-build-") + 8 + 1 + 6

        # Password is a URL-safe base64 token (32 bytes → 43 chars)
        assert len(result["password"]) == 43

        # Expiry is an epoch timestamp (integer)
        assert isinstance(result["expires_at"], int)
        assert result["expires_at"] > int(datetime.now(timezone.utc).timestamp())

        # Three PUT calls: user, permissions, topic-permissions
        assert mock_client.put.call_count == 3

    @pytest.mark.asyncio
    async def test_create_scoped_user_sets_correct_permissions(self, admin):
        """Verify the permission bodies sent to the Management API."""
        build_id = "abc12345-rest-of-id"

        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 204
        mock_response.text = ""

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.put = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            await admin.create_scoped_user(
                build_id=build_id,
                exchange="build.events",
                ttl_seconds=3600,
            )

        calls = mock_client.put.call_args_list

        # Call 1: Create user
        user_url = calls[0].args[0]
        user_body = calls[0].kwargs["json"]
        assert "/api/users/tmp-build-abc12345-" in user_url
        assert "password" in user_body
        assert "tmp-build" in user_body["tags"]
        assert "expires:" in user_body["tags"]

        # Call 2: Queue permissions
        perm_url = calls[1].args[0]
        perm_body = calls[1].kwargs["json"]
        assert "/api/permissions/%2F/" in perm_url
        assert perm_body["configure"] == f"events\\.{build_id}\\..*"
        assert perm_body["write"] == f"events\\.{build_id}\\..*"
        assert f"events\\.{build_id}\\..*" in perm_body["read"]

        # Call 3: Topic permissions
        topic_url = calls[2].args[0]
        topic_body = calls[2].kwargs["json"]
        assert "/api/topic-permissions/%2F/" in topic_url
        assert topic_body["exchange"] == "build.events"
        assert topic_body["write"] == ""
        assert topic_body["read"] == f"build\\.{build_id}\\..*"

    @pytest.mark.asyncio
    async def test_create_scoped_user_failure_on_user_create(self, admin):
        """RabbitMQAdminError raised when user creation fails."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.put = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            with pytest.raises(RabbitMQAdminError, match="Failed to create user"):
                await admin.create_scoped_user(
                    build_id="some-build-id",
                    exchange="ex",
                )

    @pytest.mark.asyncio
    async def test_create_scoped_user_failure_on_permissions(self, admin):
        """RabbitMQAdminError raised when setting permissions fails."""
        success_resp = AsyncMock(spec=httpx.Response)
        success_resp.status_code = 201
        success_resp.text = ""

        fail_resp = AsyncMock(spec=httpx.Response)
        fail_resp.status_code = 403
        fail_resp.text = "Forbidden"

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            # First PUT succeeds (create user), second fails (permissions)
            mock_client.put = AsyncMock(side_effect=[success_resp, fail_resp])
            mock_client_cls.return_value = mock_client

            with pytest.raises(RabbitMQAdminError, match="Failed to set permissions"):
                await admin.create_scoped_user(
                    build_id="build-id",
                    exchange="ex",
                )


class TestCleanupExpiredUsers:
    """Tests for cleanup_expired_users method."""

    @pytest.mark.asyncio
    async def test_cleanup_deletes_expired_users(self, admin):
        """Expired tmp-build users are deleted."""
        past = int((datetime.now(timezone.utc) - timedelta(hours=1)).timestamp())
        future = int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp())

        users_response = AsyncMock(spec=httpx.Response)
        users_response.status_code = 200
        users_response.json = lambda: [
            {"name": "admin", "tags": "administrator"},
            {"name": "expired-user", "tags": f"tmp-build,expires:{past}"},
            {"name": "active-user", "tags": f"tmp-build,expires:{future}"},
            {"name": "normal-user", "tags": "monitoring"},
        ]

        delete_response = AsyncMock(spec=httpx.Response)
        delete_response.status_code = 204
        delete_response.text = ""

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=users_response)
            mock_client.delete = AsyncMock(return_value=delete_response)
            mock_client_cls.return_value = mock_client

            count = await admin.cleanup_expired_users()

        assert count == 1
        mock_client.delete.assert_called_once()
        delete_url = mock_client.delete.call_args.args[0]
        assert "expired-user" in delete_url

    @pytest.mark.asyncio
    async def test_cleanup_no_expired_users(self, admin):
        """Returns 0 when no users are expired."""
        future = int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp())

        users_response = AsyncMock(spec=httpx.Response)
        users_response.status_code = 200
        users_response.json = lambda: [
            {"name": "active-user", "tags": f"tmp-build,expires:{future}"},
        ]

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=users_response)
            mock_client_cls.return_value = mock_client

            count = await admin.cleanup_expired_users()

        assert count == 0

    @pytest.mark.asyncio
    async def test_cleanup_handles_missing_expiry(self, admin):
        """Users with tmp-build tag but no parseable expiry are skipped."""
        users_response = AsyncMock(spec=httpx.Response)
        users_response.status_code = 200
        users_response.json = lambda: [
            {"name": "bad-user", "tags": "tmp-build"},
        ]

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=users_response)
            mock_client_cls.return_value = mock_client

            count = await admin.cleanup_expired_users()

        assert count == 0

    @pytest.mark.asyncio
    async def test_cleanup_raises_on_list_failure(self, admin):
        """RabbitMQAdminError raised when listing users fails."""
        users_response = AsyncMock(spec=httpx.Response)
        users_response.status_code = 500
        users_response.text = "Server Error"

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=users_response)
            mock_client_cls.return_value = mock_client

            with pytest.raises(RabbitMQAdminError, match="Failed to list users"):
                await admin.cleanup_expired_users()


class TestDeleteUser:
    """Tests for delete_user method."""

    @pytest.mark.asyncio
    async def test_delete_user_success(self, admin):
        """Successful deletion with new client."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 204
        mock_response.text = ""

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.delete = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            await admin.delete_user("tmp-build-abc12345-xyz789")

        mock_client.delete.assert_called_once_with(
            "http://localhost:15672/api/users/tmp-build-abc12345-xyz789"
        )

    @pytest.mark.asyncio
    async def test_delete_user_with_existing_client(self, admin):
        """Deletion reuses provided client."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.text = ""

        existing_client = AsyncMock()
        existing_client.delete = AsyncMock(return_value=mock_response)

        await admin.delete_user("some-user", client=existing_client)

        existing_client.delete.assert_called_once_with(
            "http://localhost:15672/api/users/some-user"
        )

    @pytest.mark.asyncio
    async def test_delete_user_failure(self, admin):
        """RabbitMQAdminError raised on deletion failure."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.text = "Not Found"

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.delete = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            with pytest.raises(RabbitMQAdminError, match="Failed to delete user"):
                await admin.delete_user("nonexistent-user")


class TestParseExpiryFromTags:
    """Tests for the _parse_expiry_from_tags static method."""

    def test_valid_expiry(self):
        tags = "tmp-build,expires:1748865600"
        result = RabbitMQAdmin._parse_expiry_from_tags(tags)
        assert result == datetime(2025, 6, 2, 12, 0, 0, tzinfo=timezone.utc)

    def test_no_expiry_tag(self):
        tags = "tmp-build,monitoring"
        result = RabbitMQAdmin._parse_expiry_from_tags(tags)
        assert result is None

    def test_invalid_format(self):
        tags = "tmp-build,expires:not-a-number"
        result = RabbitMQAdmin._parse_expiry_from_tags(tags)
        assert result is None

    def test_empty_tags(self):
        result = RabbitMQAdmin._parse_expiry_from_tags("")
        assert result is None

    def test_expiry_with_spaces(self):
        tags = "tmp-build, expires:1748865600"
        result = RabbitMQAdmin._parse_expiry_from_tags(tags)
        assert result == datetime(2025, 6, 2, 12, 0, 0, tzinfo=timezone.utc)


class TestRandomString:
    """Tests for the _random_string static method."""

    def test_correct_length(self):
        result = RabbitMQAdmin._random_string(24)
        assert len(result) == 24

    def test_custom_alphabet(self):
        result = RabbitMQAdmin._random_string(10, alphabet="abc")
        assert len(result) == 10
        assert all(c in "abc" for c in result)

    def test_randomness(self):
        """Two calls should produce different values (probabilistically)."""
        a = RabbitMQAdmin._random_string(24)
        b = RabbitMQAdmin._random_string(24)
        assert a != b
