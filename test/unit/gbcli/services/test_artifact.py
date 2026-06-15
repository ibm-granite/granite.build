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

"""Unit tests for gbcli artifact service status updates.

Ported from gbcli-upstream test/unit_tests/services/test_artifact.py. All
collaborators are mocked — no IBM infrastructure required.

Adapted to the current API: ``update_artifact(github_token, artifact_id, ...)`` takes
the user token as its first positional argument and authenticates via
``get_user(github_token).login`` (no ``GBCredentials``); ``user_is_space_admin`` is
called as ``(github_token, space_name, callback)``.
"""

from unittest.mock import MagicMock, patch

import pytest

from gbcli.services.service_artifact import update_artifact

pytestmark = pytest.mark.standalone

_TOKEN = "test_token"


def _logged_in_user():
    """A get_user() return value for a logged-in user."""
    return MagicMock(login="test_user")


class TestUpdateArtifactStatus:
    """Test artifact status update with admin permission enforcement"""

    @patch("gbcli.services.service_artifact.update_artifact_gserver")
    @patch("gbcli.services.service_artifact.user_is_space_admin")
    @patch("gbcli.services.service_artifact.get_artifact")
    @patch("gbcli.services.service_artifact.get_user")
    def test_admin_can_update_status(
        self, mock_get_user, mock_get_artifact, mock_is_admin, mock_update_gserver
    ):
        """Test that admin user can successfully update artifact status"""
        mock_get_user.return_value = _logged_in_user()
        mock_get_artifact.return_value = {
            "uuid": "test-uuid",
            "space_name": "test-space",
        }
        mock_is_admin.return_value = True
        mock_update_gserver.return_value = {
            "uuid": "test-uuid",
            "status": "cancelled",
        }

        result = update_artifact(
            _TOKEN,
            artifact_id="test-uuid",
            status="cancelled",
            isUpdate=True,
        )

        assert result == {"uuid": "test-uuid", "status": "cancelled"}
        mock_get_artifact.assert_called_once()
        mock_is_admin.assert_called_once_with(_TOKEN, "test-space", None)
        mock_update_gserver.assert_called_once()

    @patch("gbcli.services.service_artifact.update_artifact_gserver")
    @patch("gbcli.services.service_artifact.user_is_space_admin")
    @patch("gbcli.services.service_artifact.get_artifact")
    @patch("gbcli.services.service_artifact.get_user")
    def test_non_admin_cannot_update_status(
        self, mock_get_user, mock_get_artifact, mock_is_admin, mock_update_gserver
    ):
        """Test that non-admin user cannot update artifact status"""
        mock_get_user.return_value = _logged_in_user()
        mock_get_artifact.return_value = {
            "uuid": "test-uuid",
            "space_name": "test-space",
        }
        mock_is_admin.return_value = False

        callback_called = False
        callback_args = {}

        def mock_callback(callback_event, **kwargs):
            nonlocal callback_called, callback_args
            callback_called = True
            callback_args = kwargs.get("callback_args", {})

        result = update_artifact(
            _TOKEN,
            artifact_id="test-uuid",
            status="pending",
            isUpdate=True,
            callback=mock_callback,
        )

        assert result is None
        assert callback_called
        assert "admin" in callback_args["reason"].lower()
        mock_update_gserver.assert_not_called()

    @patch("gbcli.services.service_artifact.update_artifact_gserver")
    @patch("gbcli.services.service_artifact.user_is_space_admin")
    @patch("gbcli.services.service_artifact.get_artifact")
    @patch("gbcli.services.service_artifact.get_user")
    def test_non_admin_can_update_description_without_status(
        self, mock_get_user, mock_get_artifact, mock_is_admin, mock_update_gserver
    ):
        """Test that non-admin can update description without status"""
        mock_get_user.return_value = _logged_in_user()
        mock_update_gserver.return_value = {
            "uuid": "test-uuid",
            "description": "Updated description",
        }

        result = update_artifact(
            _TOKEN,
            artifact_id="test-uuid",
            description="Updated description",
        )

        assert result == {"uuid": "test-uuid", "description": "Updated description"}
        # get_artifact should not be called when status is not provided
        mock_get_artifact.assert_not_called()
        # is_admin should not be called when status is not provided
        mock_is_admin.assert_not_called()
        mock_update_gserver.assert_called_once()

    @patch("gbcli.services.service_artifact.update_artifact_gserver")
    @patch("gbcli.services.service_artifact.user_is_space_admin")
    @patch("gbcli.services.service_artifact.get_artifact")
    @patch("gbcli.services.service_artifact.get_user")
    def test_artifact_not_found_when_updating_status(
        self, mock_get_user, mock_get_artifact, mock_is_admin, mock_update_gserver
    ):
        """Test error when artifact not found during status update"""
        mock_get_user.return_value = _logged_in_user()
        mock_get_artifact.return_value = None

        callback_called = False
        callback_args = {}

        def mock_callback(callback_event, **kwargs):
            nonlocal callback_called, callback_args
            callback_called = True
            callback_args = kwargs.get("callback_args", {})

        result = update_artifact(
            _TOKEN,
            artifact_id="nonexistent-uuid",
            status="cancelled",
            isUpdate=True,
            callback=mock_callback,
        )

        assert result is None
        assert callback_called
        assert "not found" in callback_args["reason"].lower()
        mock_update_gserver.assert_not_called()

    @patch("gbcli.services.service_artifact.update_artifact_gserver")
    @patch("gbcli.services.service_artifact.user_is_space_admin")
    @patch("gbcli.services.service_artifact.get_artifact")
    @patch("gbcli.services.service_artifact.get_user")
    def test_status_update_with_missing_space_name(
        self, mock_get_user, mock_get_artifact, mock_is_admin, mock_update_gserver
    ):
        """Test status update when artifact has missing space_name"""
        mock_get_user.return_value = _logged_in_user()
        # Return artifact without space_name (testing .get() safety)
        mock_get_artifact.return_value = {"uuid": "test-uuid"}
        mock_is_admin.return_value = False

        callback_called = False
        callback_args = {}

        def mock_callback(callback_event, **kwargs):
            nonlocal callback_called, callback_args
            callback_called = True
            callback_args = kwargs.get("callback_args", {})

        result = update_artifact(
            _TOKEN,
            artifact_id="test-uuid",
            status="cancelled",
            isUpdate=True,
            callback=mock_callback,
        )

        assert result is None
        assert callback_called
        # Should call is_admin with None space (safe .get() returns None)
        mock_is_admin.assert_called_once_with(_TOKEN, None, mock_callback)
        mock_update_gserver.assert_not_called()

    @patch("gbcli.services.service_artifact.update_artifact_gserver")
    @patch("gbcli.services.service_artifact.user_is_space_admin")
    @patch("gbcli.services.service_artifact.get_artifact")
    @patch("gbcli.services.service_artifact.get_user")
    def test_admin_update_multiple_status_values(
        self, mock_get_user, mock_get_artifact, mock_is_admin, mock_update_gserver
    ):
        """Test admin can update to all valid status values"""
        mock_get_user.return_value = _logged_in_user()
        mock_get_artifact.return_value = {
            "uuid": "test-uuid",
            "space_name": "test-space",
        }
        mock_is_admin.return_value = True

        valid_statuses = ["pending", "success", "failed", "cancelled"]

        for status in valid_statuses:
            mock_update_gserver.return_value = {
                "uuid": "test-uuid",
                "status": status,
            }

            result = update_artifact(
                _TOKEN,
                artifact_id="test-uuid",
                status=status,
            )

            assert result["status"] == status
            mock_update_gserver.assert_called()

    @patch("gbcli.services.service_artifact.update_artifact_gserver")
    @patch("gbcli.services.service_artifact.get_user")
    def test_empty_login_cannot_update(self, mock_get_user, mock_update_gserver):
        """The ``not username`` half of the auth guard aborts before any server call.

        A valid token that resolves to a user with no login.
        """
        # Simulate not logged in: get_user returns a user with no login
        mock_get_user.return_value = MagicMock(login="")

        with pytest.raises(Exception, match="not logged in"):
            update_artifact(
                _TOKEN,
                artifact_id="test-uuid",
                status="cancelled",
            )

        mock_update_gserver.assert_not_called()

    @patch("gbcli.services.service_artifact.update_artifact_gserver")
    @patch("gbcli.services.service_artifact.get_user")
    def test_empty_token_cannot_update(self, mock_get_user, mock_update_gserver):
        """The ``not github_token`` half of the auth guard aborts before any server call.

        An empty token with a valid resolved login, isolating the token check.
        """
        # Login resolves fine, but the token itself is empty.
        mock_get_user.return_value = MagicMock(login="test_user")

        with pytest.raises(Exception, match="not logged in"):
            update_artifact(
                "",
                artifact_id="test-uuid",
                status="cancelled",
            )

        mock_update_gserver.assert_not_called()
