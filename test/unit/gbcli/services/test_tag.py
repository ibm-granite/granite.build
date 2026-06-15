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

"""Unit tests for gbcli tag service.

Ported from gbcli-upstream test/unit_tests/services/test_tag.py. All collaborators
are mocked — no IBM infrastructure required.

Adapted to the current API: ``get_tags(github_token, resource_type, username=None,
space=None, callback=None)`` takes the user token as its first positional argument
and authenticates with it directly (no ``GBCredentials`` / "not logged in" path), so
those obsolete cases are dropped.
"""

from unittest.mock import patch

import pytest

from gbcli.services.service_tag import artifact_tag_list, build_tag_list, get_tags

pytestmark = pytest.mark.standalone

_TOKEN = "test_token"


class TestGetTags:
    """Test the unified get_tags function"""

    @patch("gbcli.services.service_tag.gb_server_request")
    @patch("gbcli.services.service_tag.resolve_space")
    def test_artifact_tag_list_success(self, mock_resolve_space, mock_gb_request):
        """Test successfully fetching artifact tags"""
        mock_resolve_space.return_value = {"name": "public"}
        mock_gb_request.return_value = ["tag1", "tag2", "tag3"]

        result = get_tags(_TOKEN, "artifacts", space="public")

        assert result == ["tag1", "tag2", "tag3"]
        mock_gb_request.assert_called_once()

    @patch("gbcli.services.service_tag.gb_server_request")
    @patch("gbcli.services.service_tag.resolve_space")
    def test_build_tag_list_success(self, mock_resolve_space, mock_gb_request):
        """Test successfully fetching build tags"""
        mock_resolve_space.return_value = {"name": "public"}
        mock_gb_request.return_value = ["tag1", "tag2"]

        result = get_tags(_TOKEN, "builds", space="public")

        assert result == ["tag1", "tag2"]
        mock_gb_request.assert_called_once()

    @patch("gbcli.services.service_tag.gb_server_request")
    @patch("gbcli.services.service_tag.resolve_space")
    def test_get_tags_no_space(self, mock_resolve_space, mock_gb_request):
        """Test fetching tags without explicit space filter resolves to default"""
        mock_resolve_space.return_value = {"name": "public"}
        mock_gb_request.return_value = ["tag1", "tag2", "tag3"]

        result = get_tags(_TOKEN, "artifacts")

        assert result == ["tag1", "tag2", "tag3"]
        mock_gb_request.assert_called_once()
        call_kwargs = mock_gb_request.call_args[1]
        assert call_kwargs["params"]["space_name"] == "public"

    @patch("gbcli.services.service_tag.gb_server_request")
    @patch("gbcli.services.service_tag.resolve_space")
    def test_get_tags_with_username(self, mock_resolve_space, mock_gb_request):
        """Test fetching tags with username filter"""
        mock_resolve_space.return_value = {"name": "public"}
        mock_gb_request.return_value = ["tag1"]

        result = get_tags(_TOKEN, "builds", username="testuser")

        assert result == ["tag1"]
        call_kwargs = mock_gb_request.call_args[1]
        assert call_kwargs["params"]["username"] == "testuser"

    @patch("gbcli.services.service_tag.resolve_space")
    def test_get_tags_space_not_found(self, mock_resolve_space):
        """Test error when space is not found"""
        mock_resolve_space.return_value = None

        with pytest.raises(Exception, match="Space .* not found"):
            get_tags(_TOKEN, "artifacts", space="nonexistent")

    @patch("gbcli.services.service_tag.resolve_space")
    def test_get_tags_invalid_resource_type(self, mock_resolve_space):
        """Test error with invalid resource type"""
        mock_resolve_space.return_value = {"name": "public"}

        with pytest.raises(ValueError, match="Invalid resource_type"):
            get_tags(_TOKEN, "invalid_type")

    @patch("gbcli.services.service_tag.gb_server_request")
    @patch("gbcli.services.service_tag.resolve_space")
    def test_get_tags_response_as_dict_with_tags_key(
        self, mock_resolve_space, mock_gb_request
    ):
        """Test handling response when it's a dict with 'tags' key"""
        mock_resolve_space.return_value = {"name": "public"}
        mock_gb_request.return_value = {"tags": ["tag1", "tag2"]}

        result = get_tags(_TOKEN, "artifacts")

        assert result == ["tag1", "tag2"]

    @patch("gbcli.services.service_tag.gb_server_request")
    @patch("gbcli.services.service_tag.resolve_space")
    def test_get_tags_empty_response(self, mock_resolve_space, mock_gb_request):
        """Test handling empty response"""
        mock_resolve_space.return_value = {"name": "public"}
        mock_gb_request.return_value = []

        result = get_tags(_TOKEN, "artifacts")

        assert result == []


class TestArtifactTagsWrapper:
    """Test the artifact_tag_list wrapper function"""

    @patch("gbcli.services.service_tag.get_tags")
    def test_artifact_tag_list(self, mock_get_tags):
        """Test artifact_tag_list calls get_tags with correct parameters"""
        mock_get_tags.return_value = ["tag1", "tag2"]

        result = artifact_tag_list(_TOKEN, username="user1", space="public")

        mock_get_tags.assert_called_once_with(
            _TOKEN, "artifacts", "user1", "public", None
        )
        assert result == ["tag1", "tag2"]


class TestBuildTagsWrapper:
    """Test the build_tag_list wrapper function"""

    @patch("gbcli.services.service_tag.get_tags")
    def test_build_tag_list(self, mock_get_tags):
        """Test build_tag_list calls get_tags with correct parameters"""
        mock_get_tags.return_value = ["tag1", "tag2", "tag3"]

        result = build_tag_list(_TOKEN, username="user1", space="public")

        mock_get_tags.assert_called_once_with(_TOKEN, "builds", "user1", "public", None)
        assert result == ["tag1", "tag2", "tag3"]
