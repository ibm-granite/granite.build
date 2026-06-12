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

"""Tests for the --format flag in various commands."""

import json
from unittest.mock import ANY, MagicMock, patch

import pytest
from click.testing import CliRunner

from gbcli.commands.command_artifact import cli as artifact_cli
from gbcli.commands.command_secret import cli as secret_cli
from gbcli.commands.command_step import cli as step_cli
from gbcli.commands.command_tag import cli as tag_cli
from gbcli.commands.command_template import cli as template_cli
from gbcli.commands.command_version import cli as version_cli
from gbcli.utils.gbconstants import ARTIFACT_LIST_HEADERS


class TestFormatFlag:
    """Test suite for --format flag across all commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def _mock_token(self):
        """Mock get_user_token to return a dummy token."""
        return "test-token"

    # Tag command tests
    @patch("gbcli.commands.command_tag.get_user_token")
    @patch("gbcli.commands.command_tag.GBClient")
    @patch("gbcli.commands.command_tag.check_current_and_latest_versions")
    def test_tag_list_format_plain(self, mock_check_version, mock_client, mock_token):
        """Test tag list with plain format."""
        mock_token.return_value = self._mock_token()
        mock_check_version.return_value = None
        mock_tag_client = MagicMock()
        mock_tag_client.build_tag_list.return_value = ["tag1", "tag2"]
        mock_client.Tag.return_value = mock_tag_client

        result = self.runner.invoke(tag_cli, ["list", "--builds", "--format", "plain"])

        assert (
            result.exit_code == 0
        ), f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"

    @patch("gbcli.commands.command_tag.get_user_token")
    @patch("gbcli.commands.command_tag.GBClient")
    @patch("gbcli.commands.command_tag.check_current_and_latest_versions")
    def test_tag_list_format_json(self, mock_check_version, mock_client, mock_token):
        """Test tag list with json format."""
        mock_token.return_value = self._mock_token()
        mock_check_version.return_value = None
        mock_tag_client = MagicMock()
        mock_tag_client.build_tag_list.return_value = ["tag1", "tag2"]
        mock_client.Tag.return_value = mock_tag_client

        result = self.runner.invoke(tag_cli, ["list", "--builds", "--format", "json"])

        assert (
            result.exit_code == 0
        ), f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"
        # Check that JSON output is present
        assert "[" in result.output

    @patch("gbcli.commands.command_tag.get_user_token")
    @patch("gbcli.commands.command_tag.GBClient")
    @patch("gbcli.commands.command_tag.check_current_and_latest_versions")
    def test_tag_list_format_pretty(self, mock_check_version, mock_client, mock_token):
        """Test tag list with pretty format."""
        mock_token.return_value = self._mock_token()
        mock_check_version.return_value = None
        mock_tag_client = MagicMock()
        mock_tag_client.build_tag_list.return_value = ["tag1", "tag2"]
        mock_client.Tag.return_value = mock_tag_client

        result = self.runner.invoke(tag_cli, ["list", "--builds", "--format", "pretty"])

        assert (
            result.exit_code == 0
        ), f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"

    @patch("gbcli.commands.command_tag.get_user_token")
    @patch("gbcli.commands.command_tag.GBClient")
    @patch("gbcli.commands.command_tag.check_current_and_latest_versions")
    def test_tag_list_format_default(self, mock_check_version, mock_client, mock_token):
        """Test tag list with default format (plain)."""
        mock_token.return_value = self._mock_token()
        mock_check_version.return_value = None
        mock_tag_client = MagicMock()
        mock_tag_client.build_tag_list.return_value = ["tag1", "tag2"]
        mock_client.Tag.return_value = mock_tag_client

        result = self.runner.invoke(tag_cli, ["list", "--builds"])

        assert (
            result.exit_code == 0
        ), f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"

    @patch("gbcli.commands.command_tag.get_user_token")
    @patch("gbcli.commands.command_tag.check_current_and_latest_versions")
    def test_tag_list_format_invalid(self, mock_check_version, mock_token):
        """Test tag list with invalid format value."""
        mock_token.return_value = self._mock_token()
        mock_check_version.return_value = None

        result = self.runner.invoke(
            tag_cli, ["list", "--builds", "--format", "invalid"]
        )

        assert result.exit_code != 0, "Expected non-zero exit code for invalid format"

    # Template command tests
    @patch("gbcli.commands.command_template.get_user_token")
    @patch("gbcli.commands.command_template.GBClient")
    @patch("gbcli.commands.command_template.check_current_and_latest_versions")
    def test_template_list_format_plain(
        self, mock_check_version, mock_client, mock_token
    ):
        """Test template list with plain format."""
        mock_token.return_value = self._mock_token()
        mock_check_version.return_value = None
        mock_template_client = MagicMock()
        mock_template_client.list.return_value = [
            {"name": "template1", "description": "desc1"},
            {"name": "template2", "description": "desc2"},
        ]
        mock_client.Template.return_value = mock_template_client

        result = self.runner.invoke(template_cli, ["list", "--format", "plain"])

        assert (
            result.exit_code == 0
        ), f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"

    @patch("gbcli.commands.command_template.get_user_token")
    @patch("gbcli.commands.command_template.GBClient")
    @patch("gbcli.commands.command_template.check_current_and_latest_versions")
    def test_template_list_format_json(
        self, mock_check_version, mock_client, mock_token
    ):
        """Test template list with json format."""
        mock_token.return_value = self._mock_token()
        mock_check_version.return_value = None
        mock_template_client = MagicMock()
        mock_template_client.list.return_value = [
            {"name": "template1", "description": "desc1"}
        ]
        mock_client.Template.return_value = mock_template_client

        result = self.runner.invoke(template_cli, ["list", "--format", "json"])

        assert (
            result.exit_code == 0
        ), f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"

    @patch("gbcli.commands.command_template.get_user_token")
    @patch("gbcli.commands.command_template.GBClient")
    @patch("gbcli.commands.command_template.check_current_and_latest_versions")
    def test_template_describe_format_plain(
        self, mock_check_version, mock_client, mock_token
    ):
        """Test template describe with plain format."""
        mock_token.return_value = self._mock_token()
        mock_check_version.return_value = None
        mock_template_client = MagicMock()
        mock_template_client.describe.return_value = {
            "name": "test-template",
            "content": "test content",
        }
        mock_client.Template.return_value = mock_template_client

        result = self.runner.invoke(
            template_cli, ["describe", "test-template", "--format", "plain"]
        )

        assert (
            result.exit_code == 0
        ), f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"

    # Artifact command tests
    @patch("gbcli.commands.command_artifact.get_user_token")
    @patch("gbcli.commands.command_artifact.GBClient")
    @patch("gbcli.commands.command_artifact.check_current_and_latest_versions")
    def test_artifact_list_format_plain(
        self, mock_check_version, mock_client, mock_token
    ):
        """Test artifact list with plain format."""
        mock_token.return_value = self._mock_token()
        mock_check_version.return_value = None
        mock_artifact_client = MagicMock()
        mock_artifact_client.list.return_value = {
            "artifacts": [
                {"uuid": "id1", "uri": "uri1", "tags": []},
                {"uuid": "id2", "uri": "uri2", "tags": []},
            ]
        }
        mock_client.Artifact.return_value = mock_artifact_client

        result = self.runner.invoke(artifact_cli, ["list", "--format", "plain"])

        assert (
            result.exit_code == 0
        ), f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"

    @patch("gbcli.commands.command_artifact.get_user_token")
    @patch("gbcli.commands.command_artifact.GBClient")
    @patch("gbcli.commands.command_artifact.check_current_and_latest_versions")
    def test_artifact_list_format_json(
        self, mock_check_version, mock_client, mock_token
    ):
        """Test artifact list with json format."""
        mock_token.return_value = self._mock_token()
        mock_check_version.return_value = None
        mock_artifact_client = MagicMock()
        mock_artifact_client.list.return_value = {
            "artifacts": [{"uuid": "id1", "uri": "uri1", "tags": []}]
        }
        mock_client.Artifact.return_value = mock_artifact_client

        result = self.runner.invoke(artifact_cli, ["list", "--format", "json"])

        assert (
            result.exit_code == 0
        ), f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"

    def _sample_artifacts(self):
        """A non-empty artifact list that exercises the rendering path."""
        return [
            {
                "uuid": "id1",
                "name": "artifact-one",
                "uri": "hf://datasets/org/one",
                "type": "dataset",
                "tags": [],
                "status": "success",
                "created_by_build_id": "build1",
                "username": "alice",
                "created_at": "2026-06-10T12:00:00Z",
                "description": "first artifact",
                "checksum": "abc123",
                "is_archived": False,
            },
        ]

    @patch("gbcli.commands.command_artifact.get_user_token")
    @patch("gbcli.commands.command_artifact.GBClient")
    @patch("gbcli.commands.command_artifact.check_current_and_latest_versions")
    def test_artifact_list_plain_is_borderless(
        self, mock_check_version, mock_client, mock_token
    ):
        """Default/plain artifact list must stay borderless (pipe-friendly)."""
        mock_token.return_value = self._mock_token()
        mock_check_version.return_value = None
        mock_artifact_client = MagicMock()
        mock_artifact_client.artifact_list.return_value = self._sample_artifacts()
        mock_client.Artifact.return_value = mock_artifact_client

        result = self.runner.invoke(
            artifact_cli, ["list", "--quiet", "--format", "plain"]
        )

        assert (
            result.exit_code == 0
        ), f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"
        # No rich box-drawing characters should appear in borderless output.
        for border_char in ("│", "─", "┃", "━", "┌", "┐", "└", "┘", "├", "┤"):
            assert (
                border_char not in result.output
            ), f"plain output should be borderless but contained {border_char!r}"
        # The default (no --format) must match plain, not the old bordered table.
        default_result = self.runner.invoke(artifact_cli, ["list", "--quiet"])
        for border_char in ("│", "─", "┃", "━"):
            assert border_char not in default_result.output
        # The bordered "Artifacts" title should only appear via --format pretty.
        assert "Artifacts" not in result.output
        assert "Artifacts" not in default_result.output

    @patch("gbcli.commands.command_artifact.get_user_token")
    @patch("gbcli.commands.command_artifact.GBClient")
    @patch("gbcli.commands.command_artifact.check_current_and_latest_versions")
    def test_artifact_list_does_not_mutate_module_headers(
        self, mock_check_version, mock_client, mock_token
    ):
        """Repeated invocations must not mutate the module-level header list."""
        mock_token.return_value = self._mock_token()
        mock_check_version.return_value = None
        mock_artifact_client = MagicMock()
        mock_artifact_client.artifact_list.return_value = self._sample_artifacts()
        mock_client.Artifact.return_value = mock_artifact_client

        original_headers = list(ARTIFACT_LIST_HEADERS)

        # --wide and --show-archived both insert/append into the header copy;
        # if the command mutated the module list, a second run would see extras.
        for _ in range(2):
            result = self.runner.invoke(
                artifact_cli,
                ["list", "--quiet", "--wide", "--show-archived"],
            )
            assert (
                result.exit_code == 0
            ), f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"
            assert (
                ARTIFACT_LIST_HEADERS == original_headers
            ), f"ARTIFACT_LIST_HEADERS was mutated: {ARTIFACT_LIST_HEADERS}"

    # Step command tests
    @patch("gbcli.commands.command_step.get_user_token")
    @patch("gbcli.commands.command_step.GBClient")
    @patch("gbcli.commands.command_step.check_current_and_latest_versions")
    def test_step_list_format_plain(self, mock_check_version, mock_client, mock_token):
        """Test step list with plain format."""
        mock_token.return_value = self._mock_token()
        mock_check_version.return_value = None
        mock_step_client = MagicMock()
        mock_step_client.list.return_value = [
            {"name": "step1", "description": "desc1"},
            {"name": "step2", "description": "desc2"},
        ]
        mock_client.Step.return_value = mock_step_client

        result = self.runner.invoke(step_cli, ["list", "--format", "plain"])

        assert (
            result.exit_code == 0
        ), f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"

    @patch("gbcli.commands.command_step.get_user_token")
    @patch("gbcli.commands.command_step.GBClient")
    @patch("gbcli.commands.command_step.check_current_and_latest_versions")
    def test_step_list_format_json(self, mock_check_version, mock_client, mock_token):
        """Test step list with json format."""
        mock_token.return_value = self._mock_token()
        mock_check_version.return_value = None
        mock_step_client = MagicMock()
        mock_step_client.list.return_value = [{"name": "step1"}]
        mock_client.Step.return_value = mock_step_client

        result = self.runner.invoke(step_cli, ["list", "--format", "json"])

        assert (
            result.exit_code == 0
        ), f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"

    # Version command tests (uses "simple" and "json" format, not "plain")
    @patch("gbcli.commands.command_version.get_user_token")
    @patch("gbcli.commands.command_version.GBClient")
    def test_version_current_format_json(self, mock_client, mock_token):
        """Test version current with json format."""
        mock_token.return_value = self._mock_token()
        mock_version_client = MagicMock()
        mock_version_client.current_version.return_value = "1.0.0"
        mock_client.Version.return_value = mock_version_client

        result = self.runner.invoke(version_cli, ["current", "--format", "json"])

        # Just verify the format parameter is accepted (no format validation error)
        assert "--format" not in result.output or "not one of" not in result.output

    @patch("gbcli.commands.command_version.get_user_token")
    @patch("gbcli.commands.command_version.GBClient")
    def test_version_current_format_simple(self, mock_client, mock_token):
        """Test version current with simple format."""
        mock_token.return_value = self._mock_token()
        mock_version_client = MagicMock()
        mock_version_client.current_version.return_value = "1.0.0"
        mock_client.Version.return_value = mock_version_client

        result = self.runner.invoke(version_cli, ["current", "--format", "simple"])

        # Just verify the format parameter is accepted (no format validation error)
        assert "--format" not in result.output or "not one of" not in result.output

    # Secret command tests
    @patch("gbcli.commands.command_secret.get_user_token")
    @patch("gbcli.commands.command_secret.GBClient")
    @patch("gbcli.commands.command_secret.check_current_and_latest_versions")
    def test_secret_list_format_plain(
        self, mock_check_version, mock_client, mock_token
    ):
        """Test secret list with plain format."""
        mock_token.return_value = self._mock_token()
        mock_check_version.return_value = None
        mock_secret_client = MagicMock()
        mock_secret_client.list.return_value = [
            {"name": "secret1", "type": "apikey"},
            {"name": "secret2", "type": "password"},
        ]
        mock_client.Secret.return_value = mock_secret_client

        result = self.runner.invoke(secret_cli, ["list", "--format", "plain"])

        assert (
            result.exit_code == 0
        ), f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"

    @patch("gbcli.commands.command_secret.get_user_token")
    @patch("gbcli.commands.command_secret.GBClient")
    @patch("gbcli.commands.command_secret.check_current_and_latest_versions")
    def test_secret_list_format_json(self, mock_check_version, mock_client, mock_token):
        """Test secret list with json format accepts format parameter."""
        mock_token.return_value = self._mock_token()
        mock_check_version.return_value = None
        mock_secret_client = MagicMock()
        mock_secret_client.list.return_value = []
        mock_client.Secret.return_value = mock_secret_client

        result = self.runner.invoke(secret_cli, ["list", "--format", "json"])

        # Just verify the format parameter is accepted (no format validation error)
        assert "--format" not in result.output or "not one of" not in result.output
