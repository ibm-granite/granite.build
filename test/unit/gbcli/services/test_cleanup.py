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

"""Unit tests for gbcli cleanup service.

Ported from gbcli-upstream test/unit_tests/services/test_cleanup.py. Filesystem and
GitHub collaborators are mocked — no IBM infrastructure required.

The path-resolution helpers read ``GB_CONFIG`` live via ``get_local_gb_config()``.
Outside CLI invocation ``configureGBWorkingEnv()`` never runs, so we point
``GB_CONFIG`` at a throwaway directory to avoid touching ``~/.gbcli`` and to keep
``get_local_gb_config()`` from raising ``KeyError``.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import pytest

from gbcli.services.service_cleanup import (
    remove_config,
    remove_credentials,
    remove_user_fork_from_default,
)
from gbcli.utils.cli_config import get_local_gb_config

pytestmark = pytest.mark.standalone


@pytest.fixture(autouse=True)
def isolated_gb_config(monkeypatch, tmp_path):
    """Redirect GB_CONFIG to a throwaway dir so config/credential paths stay isolated."""
    config_dir = tmp_path / "gbcli"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("GB_CONFIG", str(config_dir))
    return config_dir


class TestCleanup(unittest.TestCase):
    @patch("os.remove")
    @patch("os.path.exists")
    def test_remove_config(self, mock_path_exists, mock_remove):
        mock_path_exists.return_value = True

        returned_path = remove_config()

        mock_remove.assert_called_once_with(returned_path)
        self.assertEqual(
            returned_path,
            os.path.abspath(os.path.join(get_local_gb_config(), "config")),
        )

    @patch("os.remove")
    @patch("os.path.exists")
    def test_remove_credentials(self, mock_path_exists, mock_remove):
        mock_path_exists.return_value = True

        returned_path = remove_credentials()

        mock_remove.assert_called_once_with(returned_path)
        self.assertEqual(
            returned_path,
            os.path.abspath(os.path.join(get_local_gb_config(), "credentials")),
        )

    @patch("gbcli.services.service_cleanup.get_forks")
    @patch("gbcli.services.service_cleanup.resolve_space")
    @patch("gbcli.services.service_cleanup.get_user")
    def test_remove_user_fork_from_default(
        self,
        mock_get_user,
        mock_resolve_space,
        mock_get_forks,
    ):
        mock_get_user.return_value = MagicMock(login="login")
        mock_resolve_space.return_value = {
            "git_repo_uri": "https://github.ibm.com/granite-dot-build/test-space"
        }
        mock_get_forks.return_value = (
            [
                {"owner": {"login": "login"}, "full_name": "full_name"},
            ],
            None,
        )

        fork_name = remove_user_fork_from_default("fake-token")

        self.assertEqual(
            fork_name,
            "full_name",
        )
