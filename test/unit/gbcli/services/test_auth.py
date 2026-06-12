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

"""Unit tests for gbcli auth service.

Ported from gbcli-upstream test/unit_tests/services/test_auth.py. ``get_user`` is
mocked — no IBM infrastructure or network required.

``gh_login`` persists to the GB_CONFIG credentials file via ``GBCredentials``. We
point ``GB_CONFIG`` at a throwaway directory so the test starts from an empty store
and never touches the developer's real ``~/.gbcli/credentials``.
"""

from unittest.mock import MagicMock, patch

import pytest

from gbcli.services.service_auth import gh_login
from gbcli.utils.gbcredentials import GBCredentials
from gbcommon.types.constants import get_gh_credentials_section

pytestmark = pytest.mark.standalone


@pytest.fixture(autouse=True)
def isolated_gb_config(monkeypatch, tmp_path):
    """Redirect GB_CONFIG to a throwaway dir so credentials writes stay isolated."""
    config_dir = tmp_path / "gbcli"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("GB_CONFIG", str(config_dir))
    return config_dir


class TestAuth:
    @patch("gbcli.services.service_auth.get_user")
    def test_gh_login(self, mock_get_user):
        """
        Test login to GitHub
        """
        mock_token_obj = MagicMock()
        mock_token_obj.access_token = ["token"]
        mock_user_obj = MagicMock()
        mock_user_obj.login = "user"
        mock_user_obj.email = "user@example.com"
        mock_get_user.return_value = mock_user_obj

        gh_section = get_gh_credentials_section()

        gh_login(mock_token_obj.access_token[0])

        credentials = GBCredentials()
        mock_get_user.assert_called_once_with(token="token")
        assert "token" == credentials.get("token", section=gh_section)
        assert mock_user_obj.login == credentials.get("login", section=gh_section)
        assert mock_user_obj.email == credentials.get("email", section=gh_section)
