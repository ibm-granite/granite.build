#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Unit tests for gbcli auth service.

Ported from gbcli-upstream test/unit_tests/services/test_auth.py. The GitHub
``/user`` lookup is mocked — no IBM infrastructure or network required.

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
    @patch("gbcommon.types.gbenvconfig.is_standalone", return_value=False)
    @patch("gbcli.services.service_auth._request_github_user")
    def test_gh_login(self, mock_request_user, _standalone):
        """
        Test login to GitHub
        """
        mock_token_obj = MagicMock()
        mock_token_obj.access_token = ["token"]
        mock_user_obj = MagicMock()
        mock_user_obj.login = "user"
        mock_user_obj.email = "user@example.com"
        mock_request_user.return_value = mock_user_obj

        gh_section = get_gh_credentials_section()

        gh_login(mock_token_obj.access_token[0])

        credentials = GBCredentials()
        mock_request_user.assert_called_once_with("token")
        assert "token" == credentials.get("token", section=gh_section)
        assert mock_user_obj.login == credentials.get("login", section=gh_section)
        assert mock_user_obj.email == credentials.get("email", section=gh_section)

    @patch("gbcommon.types.gbenvconfig.is_standalone", return_value=False)
    @patch("gbcli.utils.gh_auth.requests.get")
    def test_gh_login_after_sso_stores_github_identity(
        self, mock_get, _standalone, isolated_gb_config
    ):
        """A prior SSO login must not leak its identity into the GitHub section."""
        (isolated_gb_config / "credentials").write_text(
            '[user]\ndefault_provider = "ibmid"\n'
            '[user.ibmid]\naccess_token = "a"\nid_token = "i"\n'
            'login = "sso@example.com"\nemail = "sso@example.com"\n'
        )
        mock_get.return_value.json.return_value = {
            "login": "ghuser",
            "id": 1,
            "url": "u",
            "html_url": "h",
            "name": "GH User",
            "email": "gh@example.com",
        }

        assert gh_login("token") == "ghuser"

        gh_section = get_gh_credentials_section()
        credentials = GBCredentials()
        assert credentials.get("login", section=gh_section) == "ghuser"
        assert credentials.get("email", section=gh_section) == "gh@example.com"
        assert credentials.get("default_provider", section="user") == "github"
        assert credentials.get("login", section="user.ibmid") == "sso@example.com"

    @patch("gbcommon.types.gbenvconfig.is_standalone", return_value=True)
    @patch(
        "gbcli.utils.gh_auth.requests.get",
        side_effect=AssertionError("unexpected network"),
    )
    def test_gh_login_standalone_skips_github(self, _get, _standalone, monkeypatch):
        """Standalone stores its synthetic user without calling GitHub."""
        monkeypatch.delenv("GBSERVER_API_USER", raising=False)

        assert gh_login("token") == "standalone"
        assert GBCredentials().get("login", section=get_gh_credentials_section()) == (
            "standalone"
        )

    @patch("gbcommon.types.gbenvconfig.is_standalone", return_value=False)
    @patch("gbcli.utils.gh_auth.requests.get")
    def test_gh_login_private_email(self, mock_get, _standalone):
        """GitHub returns null name/email for a private profile; login still works."""
        mock_get.return_value.json.return_value = {
            "login": "ghuser",
            "id": 1,
            "url": "u",
            "html_url": "h",
            "name": None,
            "email": None,
        }

        assert gh_login("token") == "ghuser"

        credentials = GBCredentials()
        assert credentials.get("email", section=get_gh_credentials_section()) == ""
        assert credentials.check_values()
