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

import base64
import hashlib
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.g4os


# ---------------------------------------------------------------------------
# PKCE helper tests
# ---------------------------------------------------------------------------


class TestPKCEHelpers:
    def test_code_verifier_length(self):
        from gbcli.utils.ibmid_auth import _generate_code_verifier

        verifier = _generate_code_verifier()
        assert 43 <= len(verifier) <= 128

    def test_code_verifier_url_safe(self):
        from gbcli.utils.ibmid_auth import _generate_code_verifier

        verifier = _generate_code_verifier()
        # URL-safe characters only
        allowed = set(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
        )
        assert all(c in allowed for c in verifier)

    def test_code_challenge_s256(self):
        from gbcli.utils.ibmid_auth import (
            _generate_code_challenge,
            _generate_code_verifier,
        )

        verifier = _generate_code_verifier()
        challenge = _generate_code_challenge(verifier)

        # Verify the challenge matches the S256 spec
        expected = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
            .rstrip(b"=")
            .decode("ascii")
        )
        assert challenge == expected

    def test_code_verifier_uniqueness(self):
        from gbcli.utils.ibmid_auth import _generate_code_verifier

        v1 = _generate_code_verifier()
        v2 = _generate_code_verifier()
        assert v1 != v2


# ---------------------------------------------------------------------------
# IBMidOIDCClient tests (proxy-based)
# ---------------------------------------------------------------------------


class TestIBMidOIDCClient:
    def test_client_uses_gbserver_auth_base(self):
        from gbcli.utils.ibmid_auth import IBMidOIDCClient

        client = IBMidOIDCClient(
            gbserver_auth_base="https://gbserver.example.com/api/v1/auth"
        )
        assert client.gbserver_auth_base == "https://gbserver.example.com/api/v1/auth"

    def test_start_auth_code_flow_polls_and_returns_result(self):
        from gbcli.utils.ibmid_auth import IBMidOIDCClient

        client = IBMidOIDCClient(
            gbserver_auth_base="https://gbserver.example.com/api/v1/auth"
        )

        # First poll: pending, second poll: complete
        pending_resp = MagicMock()
        pending_resp.status_code = 200
        pending_resp.json.return_value = {"status": "pending"}
        pending_resp.raise_for_status = MagicMock()

        complete_resp = MagicMock()
        complete_resp.status_code = 200
        complete_resp.json.return_value = {
            "status": "complete",
            "access_token": "at_test",
            "id_token": "idt_test",
            "refresh_token": "rt_test",
            "expires_in": 3600,
            "user_info": {
                "sub": "IBMid-12345",
                "name": "Test User",
                "email": "test@ibm.com",
                "preferred_username": "testuser",
            },
        }
        complete_resp.raise_for_status = MagicMock()

        with patch("gbcli.utils.ibmid_auth.webbrowser.open"):
            with patch(
                "gbcli.utils.ibmid_auth.requests.get",
                side_effect=[pending_resp, complete_resp],
            ):
                with patch("gbcli.utils.ibmid_auth.time.sleep"):
                    result = client.start_auth_code_flow(timeout=10, poll_interval=0)

        assert result.access_token == "at_test"
        assert result.user_info.email == "test@ibm.com"
        assert result.user_info.preferred_username == "testuser"

    def test_start_auth_code_flow_timeout(self):
        from gbcli.utils.ibmid_auth import IBMidOIDCClient

        client = IBMidOIDCClient(
            gbserver_auth_base="https://gbserver.example.com/api/v1/auth"
        )

        pending_resp = MagicMock()
        pending_resp.status_code = 200
        pending_resp.json.return_value = {"status": "pending"}
        pending_resp.raise_for_status = MagicMock()

        with patch("gbcli.utils.ibmid_auth.webbrowser.open"):
            with patch(
                "gbcli.utils.ibmid_auth.requests.get", return_value=pending_resp
            ):
                with patch("gbcli.utils.ibmid_auth.time.sleep"):
                    with patch(
                        "gbcli.utils.ibmid_auth.time.time", side_effect=[0, 0, 200]
                    ):
                        with pytest.raises(Exception, match="timed out"):
                            client.start_auth_code_flow(timeout=10, poll_interval=0)

    def test_start_auth_code_flow_error_status(self):
        from gbcli.utils.ibmid_auth import IBMidOIDCClient

        client = IBMidOIDCClient(
            gbserver_auth_base="https://gbserver.example.com/api/v1/auth"
        )

        error_resp = MagicMock()
        error_resp.status_code = 200
        error_resp.json.return_value = {
            "status": "error",
            "error": "access_denied",
        }
        error_resp.raise_for_status = MagicMock()

        with patch("gbcli.utils.ibmid_auth.webbrowser.open"):
            with patch("gbcli.utils.ibmid_auth.requests.get", return_value=error_resp):
                with patch("gbcli.utils.ibmid_auth.time.sleep"):
                    with pytest.raises(Exception, match="access_denied"):
                        client.start_auth_code_flow(timeout=10, poll_interval=0)

    def test_start_auth_code_flow_404_keeps_polling(self):
        """404 means the session doesn't exist yet (browser hasn't hit
        /authorize); the client should keep polling until timeout."""
        from gbcli.utils.ibmid_auth import IBMidOIDCClient

        client = IBMidOIDCClient(
            gbserver_auth_base="https://gbserver.example.com/api/v1/auth"
        )

        not_found_resp = MagicMock()
        not_found_resp.status_code = 404
        not_found_resp.json.return_value = {
            "detail": "Auth session not found or expired."
        }

        with patch("gbcli.utils.ibmid_auth.webbrowser.open"):
            with patch(
                "gbcli.utils.ibmid_auth.requests.get", return_value=not_found_resp
            ):
                with patch("gbcli.utils.ibmid_auth.time.sleep"):
                    with patch(
                        "gbcli.utils.ibmid_auth.time.time", side_effect=[0, 0, 200]
                    ):
                        with pytest.raises(Exception, match="timed out"):
                            client.start_auth_code_flow(timeout=10, poll_interval=0)

    def test_start_auth_code_flow_pkce_failure(self):
        from gbcli.utils.ibmid_auth import IBMidOIDCClient

        client = IBMidOIDCClient(
            gbserver_auth_base="https://gbserver.example.com/api/v1/auth"
        )

        pkce_fail_resp = MagicMock()
        pkce_fail_resp.status_code = 401
        pkce_fail_resp.json.return_value = {"detail": "PKCE verification failed."}

        with patch("gbcli.utils.ibmid_auth.webbrowser.open"):
            with patch(
                "gbcli.utils.ibmid_auth.requests.get", return_value=pkce_fail_resp
            ):
                with patch("gbcli.utils.ibmid_auth.time.sleep"):
                    with pytest.raises(Exception, match="PKCE"):
                        client.start_auth_code_flow(timeout=10, poll_interval=0)


# ---------------------------------------------------------------------------
# Credential storage tests
# ---------------------------------------------------------------------------


class TestCredentialStorage:
    def test_get_user_token_returns_ibmid_when_default_provider(self, tmp_path):
        """When default_provider is ibmid, get_user_token should return the IBMid id_token."""
        creds_path = tmp_path / "credentials"
        creds_path.write_text(
            '[user]\ndefault_provider = "ibmid"\n\n'
            '[user.ibmid]\naccess_token = "ibmid_access_token"\n'
            'id_token = "ibmid_id_token"\n'
            'login = "user@ibm.com"\nemail = "user@ibm.com"\nexpires_at = 9999999999\n'
        )

        with patch(
            "gbcli.utils.gbcredentials.get_local_gb_config", return_value=str(tmp_path)
        ):
            with patch("gbcli.utils.gbcredentials.is_standalone", return_value=False):
                from gbcli.utils.gbcredentials import get_user_token

                token = get_user_token()

        assert token == "ibmid_id_token"

    def test_get_user_token_exits_when_ibmid_token_expired(self, tmp_path):
        """When default_provider is ibmid and the token has expired, get_user_token should sys.exit."""
        creds_path = tmp_path / "credentials"
        creds_path.write_text(
            '[user]\ndefault_provider = "ibmid"\n\n'
            '[user.ibmid]\naccess_token = "ibmid_access_token"\n'
            'id_token = "ibmid_id_token"\n'
            'login = "user@ibm.com"\nemail = "user@ibm.com"\nexpires_at = 1000000000\n'
        )

        with patch(
            "gbcli.utils.gbcredentials.get_local_gb_config", return_value=str(tmp_path)
        ):
            with patch("gbcli.utils.gbcredentials.is_standalone", return_value=False):
                from gbcli.utils.gbcredentials import get_user_token

                with pytest.raises(SystemExit) as exc_info:
                    get_user_token()

        assert "expired" in str(exc_info.value).lower()

    def test_get_user_token_falls_back_to_github(self, tmp_path):
        """When no default_provider is set, fall back to GitHub token."""
        creds_path = tmp_path / "credentials"
        creds_path.write_text(
            '[user.github]\ntoken = "ghp_test_token"\n'
            'login = "testuser"\nemail = "test@ibm.com"\n'
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch(
            "gbcli.utils.gbcredentials.get_local_gb_config", return_value=str(tmp_path)
        ):
            with patch("gbcli.utils.gbcredentials.is_standalone", return_value=False):
                with patch(
                    "gbcli.utils.gbcredentials.requests.get", return_value=mock_response
                ):
                    from gbcli.utils.gbcredentials import get_user_token

                    token = get_user_token()

        assert token == "ghp_test_token"

    def test_check_ibmid_values(self, tmp_path):
        creds_path = tmp_path / "credentials"
        creds_path.write_text(
            '[user.ibmid]\naccess_token = "at"\nid_token = "idt"\nlogin = "user"\nemail = "u@ibm.com"\n'
        )

        with patch(
            "gbcli.utils.gbcredentials.get_local_gb_config", return_value=str(tmp_path)
        ):
            from gbcli.utils.gbcredentials import GBCredentials

            creds = GBCredentials()
            assert creds.check_ibmid_values() is True

    def test_check_ibmid_values_missing(self, tmp_path):
        creds_path = tmp_path / "credentials"
        creds_path.write_text("[user.github]\ntoken = 'x'\n")

        with patch(
            "gbcli.utils.gbcredentials.get_local_gb_config", return_value=str(tmp_path)
        ):
            from gbcli.utils.gbcredentials import GBCredentials

            creds = GBCredentials()
            assert creds.check_ibmid_values() is False

    def test_get_user_token_returns_gbserver_api_key(self, tmp_path):
        """When default_provider is apikey, get_user_token should return the API key."""
        creds_path = tmp_path / "credentials"
        creds_path.write_text(
            '[user]\ndefault_provider = "apikey"\n\n'
            '[user.gbserver]\napi_key = "gbserver_test_key"\nlogin = "admin"\n'
        )

        with patch(
            "gbcli.utils.gbcredentials.get_local_gb_config", return_value=str(tmp_path)
        ):
            with patch("gbcli.utils.gbcredentials.is_standalone", return_value=False):
                from gbcli.utils.gbcredentials import get_user_token

                token = get_user_token()

        assert token == "gbserver_test_key"


# ---------------------------------------------------------------------------
# CLI command option tests
# ---------------------------------------------------------------------------


class TestAuthLoginCommand:
    def test_sso_and_token_mutually_exclusive(self):
        """--sso and --token should not be used together."""
        from click.testing import CliRunner

        from gbcli.commands.command_auth import login

        runner = CliRunner()
        result = runner.invoke(login, ["--sso", "ibm", "--token", "ghp_test"])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower() or result.exit_code == 1

    def test_sso_and_gbserver_mutually_exclusive(self):
        """--sso and --gbserver should not be used together."""
        from click.testing import CliRunner

        from gbcli.commands.command_auth import login

        runner = CliRunner()
        result = runner.invoke(login, ["--sso", "ibm", "--gbserver"])
        assert result.exit_code != 0

    def test_sso_ibm_invokes_ibmid_login(self):
        """--sso ibm should call the IBMid proxy-based login flow."""
        from click.testing import CliRunner

        from gbcli.commands.command_auth import login

        runner = CliRunner()
        with patch(
            "gbcli.client.GBClient.Auth.login_ibmid", return_value="user@ibm.com"
        ) as mock_login:
            result = runner.invoke(login, ["--sso", "ibm"], input="y\n")

        mock_login.assert_called_once()
        assert mock_login.call_args.kwargs.get("open_browser") is not None
        assert "user@ibm.com" in result.output
        assert "IBMid authentication is successful" in result.output

    def test_sso_bare_defaults_to_ibm(self):
        """--sso without a value should default to ibm."""
        from click.testing import CliRunner

        from gbcli.commands.command_auth import login

        runner = CliRunner()
        with patch(
            "gbcli.client.GBClient.Auth.login_ibmid", return_value="user@ibm.com"
        ) as mock_login:
            result = runner.invoke(login, ["--sso"], input="y\n")

        mock_login.assert_called_once()
        assert mock_login.call_args.kwargs.get("open_browser") is not None
        assert "IBMid authentication is successful" in result.output


# ---------------------------------------------------------------------------
# auth provider subcommand tests
# ---------------------------------------------------------------------------


class TestAuthProviderCommand:
    def test_provider_shows_current_github(self, tmp_path):
        """auth provider (no options) shows the current default provider."""
        from click.testing import CliRunner

        from gbcli.commands.command_auth import provider

        creds_path = tmp_path / "credentials"
        creds_path.write_text(
            '[user]\ndefault_provider = "github"\n\n'
            '[user.github]\ntoken = "ghp_x"\nlogin = "testuser"\nemail = "t@ibm.com"\n'
        )

        runner = CliRunner()
        with patch("gbcli.commands.command_auth.is_standalone", return_value=False):
            with patch("gbcli.commands.command_auth.GBCredentials") as MockCreds:
                instance = MockCreds.return_value
                instance.get.side_effect = lambda key, section="default": {
                    ("default_provider", "user"): "github",
                    ("login", "user.github"): "testuser",
                }.get((key, section))
                result = runner.invoke(provider)

        assert result.exit_code == 0
        assert "github" in result.output
        assert "testuser" in result.output

    def test_provider_shows_current_sso(self, tmp_path):
        """auth provider shows 'sso' when internal value is 'ibmid'."""
        from click.testing import CliRunner

        from gbcli.commands.command_auth import provider

        runner = CliRunner()
        with patch("gbcli.commands.command_auth.is_standalone", return_value=False):
            with patch("gbcli.commands.command_auth.GBCredentials") as MockCreds:
                instance = MockCreds.return_value
                instance.get.side_effect = lambda key, section="default": {
                    ("default_provider", "user"): "ibmid",
                    ("login", "user.ibmid"): "user@ibm.com",
                }.get((key, section))
                result = runner.invoke(provider)

        assert result.exit_code == 0
        assert "sso" in result.output
        assert "user@ibm.com" in result.output

    def test_provider_set_sso(self):
        """auth provider --set sso stores 'ibmid' as default_provider."""
        from click.testing import CliRunner

        from gbcli.commands.command_auth import provider

        runner = CliRunner()
        with patch("gbcli.commands.command_auth.GBCredentials") as MockCreds:
            instance = MockCreds.return_value
            instance.check_ibmid_values.return_value = True
            result = runner.invoke(provider, ["--set", "sso"])

        assert result.exit_code == 0
        instance.set.assert_called_with("default_provider", "ibmid", section="user")
        instance.save.assert_called_once()
        assert "sso" in result.output

    def test_provider_set_github(self):
        """auth provider --set github stores 'github' as default_provider."""
        from click.testing import CliRunner

        from gbcli.commands.command_auth import provider

        runner = CliRunner()
        with patch("gbcli.commands.command_auth.GBCredentials") as MockCreds:
            instance = MockCreds.return_value
            instance.get.side_effect = lambda key, section="default": {
                ("token", "user.github"): "ghp_x",
                ("login", "user.github"): "testuser",
                ("email", "user.github"): "t@ibm.com",
            }.get((key, section))
            result = runner.invoke(provider, ["--set", "github"])

        assert result.exit_code == 0
        instance.set.assert_called_with("default_provider", "github", section="user")
        instance.save.assert_called_once()

    def test_provider_set_gbserver(self):
        """auth provider --set gbserver stores 'apikey' as default_provider."""
        from click.testing import CliRunner

        from gbcli.commands.command_auth import provider

        runner = CliRunner()
        with patch("gbcli.commands.command_auth.GBCredentials") as MockCreds:
            instance = MockCreds.return_value
            instance.check_gbserver_values.return_value = True
            result = runner.invoke(provider, ["--set", "gbserver"])

        assert result.exit_code == 0
        instance.set.assert_called_with("default_provider", "apikey", section="user")
        instance.save.assert_called_once()

    def test_provider_set_apikey(self):
        """auth provider --set apikey stores 'apikey' as default_provider."""
        from click.testing import CliRunner

        from gbcli.commands.command_auth import provider

        runner = CliRunner()
        with patch("gbcli.commands.command_auth.GBCredentials") as MockCreds:
            instance = MockCreds.return_value
            instance.check_gbserver_values.return_value = True
            result = runner.invoke(provider, ["--set", "apikey"])

        assert result.exit_code == 0
        instance.set.assert_called_with("default_provider", "apikey", section="user")
        instance.save.assert_called_once()

    def test_provider_set_ibmid_synonym(self):
        """auth provider --set ibmid stores 'ibmid' as default_provider (synonym for sso)."""
        from click.testing import CliRunner

        from gbcli.commands.command_auth import provider

        runner = CliRunner()
        with patch("gbcli.commands.command_auth.GBCredentials") as MockCreds:
            instance = MockCreds.return_value
            instance.check_ibmid_values.return_value = True
            result = runner.invoke(provider, ["--set", "ibmid"])

        assert result.exit_code == 0
        instance.set.assert_called_with("default_provider", "ibmid", section="user")
        instance.save.assert_called_once()

    def test_provider_set_sso_missing_credentials(self):
        """auth provider --set sso fails when no IBMid credentials exist."""
        from click.testing import CliRunner

        from gbcli.commands.command_auth import provider

        runner = CliRunner()
        with patch("gbcli.commands.command_auth.GBCredentials") as MockCreds:
            instance = MockCreds.return_value
            instance.check_ibmid_values.return_value = False
            result = runner.invoke(provider, ["--set", "sso"])

        assert result.exit_code != 0
        assert "No IBMid credentials" in result.output or "No IBMid credentials" in (
            result.output + (result.stderr if hasattr(result, "stderr") else "")
        )
