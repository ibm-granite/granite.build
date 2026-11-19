#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# PKCE helper tests
# ---------------------------------------------------------------------------


class TestPKCEHelpers:
    def test_code_verifier_length(self):
        from gbcli.utils.sso_auth import _generate_code_verifier

        verifier = _generate_code_verifier()
        assert 43 <= len(verifier) <= 128

    def test_code_verifier_url_safe(self):
        from gbcli.utils.sso_auth import _generate_code_verifier

        verifier = _generate_code_verifier()
        # URL-safe characters only
        allowed = set(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
        )
        assert all(c in allowed for c in verifier)

    def test_code_challenge_s256(self):
        from gbcli.utils.sso_auth import (
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
        from gbcli.utils.sso_auth import _generate_code_verifier

        v1 = _generate_code_verifier()
        v2 = _generate_code_verifier()
        assert v1 != v2


# ---------------------------------------------------------------------------
# SSOLoginClient tests (loopback and --remote flows)
# ---------------------------------------------------------------------------


AUTH_BASE = "https://gbserver.example.com/api/v1/auth"

TOKEN_RESPONSE = {
    "access_token": "at_test",
    "id_token": "idt_test",
    "refresh_token": "rt_test",
    "expires_in": 3600,
    "user_info": {
        "sub": "user-123",
        "name": "Test User",
        "email": "test@example.com",
        "preferred_username": "testuser",
    },
}


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _token_ok():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = TOKEN_RESPONSE
    return resp


def _browser_hitting(query_for):
    """An open_browser stand-in that follows the authorize URL like the SSO provider would:
    calls the loopback redirect_uri with ``query_for(params)`` (one query or a
    list, hit in order) on a thread."""
    import threading
    import urllib.parse
    import urllib.request

    calls = {}

    def open_browser(url):
        params = {
            k: v[0]
            for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(url).query).items()
        }
        calls["params"] = params
        queries = query_for(params)
        if isinstance(queries, dict):
            queries = [queries]

        def hit():
            pages = calls.setdefault("pages", [])
            for q in queries:
                target = f"{params['redirect_uri']}?{urllib.parse.urlencode(q)}"
                pages.append(urllib.request.urlopen(target, timeout=5).read().decode())
            calls["page"] = pages[-1]

        t = threading.Thread(target=hit, daemon=True)
        t.start()
        calls["thread"] = t

    return open_browser, calls


class TestSSOLoginClient:
    def test_client_uses_gbserver_auth_base(self):
        from gbcli.utils.sso_auth import SSOLoginClient

        client = SSOLoginClient(gbserver_auth_base=AUTH_BASE)
        assert client.gbserver_auth_base == AUTH_BASE

    def test_loopback_receives_redirect_and_redeems(self):
        from gbcli.utils.sso_auth import SSOLoginClient, _generate_code_challenge

        port = _free_port()
        open_browser, calls = _browser_hitting(
            lambda p: {"state": p["state"], "code": "auth-code-1"}
        )
        with patch(
            "gbcli.utils.sso_auth.requests.post", return_value=_token_ok()
        ) as post:
            result = SSOLoginClient(AUTH_BASE).login_loopback(
                open_browser=open_browser, port=port, timeout=10
            )
        calls["thread"].join(5)

        params = calls["params"]
        assert params["redirect_uri"] == f"http://127.0.0.1:{port}/callback"
        assert params["code_challenge_method"] == "S256"
        assert "Sign-in received" in calls["page"]

        assert post.call_args.args[0] == f"{AUTH_BASE}/oidc/token"
        sent = post.call_args.kwargs["json"]
        assert sent["code"] == "auth-code-1"
        assert sent["redirect_uri"] == params["redirect_uri"]
        # The verifier sent to /oidc/token matches the challenge sent to the provider.
        assert (
            _generate_code_challenge(sent["code_verifier"]) == params["code_challenge"]
        )

        assert result.access_token == "at_test"
        assert result.user_info.email == "test@example.com"

    def test_loopback_ignores_other_state_and_keeps_waiting(self):
        """A stale tab (or another page) must not abort or complete the login."""
        from gbcli.utils.sso_auth import SSOLoginClient

        open_browser, calls = _browser_hitting(
            lambda p: [
                {"state": "stale-state", "code": "stale-code"},
                {"state": "ünïcode", "code": "x"},
                {"state": p["state"], "code": "auth-code-1"},
            ]
        )
        with patch(
            "gbcli.utils.sso_auth.requests.post", return_value=_token_ok()
        ) as post:
            result = SSOLoginClient(AUTH_BASE).login_loopback(
                open_browser=open_browser, port=_free_port(), timeout=10
            )
        calls["thread"].join(5)

        assert "different login attempt" in calls["pages"][0]
        assert "different login attempt" in calls["pages"][1]
        post.assert_called_once()
        assert post.call_args.kwargs["json"]["code"] == "auth-code-1"
        assert result.access_token == "at_test"

    def test_loopback_reports_provider_error(self):
        from gbcli.utils.sso_auth import SSOLoginClient

        open_browser, _ = _browser_hitting(
            lambda p: {
                "state": p["state"],
                "error": "access_denied",
                "error_description": "User denied access",
            }
        )
        with patch("gbcli.utils.sso_auth.requests.post") as post:
            with pytest.raises(Exception, match="User denied access"):
                SSOLoginClient(AUTH_BASE).login_loopback(
                    open_browser=open_browser, port=_free_port(), timeout=10
                )
        post.assert_not_called()

    def test_loopback_timeout_suggests_remote(self):
        from gbcli.utils.sso_auth import SSOLoginClient

        with pytest.raises(Exception, match="--remote"):
            SSOLoginClient(AUTH_BASE).login_loopback(
                open_browser=lambda url: None, port=_free_port(), timeout=1
            )

    def test_loopback_port_reusable_right_after_a_login(self):
        """A quick retry must not hit TIME_WAIT 'Address already in use'."""
        from gbcli.utils.sso_auth import SSOLoginClient

        port = _free_port()
        for _ in range(2):
            open_browser, calls = _browser_hitting(
                lambda p: {"state": p["state"], "code": "c"}
            )
            with patch("gbcli.utils.sso_auth.requests.post", return_value=_token_ok()):
                SSOLoginClient(AUTH_BASE).login_loopback(
                    open_browser=open_browser, port=port, timeout=10
                )
            calls["thread"].join(5)

    def test_loopback_port_in_use_suggests_remote(self):
        import socket

        from gbcli.utils.sso_auth import SSOLoginClient

        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen()
            port = busy.getsockname()[1]
            with pytest.raises(Exception, match="--remote"):
                SSOLoginClient(AUTH_BASE).login_loopback(
                    open_browser=lambda url: None, port=port, timeout=1
                )

    def test_remote_decodes_pasted_code_and_redeems(self):
        from gbcli.utils.sso_auth import SSOLoginClient, _generate_code_challenge
        from gbserver.api.auth_routes import encode_remote_code

        seen = {}

        def show_url(url):
            import urllib.parse

            seen["params"] = {
                k: v[0]
                for k, v in urllib.parse.parse_qs(
                    urllib.parse.urlparse(url).query
                ).items()
            }

        def read_code():
            # As printed by gbserver's remote-callback page, with stray whitespace.
            return (
                "  " + encode_remote_code(seen["params"]["state"], "auth-code-r") + "\n"
            )

        with patch(
            "gbcli.utils.sso_auth.requests.post", return_value=_token_ok()
        ) as post:
            result = SSOLoginClient(AUTH_BASE).login_remote(
                show_url=show_url, read_code=read_code
            )

        # No redirect_uri: gbserver uses its own remote-callback URL.
        assert "redirect_uri" not in seen["params"]
        sent = post.call_args.kwargs["json"]
        assert "redirect_uri" not in sent
        assert sent["code"] == "auth-code-r"
        assert (
            _generate_code_challenge(sent["code_verifier"])
            == seen["params"]["code_challenge"]
        )
        assert result.id_token == "idt_test"

    def test_remote_rejects_code_from_another_login(self):
        from gbcli.utils.sso_auth import SSOLoginClient
        from gbserver.api.auth_routes import encode_remote_code

        with patch("gbcli.utils.sso_auth.requests.post") as post:
            with pytest.raises(ValueError, match="different login"):
                SSOLoginClient(AUTH_BASE).login_remote(
                    show_url=lambda url: None,
                    read_code=lambda: encode_remote_code("other-state", "code"),
                    on_bad_code=lambda msg: None,
                )
        post.assert_not_called()

    def test_remote_reprompts_after_bad_paste(self):
        from gbcli.utils.sso_auth import SSOLoginClient
        from gbserver.api.auth_routes import encode_remote_code

        seen, bad = {}, []
        pastes = iter(["garbled!!", None])

        def show_url(url):
            import urllib.parse

            seen["state"] = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)[
                "state"
            ][0]

        def read_code():
            p = next(pastes)
            return p if p is not None else encode_remote_code(seen["state"], "c2")

        with patch(
            "gbcli.utils.sso_auth.requests.post", return_value=_token_ok()
        ) as post:
            SSOLoginClient(AUTH_BASE).login_remote(
                show_url=show_url, read_code=read_code, on_bad_code=bad.append
            )

        assert len(bad) == 1 and "not valid" in bad[0]
        assert post.call_args.kwargs["json"]["code"] == "c2"

    def test_remote_non_ascii_state_is_a_clean_error(self):
        from gbcli.utils.sso_auth import decode_remote_code
        from gbserver.api.auth_routes import encode_remote_code

        with pytest.raises(ValueError, match="different login"):
            decode_remote_code(encode_remote_code("ünïcode", "c"), "state-1")

    def test_remote_rejects_garbled_code(self):
        from gbcli.utils.sso_auth import SSOLoginClient

        with pytest.raises(ValueError, match="not valid"):
            SSOLoginClient(AUTH_BASE).login_remote(
                show_url=lambda url: None,
                read_code=lambda: "not-a-code!!",
                on_bad_code=lambda msg: None,
            )

    def test_redeem_surfaces_server_error(self):
        from gbcli.utils.sso_auth import SSOLoginClient

        resp = MagicMock()
        resp.status_code = 400
        resp.json.return_value = {"detail": "Token exchange failed: invalid_grant"}
        with patch("gbcli.utils.sso_auth.requests.post", return_value=resp):
            with pytest.raises(Exception, match="invalid_grant"):
                SSOLoginClient(AUTH_BASE)._redeem("c", "v", None)


# ---------------------------------------------------------------------------
# Credential storage tests
# ---------------------------------------------------------------------------


class TestCredentialStorage:
    def test_get_user_token_returns_sso_token_when_default_provider(self, tmp_path):
        """With the SSO default_provider, get_user_token returns the SSO id_token."""
        creds_path = tmp_path / "credentials"
        creds_path.write_text(
            '[user]\ndefault_provider = "ibmid"\n\n'
            '[user.ibmid]\naccess_token = "sso_access_token"\n'
            'id_token = "sso_id_token"\n'
            'login = "user@example.com"\nemail = "user@example.com"\nexpires_at = 9999999999\n'
        )

        with patch(
            "gbcli.utils.gbcredentials.get_local_gb_config", return_value=str(tmp_path)
        ):
            with patch("gbcli.utils.gbcredentials.is_standalone", return_value=False):
                from gbcli.utils.gbcredentials import get_user_token

                token = get_user_token()

        assert token == "sso_id_token"

    def test_get_user_token_exits_when_sso_token_expired(self, tmp_path):
        """With the SSO default_provider and an expired token, get_user_token exits."""
        creds_path = tmp_path / "credentials"
        creds_path.write_text(
            '[user]\ndefault_provider = "ibmid"\n\n'
            '[user.ibmid]\naccess_token = "sso_access_token"\n'
            'id_token = "sso_id_token"\n'
            'login = "user@example.com"\nemail = "user@example.com"\nexpires_at = 1000000000\n'
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
        """When no default_provider is set, fall back to GitHub token.

        The token is not pre-validated against GitHub (gbserver authenticates it),
        so no network call is made.
        """
        creds_path = tmp_path / "credentials"
        creds_path.write_text(
            '[user.github]\ntoken = "ghp_test_token"\n'
            'login = "testuser"\nemail = "test@example.com"\n'
        )

        with patch(
            "gbcli.utils.gbcredentials.get_local_gb_config", return_value=str(tmp_path)
        ):
            with patch("gbcli.utils.gbcredentials.is_standalone", return_value=False):
                with patch(
                    "requests.get", side_effect=AssertionError("unexpected network")
                ):
                    from gbcli.utils.gbcredentials import get_user_token

                    token = get_user_token()

        assert token == "ghp_test_token"

    def test_check_sso_values(self, tmp_path):
        creds_path = tmp_path / "credentials"
        creds_path.write_text(
            '[user.ibmid]\naccess_token = "at"\nid_token = "idt"\nlogin = "user"\nemail = "u@example.com"\n'
        )

        with patch(
            "gbcli.utils.gbcredentials.get_local_gb_config", return_value=str(tmp_path)
        ):
            from gbcli.utils.gbcredentials import GBCredentials

            creds = GBCredentials()
            assert creds.check_sso_values() is True

    def test_check_sso_values_missing(self, tmp_path):
        creds_path = tmp_path / "credentials"
        creds_path.write_text("[user.github]\ntoken = 'x'\n")

        with patch(
            "gbcli.utils.gbcredentials.get_local_gb_config", return_value=str(tmp_path)
        ):
            from gbcli.utils.gbcredentials import GBCredentials

            creds = GBCredentials()
            assert creds.check_sso_values() is False

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
    @pytest.fixture(autouse=True)
    def _local_browser(self):
        """Pretend a local browser can take the redirect (CI has no display)."""
        with patch(
            "gbcli.commands.command_auth.local_browser_unavailable_reason",
            return_value=None,
        ):
            yield

    def test_sso_without_local_browser_fails_fast(self):
        from click.testing import CliRunner

        from gbcli.commands.command_auth import login

        with (
            patch(
                "gbcli.commands.command_auth.local_browser_unavailable_reason",
                return_value="this is an SSH session",
            ),
            patch("gbcli.client.GBClient.Auth.login_sso") as mock_login,
        ):
            result = CliRunner().invoke(login, ["--sso"])

        assert result.exit_code == 1
        assert "SSH session" in result.output and "--remote" in result.output
        mock_login.assert_not_called()

    def test_sso_remote_skips_the_local_browser_check(self):
        from click.testing import CliRunner

        from gbcli.commands.command_auth import login

        with (
            patch(
                "gbcli.commands.command_auth.local_browser_unavailable_reason",
                return_value="this is an SSH session",
            ),
            patch(
                "gbcli.client.GBClient.Auth.login_sso", return_value="user@example.com"
            ),
        ):
            result = CliRunner().invoke(login, ["--sso", "--remote"])
        assert result.exit_code == 0, result.output

    def test_abort_at_paste_prompt_is_not_a_login_failure(self):
        from click.testing import CliRunner

        from gbcli.commands.command_auth import login

        def fake_login(open_browser, remote, read_code, **_):
            return read_code()  # no input: click aborts

        with patch("gbcli.client.GBClient.Auth.login_sso", side_effect=fake_login):
            result = CliRunner().invoke(login, ["--sso", "--remote"], input="")

        assert result.exit_code == 1
        assert "Aborted" in result.output
        assert "authentication failed" not in result.output

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

    def test_sso_ibm_invokes_sso_login(self):
        """--sso ibm should call the proxy-based SSO login flow."""
        from click.testing import CliRunner

        from gbcli.commands.command_auth import login

        runner = CliRunner()
        with patch(
            "gbcli.client.GBClient.Auth.login_sso", return_value="user@example.com"
        ) as mock_login:
            result = runner.invoke(login, ["--sso", "ibm"], input="y\n")

        mock_login.assert_called_once()
        assert mock_login.call_args.kwargs.get("open_browser") is not None
        assert "user@example.com" in result.output
        assert "SSO authentication is successful" in result.output

    def test_sso_bare_defaults_to_ibm(self):
        """--sso without a value should default to ibm."""
        from click.testing import CliRunner

        from gbcli.commands.command_auth import login

        runner = CliRunner()
        with patch(
            "gbcli.client.GBClient.Auth.login_sso", return_value="user@example.com"
        ) as mock_login:
            result = runner.invoke(login, ["--sso"], input="y\n")

        mock_login.assert_called_once()
        assert mock_login.call_args.kwargs.get("open_browser") is not None
        assert "SSO authentication is successful" in result.output

    def test_sso_remote_prompts_for_code(self):
        """--sso --remote prints the URL and passes the pasted code through."""
        from click.testing import CliRunner

        from gbcli.commands.command_auth import login

        def fake_login(open_browser, remote, read_code, **_):
            assert remote is True
            open_browser("https://gbserver.example.com/api/v1/auth/oidc/authorize?x=1")
            assert read_code() == "pasted-code"
            return "user@example.com"

        runner = CliRunner()
        with patch(
            "gbcli.client.GBClient.Auth.login_sso", side_effect=fake_login
        ) as mock_login:
            result = runner.invoke(login, ["--sso", "--remote"], input="pasted-code\n")

        assert result.exit_code == 0, result.output
        mock_login.assert_called_once()
        # The URL sits on its own line, with blank lines around it.
        assert (
            "\n\nhttps://gbserver.example.com/api/v1/auth/oidc/authorize?x=1\n\n"
            in result.output
        )
        # The pasted code is read hidden, never echoed.
        assert "pasted-code" not in result.output
        assert "SSO authentication is successful" in result.output

    def test_sso_redirect_port_passed_through(self):
        from click.testing import CliRunner

        from gbcli.commands.command_auth import login

        runner = CliRunner()
        with patch(
            "gbcli.client.GBClient.Auth.login_sso", return_value="user@example.com"
        ) as mock_login:
            result = runner.invoke(
                login, ["--sso", "--redirect-port", "50123"], input="y\n"
            )

        assert result.exit_code == 0, result.output
        assert mock_login.call_args.kwargs["redirect_port"] == 50123

    @pytest.mark.parametrize(
        "args",
        [
            ["--remote"],
            ["--redirect-port", "50123"],
            ["--gbserver", "--remote"],
            ["--sso", "--remote", "--redirect-port", "50123"],
        ],
    )
    def test_remote_and_port_need_sso_loopback(self, args):
        from click.testing import CliRunner

        from gbcli.commands.command_auth import login

        runner = CliRunner()
        with patch("gbcli.client.GBClient.Auth.login_sso") as mock_login:
            result = runner.invoke(login, args)

        assert result.exit_code == 1
        mock_login.assert_not_called()


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
            '[user.github]\ntoken = "ghp_x"\nlogin = "testuser"\nemail = "t@example.com"\n'
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
                    ("login", "user.ibmid"): "user@example.com",
                }.get((key, section))
                result = runner.invoke(provider)

        assert result.exit_code == 0
        assert "sso" in result.output
        assert "user@example.com" in result.output

    def test_provider_set_sso(self):
        """auth provider --set sso stores 'ibmid' as default_provider."""
        from click.testing import CliRunner

        from gbcli.commands.command_auth import provider

        runner = CliRunner()
        with patch("gbcli.commands.command_auth.GBCredentials") as MockCreds:
            instance = MockCreds.return_value
            instance.check_sso_values.return_value = True
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
                ("email", "user.github"): "t@example.com",
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
            instance.check_sso_values.return_value = True
            result = runner.invoke(provider, ["--set", "ibmid"])

        assert result.exit_code == 0
        instance.set.assert_called_with("default_provider", "ibmid", section="user")
        instance.save.assert_called_once()

    def test_provider_set_sso_missing_credentials(self):
        """auth provider --set sso fails when no SSO credentials exist."""
        from click.testing import CliRunner

        from gbcli.commands.command_auth import provider

        runner = CliRunner()
        with patch("gbcli.commands.command_auth.GBCredentials") as MockCreds:
            instance = MockCreds.return_value
            instance.check_sso_values.return_value = False
            result = runner.invoke(provider, ["--set", "sso"])

        assert result.exit_code != 0
        assert "No SSO credentials" in result.output or "No SSO credentials" in (
            result.output + (result.stderr if hasattr(result, "stderr") else "")
        )


# ---------------------------------------------------------------------------
# Local browser detection
# ---------------------------------------------------------------------------


class TestLocalBrowserUnavailableReason:
    @pytest.fixture
    def env(self, monkeypatch):
        for var in (
            "SSH_CONNECTION",
            "SSH_TTY",
            "DISPLAY",
            "WAYLAND_DISPLAY",
            "BROWSER",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr("gbcli.utils.utils.check_runnable_browser", lambda: True)
        return monkeypatch

    def test_ssh_session(self, env):
        from gbcli.utils.utils import local_browser_unavailable_reason

        env.setenv("SSH_CONNECTION", "10.0.0.1 50000 10.0.0.2 22")
        assert "SSH" in local_browser_unavailable_reason()

    def test_linux_without_display(self, env):
        from gbcli.utils.utils import local_browser_unavailable_reason

        env.setattr("gbcli.utils.utils.sys.platform", "linux")
        assert local_browser_unavailable_reason() == "no graphical display"
        env.setenv("BROWSER", "wslview")
        assert local_browser_unavailable_reason() is None

    def test_no_browser_found(self, env):
        from gbcli.utils.utils import local_browser_unavailable_reason

        env.setattr("gbcli.utils.utils.sys.platform", "darwin")
        env.setattr("gbcli.utils.utils.check_runnable_browser", lambda: False)
        assert local_browser_unavailable_reason() == "no web browser found"

    def test_desktop(self, env):
        from gbcli.utils.utils import local_browser_unavailable_reason

        env.setattr("gbcli.utils.utils.sys.platform", "darwin")
        assert local_browser_unavailable_reason() is None
