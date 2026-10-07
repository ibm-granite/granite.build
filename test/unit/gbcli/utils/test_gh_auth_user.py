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

"""``get_user`` resolves the stored GitHub identity without a /user round-trip,
and the 401 error raised by the gbserver helpers carries a re-login hint."""

from unittest.mock import MagicMock, patch

import pytest

from gbcli.utils import gbserver_errors, gh_auth
from gbcli.utils.gbserver_errors import (
    GBServerAuthError,
    GBServerHTTPError,
    gbserver_http_error,
)

pytestmark = pytest.mark.standalone

_USER_JSON = {
    "login": "fetched",
    "id": 7,
    "url": "u",
    "html_url": "h",
    "name": "Fetched",
    "email": "fetched@ibm.com",
}


@pytest.fixture
def github_creds(tmp_path):
    (tmp_path / "credentials").write_text(
        '[user.github]\ntoken = "stored_tok"\n'
        'login = "stored"\nemail = "stored@ibm.com"\n'
    )
    gh_auth._fetch_github_user.cache_clear()
    with (
        patch(
            "gbcli.utils.gbcredentials.get_local_gb_config", return_value=str(tmp_path)
        ),
        patch("gbcommon.types.gbenvconfig.is_standalone", return_value=False),
        patch(
            "gbcli.utils.gh_auth.get_gh_credentials_section",
            return_value="user.github",
        ),
    ):
        yield
    gh_auth._fetch_github_user.cache_clear()


def _ok_response():
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = _USER_JSON
    return resp


class TestGetUser:
    def test_stored_token_uses_stored_identity(self, github_creds):
        with patch(
            "gbcli.utils.gh_auth.requests.get",
            side_effect=AssertionError("unexpected network"),
        ):
            user = gh_auth.get_user("stored_tok")
        assert user.login == "stored"
        assert user.email == "stored@ibm.com"

    def test_other_token_fetches_once_per_process(self, github_creds):
        with patch(
            "gbcli.utils.gh_auth.requests.get", return_value=_ok_response()
        ) as get:
            first = gh_auth.get_user("new_tok")
            second = gh_auth.get_user("new_tok")
        assert first.login == second.login == "fetched"
        assert get.call_count == 1

    def test_verify_always_asks_github(self, github_creds):
        """`gb auth login` must validate even a token equal to the stored one."""
        with patch(
            "gbcli.utils.gh_auth.requests.get", return_value=_ok_response()
        ) as get:
            gh_auth.get_user("stored_tok", verify=True)
            gh_auth.get_user("stored_tok", verify=True)
        assert get.call_count == 2


class TestGBServerErrors:
    def test_401_maps_to_auth_error_with_hint(self):
        err = gbserver_http_error(401, "Invalid token")
        assert isinstance(err, GBServerAuthError)
        assert isinstance(err, GBServerHTTPError)
        assert err.status_code == 401
        assert "Invalid token" in err.detail and "auth login" in err.detail

    def test_other_status_is_plain_http_error(self):
        err = gbserver_http_error(404, "missing")
        assert type(err) is GBServerHTTPError
        assert (err.status_code, err.detail) == (404, "missing")
        assert str(err) == "404: missing"

    def test_auth_error_accepts_non_string_detail(self):
        err = gbserver_http_error(401, [{"msg": "bad token"}])
        assert "bad token" in err.detail and "auth login" in err.detail


class TestReloginHint:
    """The 401 hint names the login flow for the configured provider."""

    @pytest.mark.parametrize(
        "provider, command",
        [
            (None, "'gb auth login'"),
            ("github", "'gb auth login'"),
            ("ibmid", "'gb auth login --sso ibm'"),
            ("apikey", "'gb auth login --gbserver'"),
        ],
    )
    def test_hint_follows_default_provider(self, tmp_path, provider, command):
        creds = f'[user]\ndefault_provider = "{provider}"\n' if provider else ""
        (tmp_path / "credentials").write_text(creds)
        with (
            patch(
                "gbcli.utils.gbcredentials.get_local_gb_config",
                return_value=str(tmp_path),
            ),
            patch("gbcommon.types.gbenvconfig.is_standalone", return_value=False),
        ):
            assert command in gbserver_errors.relogin_hint()
            assert command in GBServerAuthError("Invalid token").detail

    def test_standalone_points_at_gbserver_login(self):
        with patch("gbcommon.types.gbenvconfig.is_standalone", return_value=True):
            assert "'gb auth login --gbserver'" in gbserver_errors.relogin_hint()
