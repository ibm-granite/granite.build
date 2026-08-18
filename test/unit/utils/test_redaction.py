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

"""Unit tests for gbserver.utils.redaction.redact_sensitive.

Guards the key-name based masking used before step config/metadata is emitted into
the member-readable build-lineage facet.
"""

import pytest

from gbserver.utils.redaction import (
    REDACTED,
    redact_sensitive,
    scrub_url_credentials,
)


@pytest.mark.standalone
def test_masks_common_secret_key_names():
    """Keys whose names look secret are masked; ``-``/``_`` and case tolerant."""
    result = redact_sensitive(
        {
            "password": "p",
            "PASSWD": "p",
            "pwd": "p",
            "db_pwd": "p",
            "api_key": "k",
            "api-key": "k",
            "apiKey": "k",
            "access_key": "a",
            "private-key": "pk",
            "token": "t",
            "credential": "c",
            "SECRET": "s",
            "authorization": "a",
            "Authorization": "a",
            "bearer": "b",
            "cookie": "c",
            "ssh_key": "sk",
            "sshKey": "sk",
            "auth": "au",
            "auth_token": "au",
            "authToken": "au",
            "AUTH_TOKEN": "au",
        }
    )
    assert set(result.values()) == {REDACTED}


@pytest.mark.standalone
def test_non_secret_keys_pass_through():
    """Operational keys (e.g. commit_hash) are returned unchanged."""
    data = {"commit_hash": "deadbeef", "uri": "space://steps/byoc", "count": 3}
    assert redact_sensitive(data) == data


@pytest.mark.standalone
def test_auth_prefix_does_not_over_mask():
    """``auth`` matches as a token but not the ``author``/``authentication`` prefix.

    Guards the bounded ``auth`` alternative: git-oriented metadata legitimately carries
    ``author``-style keys, which must survive into lineage unmasked. The exclusion keys
    on the word stem case-insensitively, so ALL-CAPS env-var-style keys (``AUTHOR``,
    ``GIT_AUTHOR``, ``AUTHORED_DATE``) are covered too — not just lowercase.
    """
    data = {
        "author": "octocat",
        "authored_date": "2026-01-01",
        "authenticated": True,
        "AUTHOR": "octocat",
        "AUTHOR_EMAIL": "octocat@example.com",
        "AUTHORED_DATE": "2026-01-01",
        "GIT_AUTHOR": "octocat",
        "AUTHENTICATED": True,
    }
    assert redact_sensitive(data) == data


@pytest.mark.standalone
def test_scrubs_credentialed_url_under_non_secret_key():
    """A ``user:secret`` clone URL under an innocuous key (``repo``) is scrubbed.

    Guards the byoc/BYOS regression: key-name masking alone misses this because the
    key name isn't secret-looking, so the value pass must strip the credential pair.
    """
    result = redact_sensitive(
        {
            "repo": "https://x-access-token:ghp_SECRET@github.com/org/repo",
            "uri": "https://user:pw@github.com/org/repo.git",
        }
    )
    assert result == {
        "repo": f"https://{REDACTED}@github.com/org/repo",
        "uri": f"https://{REDACTED}@github.com/org/repo.git",
    }


@pytest.mark.standalone
def test_scrubs_url_embedded_in_command_string():
    """Credentials survive inside a larger command line — scrub them there too."""
    result = redact_sensitive(
        {"start_command": "git clone https://user:pw@github.com/org/repo.git /w"}
    )
    assert result == {
        "start_command": f"git clone https://{REDACTED}@github.com/org/repo.git /w"
    }


@pytest.mark.standalone
def test_scrub_url_credentials_leaves_clean_urls_untouched():
    """URLs with no ``user:secret`` pair are unchanged.

    Only a credential *pair* is masked, so a bare username (``git@``, ``user@``,
    token-as-username) is preserved — as are clean URLs, ``space://`` refs, and an
    ``@`` that appears in a path segment rather than the authority.
    """
    for clean in (
        "https://github.com/org/repo.git",
        "space://steps/byoc",
        "https://github.com/org/repo/path@ref",  # @ after a path segment, not userinfo
        "git+ssh://git@github.com/org/repo.git",  # bare username, no password
        "https://ghp_TOKEN@github.com/org/repo",  # token-as-username, no ':' (see caveat)
        "just a plain string, no url",
    ):
        assert scrub_url_credentials(clean) == clean


@pytest.mark.standalone
def test_recurses_into_nested_dicts_and_lists():
    """Nested dicts and dicts inside lists are redacted in place."""
    result = redact_sensitive(
        {
            "outer": {"token": "t", "ok": 1},
            "items": [{"secret": "s"}, {"name": "n"}],
        }
    )
    assert result == {
        "outer": {"token": REDACTED, "ok": 1},
        "items": [{"secret": REDACTED}, {"name": "n"}],
    }


@pytest.mark.standalone
def test_does_not_mutate_input():
    """The original mapping is left untouched (a copy is returned)."""
    original = {"token": "t", "nested": {"password": "p"}}
    redact_sensitive(original)
    assert original == {"token": "t", "nested": {"password": "p"}}


@pytest.mark.standalone
def test_scalars_returned_unchanged():
    """Non-container values pass through as-is."""
    assert redact_sensitive("plain") == "plain"
    assert redact_sensitive(42) == 42
    assert redact_sensitive(None) is None
