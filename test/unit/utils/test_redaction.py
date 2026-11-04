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

"""Unit tests for gbserver.utils.redaction.

Guards the key-name based masking used before step config/metadata is emitted into
the member-readable build-lineage facet, and the free-text redaction and log-tail
sanitizing applied to job log tails that ride into failure events.
"""

import re
import time

import pytest

from gbserver.utils.redaction import (
    LOG_TAIL_MAX_BYTES,
    REDACTED,
    redact_sensitive,
    sanitize_log_tail,
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
    data = {"commit_hash": "deadbeef", "uri": "space://steps/misc/byoc", "count": 3}
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
        "space://steps/misc/byoc",
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
    """Non-container, non-secret values pass through as-is."""
    assert redact_sensitive("plain") == "plain"
    assert redact_sensitive(42) == 42
    assert redact_sensitive(None) is None


# --- redact_sensitive on strings: free text such as a command or log tail ------


@pytest.mark.standalone
@pytest.mark.parametrize(
    "line",
    [
        "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz",
        "export GITHUB_TOKEN=ghp_0123456789abcdefghij",
        "  api_key: s3cr3t-value-here",
        'password="hunter2hunter2"',
        "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCY",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig",
    ],
)
def test_redact_string_masks_secret_assignments(line):
    secret = re.split(r"[=:]\s*\"?|Bearer ", line)[-1].strip('"')
    out = redact_sensitive(line)
    assert secret not in out
    assert REDACTED in out


@pytest.mark.standalone
@pytest.mark.parametrize(
    "token",
    [
        "hf_abcdefghijklmnopqrstuvwxyz0123",
        "ghp_0123456789abcdefghijklmnopqrstuvwxyz",
        "github_pat_11ABCDEFG0123456789_abcdefghijklmnop",
        "AKIAIOSFODNN7EXAMPLE",
        "sk-proj-abcdefghijklmnopqrstuvwx1",
        "sk-" + "A1b2C3d4" * 6,  # OpenAI legacy, 48 chars
        "sk-svcacct-abcdefghij0123456789klmn",
        "sk-ant-api03-abcdefghij0123456789klmn",
        "sk-ant-admin01-abcdefghij0123456789klmn",
        "sk-1234abcd5678efgh9012ijkl",  # LiteLLM-style virtual key
        "sk_live_51Habcdefghij0123456789",  # Stripe
        "sk_test_51Habcdefghij0123456789",
        "xoxb-1234567890-abcdefghijklmnop",
    ],
)
def test_redact_string_masks_known_token_shapes_anywhere(token):
    out = redact_sensitive(f"pushing with {token} now")
    assert token not in out
    assert out == f"pushing with {REDACTED} now"


@pytest.mark.standalone
def test_redact_string_masks_known_secret_values_verbatim():
    """A declared secret value is masked wherever it appears, whatever it looks like."""
    out = redact_sensitive(
        "connecting as plainlookingvalue42 to db",
        secret_values=["plainlookingvalue42"],
    )
    assert out == f"connecting as {REDACTED} to db"


@pytest.mark.standalone
def test_redact_string_ignores_too_short_secret_values():
    """Very short values would mask ordinary words, so they are skipped."""
    assert redact_sensitive("run the job", secret_values=["the", ""]) == "run the job"


@pytest.mark.standalone
def test_redact_string_still_scrubs_url_credentials():
    out = redact_sensitive("git clone https://user:pa55word@github.com/o/r")
    assert "pa55word" not in out
    assert "github.com/o/r" in out


@pytest.mark.standalone
@pytest.mark.standalone
@pytest.mark.parametrize(
    "line",
    [
        "epoch 3 loss=0.123 lr=2e-5",
        "author: jane",
        "Traceback (most recent call last):",
        "GB_ARTIFACT_ID:model GB_ARTIFACT_PATH:/proj/out/model",
        # The sensitive word is only part of the name (review item 3):
        "max_tokens=2048",
        "--max-new-tokens=512",
        "tokens_per_second=4512.3",
        "Total tokens: 1000000",
        "tokenizer: /proj/models/granite-tok",
        "export TOKENIZERS_PARALLELISM=false",
        "PWD=/home/user/work",
        "secret_names_to_use_as_env_variable: [x]",
        # Hyphenated words after "sk-" that are not keys (no digit):
        "sk-learn-classifier-version-abcdefghij",
        "sk-hynix-memory-controller-firmware",
    ],
)
def test_redact_string_leaves_ordinary_log_lines_alone(line):
    assert redact_sensitive(line) == line


@pytest.mark.standalone
@pytest.mark.parametrize(
    "line",
    [
        "HF_TOKEN=abc123secretvalue",
        "api_key: s3cr3t-value-here",
        "DB_PASSWORD=hunter2hunter2",
        "--hf-token=abc123secretvalue",
        "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG",
        "github.token=abc123secretvalue",
        "password=hunter2hunter2",
        # Long prefixes (review: the {0,3} bound missed four or more parts):
        "MY_ORG_PROD_HF_TOKEN=abcdef123456",
        "ACME_TEAM_ML_PROD_WANDB_API_KEY=abcdef123456",
        "a.b.c.d.e.f.g.token=abcdef123456",  # eight prefix parts
    ],
)
def test_assignment_matches_when_the_name_ends_in_a_sensitive_word(line):
    value = re.split(r"[=:]\s*", line, maxsplit=1)[1]
    assert value not in redact_sensitive(line)


# --- sanitize_log_tail: shared LSF/SkyPilot failure-tail sanitizer ------------


@pytest.mark.standalone
def test_sanitize_log_tail_empty_is_none():
    assert sanitize_log_tail(None) is None
    assert sanitize_log_tail("") is None


@pytest.mark.standalone
def test_sanitize_log_tail_neutralizes_fences():
    assert "```" not in sanitize_log_tail("boom ``` still going")


@pytest.mark.standalone
def test_sanitize_log_tail_caps_and_marks_truncation():
    huge = "A" * (LOG_TAIL_MAX_BYTES * 2) + "REAL_ERROR_AT_END"
    out = sanitize_log_tail(huge)
    assert len(out.encode("utf-8")) <= LOG_TAIL_MAX_BYTES + len("...[truncated]...\n")
    assert out.startswith("...[truncated]...\n")
    assert out.endswith("REAL_ERROR_AT_END")


@pytest.mark.standalone
def test_sanitize_log_tail_redacts_secrets():
    out = sanitize_log_tail(
        "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz\nusing declaredvalue123\n",
        secret_values=["declaredvalue123"],
    )
    assert "hf_abcdefghijklmnopqrstuvwxyz" not in out
    assert "declaredvalue123" not in out


@pytest.mark.standalone
def test_sanitize_log_tail_redacts_before_capping():
    """Redaction runs on the full text first, so the byte cap can't cut a secret
    and leave a fragment the matchers no longer recognize."""
    secret = "declaredvalue123"
    text = "x" * 10 + " " + secret + " " + "y" * (LOG_TAIL_MAX_BYTES - 8)
    out = sanitize_log_tail(text, secret_values=[secret])
    assert "value123" not in out and "ue123" not in out


@pytest.mark.standalone
def test_masks_secret_assignment_in_string_under_non_secret_key():
    """A secret-named assignment inside a string value is masked even though the
    dict key itself (``start_command``) is not secret-looking."""
    result = redact_sensitive(
        {"start_command": "HF_TOKEN=abc123secretvalue python train.py"}
    )
    assert result == {"start_command": f"HF_TOKEN={REDACTED} python train.py"}


@pytest.mark.standalone
def test_secret_values_thread_through_nested_containers():
    result = redact_sensitive(
        {"cmds": ["echo declaredvalue123"], "nested": {"x": "use declaredvalue123"}},
        secret_values=["declaredvalue123"],
    )
    assert result == {
        "cmds": [f"echo {REDACTED}"],
        "nested": {"x": f"use {REDACTED}"},
    }


# --- Performance: redaction must stay linear (review item 2) -------------------


@pytest.mark.standalone
def test_prefix_bound_still_leaves_non_secret_names_alone():
    """Widening the prefix bound must not start matching names whose last part
    isn't sensitive, however many parts they have."""
    for line in (
        "MY_ORG_PROD_MAX_TOKENS=2048",
        "a.b.c.d.e.f.g.tokenizer=/proj/tok",
        "A_B_C_D_E_F_G_H_I_TOKEN_COUNT=5",
    ):
        assert redact_sensitive(line) == line


@pytest.mark.standalone
@pytest.mark.parametrize(
    "text",
    [
        "a" * 200_000,
        "token" * 40_000,
        "a-" * 100_000,
        "eyJaaaaaaaaaaaa-" * 12_500,
        "x://" + "a:" * 100_000,
        "HF_TOKEN=" + "v" * 200_000,
    ],
    ids=["a", "token", "a-", "jwt-ish", "url-ish", "long-value"],
)
def test_redact_sensitive_is_fast_on_pathological_strings(text):
    """Every pattern runs in linear time, so a 200 KB line of identifier
    characters (base64, minified JSON) redacts in well under a second. The
    unbounded key prefix took ~10 s on 20 KB of ``a-``."""
    start = time.perf_counter()
    redact_sensitive(text)
    assert time.perf_counter() - start < 1.0


@pytest.mark.standalone
def test_sanitize_log_tail_bounds_work_on_huge_input():
    """The sanitizer trims to a fixed window before redacting, so its cost does
    not grow with the raw tail (LSF's ``tail -n`` has no width limit)."""
    huge = "a-" * 2_000_000 + "\nTraceback: real error"  # ~4 MB, one long line
    start = time.perf_counter()
    out = sanitize_log_tail(huge)
    assert time.perf_counter() - start < 1.0
    assert out.endswith("Traceback: real error")
    assert out.startswith("...[truncated]...")


@pytest.mark.standalone
def test_trim_window_cannot_split_a_secret_at_the_cap():
    """Trimming happens before redaction with slack beyond the display cap, and
    redaction before the final cap, so a secret at the cap edge is still masked
    whole rather than left as a fragment."""
    secret = "declaredvalue123"
    text = "x" * 10 + " " + secret + " " + "y" * (LOG_TAIL_MAX_BYTES - 8)
    out = sanitize_log_tail(text, secret_values=[secret])
    assert "value123" not in out and "ue123" not in out
