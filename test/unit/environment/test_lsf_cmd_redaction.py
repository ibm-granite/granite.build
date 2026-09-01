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

"""Tests for Lsf._build_cmd_to_run_with_ssh command redaction.

The redacted command string is surfaced into build lineage (space-member readable)
and posted to the space via Slack, so a secret env-var value must never appear
there. These guard the switch from value-substring matching to masking by
env-var name — via the caller-supplied ``secret_keys`` (injected space secrets,
whatever the user named them) and ``SENSITIVE_KEY_RE`` — in particular the
empty-value case that previously spliced the placeholder between every character
and, as a side effect, left a genuinely secret value unmasked.
"""

from pathlib import Path
from typing import Self
from unittest.mock import patch

from gbserver.environment.lsf import Lsf
from gbserver.utils.redaction import REDACTED

# A synthetic secret used only in tests — not a real credential.
FAKE_SECRET = "s3cr3t-fake-value-not-real"
JOBSUB = Path("/tmp/jobsub.sh")


def _make_lsf() -> Lsf:
    """Create a minimal Lsf instance without running the real constructor."""
    with patch.object(Lsf, "__init__", lambda self, **kw: None):
        return Lsf.__new__(Lsf)


class TestBuildCmdRedaction:
    """Redaction of secret env-var values in the LSF-over-ssh command string."""

    def test_empty_value_does_not_splice_placeholder(self: Self) -> None:
        """An empty-valued env var must not interleave the placeholder per-char.

        Regression: ``str.replace("", ...)`` inserts the replacement between every
        character, mangling the whole command. Key-name masking avoids this.
        """
        lsf = _make_lsf()
        cmd, redacted = lsf._build_cmd_to_run_with_ssh(
            JOBSUB,
            {"EMPTY_VAR": "", "OTHER_VAR": "keep-me"},
        )
        assert cmd == f"EMPTY_VAR= OTHER_VAR=keep-me {JOBSUB}"
        # No per-character splicing: the placeholder appears nowhere.
        assert REDACTED not in redacted
        assert redacted == f"EMPTY_VAR= OTHER_VAR=keep-me {JOBSUB}"

    def test_secret_value_masked_even_alongside_empty_var(self: Self) -> None:
        """A secret-named var is masked, and an empty var can't unmask it.

        This is the security core of the regression: previously an empty value in
        ``to_redact`` shattered the command before the secret's own replace ran, so
        the secret leaked. Masking is now independent of the value text.
        """
        lsf = _make_lsf()
        _, redacted = lsf._build_cmd_to_run_with_ssh(
            JOBSUB,
            {
                "MOCK_FLAG": "",  # empty value, present but unset
                "HF_TOKEN": FAKE_SECRET,
            },
        )
        assert FAKE_SECRET not in redacted
        assert f"HF_TOKEN={REDACTED}" in redacted
        assert "MOCK_FLAG=" in redacted

    def test_secret_key_names_masked(self: Self) -> None:
        """Common secret-looking env-var names have their values masked."""
        lsf = _make_lsf()
        _, redacted = lsf._build_cmd_to_run_with_ssh(
            JOBSUB,
            {
                "HF_TOKEN": FAKE_SECRET,
                "MY_PASSWORD": FAKE_SECRET,
                "API_KEY": FAKE_SECRET,
                "AWS_SECRET_ACCESS_KEY": FAKE_SECRET,
            },
        )
        assert FAKE_SECRET not in redacted
        assert redacted.count(REDACTED) == 4

    def test_injected_secret_masked_under_non_secret_name(self: Self) -> None:
        """A space secret injected under an arbitrary name is masked via secret_keys.

        Injected space-secret env-var names are user-chosen and need not look
        secret (``GH_PAT``, ``DB_PASS``, ``DEPLOY_KEY``, ...). The caller marks
        those names in ``secret_keys`` so their values are masked even though
        ``SENSITIVE_KEY_RE`` would not match the name. Without this, such a secret
        would leak into the redacted command surfaced to space-readable lineage and
        Slack.
        """
        lsf = _make_lsf()
        env = {
            "GH_PAT": FAKE_SECRET,
            "DB_PASS": FAKE_SECRET,
            "DEPLOY_KEY": FAKE_SECRET,
        }
        _, redacted = lsf._build_cmd_to_run_with_ssh(JOBSUB, env, secret_keys=set(env))
        assert FAKE_SECRET not in redacted
        assert redacted.count(REDACTED) == 3

    def test_secret_keys_and_key_name_masking_combine(self: Self) -> None:
        """secret_keys and SENSITIVE_KEY_RE both mask; non-secret vars survive."""
        lsf = _make_lsf()
        _, redacted = lsf._build_cmd_to_run_with_ssh(
            JOBSUB,
            {
                "CUSTOM_CRED": FAKE_SECRET,  # masked via secret_keys only
                "HF_TOKEN": FAKE_SECRET,  # masked via key-name only
                "BUILD_ID": "abc-123",  # not secret, preserved
            },
            secret_keys={"CUSTOM_CRED"},
        )
        assert FAKE_SECRET not in redacted
        assert "CUSTOM_CRED=<redacted>" in redacted
        assert f"HF_TOKEN={REDACTED}" in redacted
        assert "BUILD_ID=abc-123" in redacted

    def test_non_secret_values_preserved(self: Self) -> None:
        """Operational (non-secret) env vars survive into the redacted command."""
        lsf = _make_lsf()
        cmd, redacted = lsf._build_cmd_to_run_with_ssh(
            JOBSUB,
            {"BUILD_ID": "abc-123", "TARGET_NAME": "download_file"},
        )
        assert redacted == cmd
        assert "BUILD_ID=abc-123" in redacted
        assert "TARGET_NAME=download_file" in redacted

    def test_url_credentials_scrubbed_under_non_secret_key(self: Self) -> None:
        """A credentialed URL under an innocuous key still gets scrubbed."""
        lsf = _make_lsf()
        _, redacted = lsf._build_cmd_to_run_with_ssh(
            JOBSUB,
            {"REPO_URL": "https://user:pw@github.com/org/repo.git"},
        )
        assert "user:pw" not in redacted
        assert f"REPO_URL=https://{REDACTED}@github.com/org/repo.git" in redacted

    def test_no_env_vars(self: Self) -> None:
        """With no env vars the command is just the jobsub path, unredacted."""
        lsf = _make_lsf()
        cmd, redacted = lsf._build_cmd_to_run_with_ssh(JOBSUB, None)
        assert cmd == str(JOBSUB)
        assert redacted == str(JOBSUB)
