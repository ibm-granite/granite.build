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

"""Behavior of the CLI version check.

``check_current_and_latest_versions()`` runs at the top of most ``gb`` commands. It
queries the public granite.build repo over unauthenticated HTTPS, so it needs no
GitHub credentials, SSH keys, or login and works everywhere (including standalone
mode). When the public lookup can't complete it silently returns "" rather than
blocking the command.
"""

import pytest

from gbcli.utils import versionutil


class TestVersionCheck:
    def test_no_credentials_required(self, monkeypatch):
        """The check never touches GitHub credentials/login — only the public API."""
        monkeypatch.setenv("GB_ENVIRONMENT", "STANDALONE")

        called = {}

        def _fake_latest(repo_org, repo_name):
            called["repo"] = (repo_org, repo_name)
            return "0.0.0"

        monkeypatch.setattr(versionutil, "get_latest_version", _fake_latest)
        monkeypatch.setattr(versionutil, "get_current_version", lambda _: "0.0.0")

        assert versionutil.check_current_and_latest_versions() == ""
        # The public granite.build repo is queried regardless of environment/auth.
        assert called["repo"] == (
            versionutil.GB_PUBLIC_REPO_ORG,
            versionutil.GB_PUBLIC_REPO_NAME,
        )

    def test_reports_when_outdated(self, monkeypatch):
        """An older installed version yields an upgrade notice mentioning the latest tag."""
        monkeypatch.setattr(versionutil, "get_latest_version", lambda *_: "2.0.0")
        monkeypatch.setattr(versionutil, "get_current_version", lambda _: "1.0.0")

        msg = versionutil.check_current_and_latest_versions()
        assert "2.0.0" in msg
        assert "1.0.0" in msg

    def test_silent_when_lookup_fails(self, monkeypatch):
        """A failed public lookup is swallowed (returns "") so the command isn't blocked."""

        def _boom(*args, **kwargs):
            raise Exception("network down")

        monkeypatch.setattr(versionutil, "get_latest_version", _boom)

        assert versionutil.check_current_and_latest_versions() == ""

    def test_get_latest_version_skips_malformed_tags(self, monkeypatch):
        """Malformed (non-PEP440) tags are ignored, not fatal, and the highest valid wins."""
        tags = [
            {"ref": "refs/tags/v1.0.0"},
            {"ref": "refs/tags/not-a-version"},  # malformed: skipped
            {"ref": "refs/tags/v2.3.1"},
            {"ref": "refs/tags/latest"},  # malformed: skipped
            {"ref": "refs/tags/v2.0.0"},
        ]
        monkeypatch.setattr(versionutil, "get_public_repo_tags", lambda *_: tags)

        assert versionutil.get_latest_version("ibm-granite", "granite.build") == "2.3.1"

    def test_get_latest_version_all_malformed(self, monkeypatch):
        """If no tag is a valid version, fall back to '0.0.0' rather than raising."""
        tags = [
            {"ref": "refs/tags/nightly"},
            {"ref": "refs/tags/release-candidate"},
        ]
        monkeypatch.setattr(versionutil, "get_public_repo_tags", lambda *_: tags)

        assert versionutil.get_latest_version("ibm-granite", "granite.build") == "0.0.0"
