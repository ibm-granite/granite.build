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

``evaluate_version_status()`` runs (via the command-layer ``enforce_version_check``) at
the top of most ``gb`` commands. It queries the public granite.build repo over
unauthenticated HTTPS, so it needs no GitHub credentials, SSH keys, or login and works
everywhere (including standalone mode). It is click-free and returns a
``VersionCheckResult`` rather than echoing or exiting.

The policy is floor-based: a client at or above the ``min-supported`` floor is allowed to
run (warning only if a newer version exists), while a client below the floor is blocked.
When the public lookup can't complete it returns ``UNKNOWN`` so the command proceeds.
"""

from gbcli.utils import versionutil
from gbcli.utils.versionutil import VersionStatus


def _tag(name, sha=None):
    """Build one entry of a ``/repos/.../tags`` response.

    ``commit.sha`` is the endpoint's commit SHA (already peeled for annotated tags).
    """
    entry = {"name": name}
    if sha is not None:
        entry["commit"] = {"sha": sha}
    return entry


class TestEvaluateVersionStatus:
    def test_no_credentials_required(self, monkeypatch):
        """The check never touches GitHub credentials/login — only the public API."""
        monkeypatch.setenv("GB_ENVIRONMENT", "STANDALONE")

        called = {}

        def _fake_tags(repo_org, repo_name):
            called["repo"] = (repo_org, repo_name)
            return []

        monkeypatch.setattr(versionutil, "get_public_repo_tags", _fake_tags)
        monkeypatch.setattr(versionutil, "get_current_version", lambda _: "0.0.0")

        result = versionutil.evaluate_version_status()
        assert result.status is VersionStatus.UP_TO_DATE
        # The public granite.build repo is queried regardless of environment/auth.
        assert called["repo"] == (
            versionutil.GB_PUBLIC_REPO_ORG,
            versionutil.GB_PUBLIC_REPO_NAME,
        )

    def test_reports_when_outdated(self, monkeypatch):
        """An outdated client at/above the floor yields OUTDATED_WARN naming both versions."""
        tags = [
            _tag("v2.0.0", "sha-latest"),
            _tag("v1.0.0", "sha-floor"),
            _tag("min-supported", "sha-floor"),  # floor = 1.0.0, client is at it
        ]
        monkeypatch.setattr(versionutil, "get_public_repo_tags", lambda *_: tags)
        monkeypatch.setattr(versionutil, "get_current_version", lambda _: "1.0.0")

        result = versionutil.evaluate_version_status()
        assert result.status is VersionStatus.OUTDATED_WARN
        assert result.latest_version == "2.0.0"
        assert result.current_version == "1.0.0"
        assert "2.0.0" in result.message
        assert "1.0.0" in result.message
        # The message points at the recommended @stable upgrade command.
        assert "granite.build.git@stable" in result.message

    def test_unknown_when_lookup_fails(self, monkeypatch):
        """A failed public lookup yields UNKNOWN so the command isn't blocked."""

        def _boom(*args, **kwargs):
            raise Exception("network down")

        monkeypatch.setattr(versionutil, "get_public_repo_tags", _boom)

        assert versionutil.evaluate_version_status().status is VersionStatus.UNKNOWN

    def test_unknown_when_current_version_unparseable(self, monkeypatch):
        """An unparseable installed version (e.g. "unknown" from a non-pip-installed
        source checkout) must not raise InvalidVersion — it yields UNKNOWN."""
        monkeypatch.setattr(
            versionutil, "get_public_repo_tags", lambda *_: [_tag("v2.0.0")]
        )
        monkeypatch.setattr(versionutil, "get_current_version", lambda _: "unknown")

        assert versionutil.evaluate_version_status().status is VersionStatus.UNKNOWN

    def test_below_floor_blocks(self, monkeypatch):
        """Below the min-supported floor yields BELOW_FLOOR with a mandatory-upgrade msg."""
        tags = [
            _tag("v2.0.0", "sha-latest"),
            _tag("v1.5.0", "sha-floor"),
            _tag("min-supported", "sha-floor"),
        ]
        monkeypatch.setattr(versionutil, "get_public_repo_tags", lambda *_: tags)
        monkeypatch.setattr(versionutil, "get_current_version", lambda _: "1.0.0")

        result = versionutil.evaluate_version_status()
        assert result.status is VersionStatus.BELOW_FLOOR
        assert result.floor_version == "1.5.0"
        assert "1.5.0" in result.message
        assert "granite.build.git@stable" in result.message

    def test_at_or_above_floor_warns_only(self, monkeypatch):
        """At/above the floor with a newer version available warns but does not block."""
        tags = [
            _tag("v2.0.0", "sha-latest"),
            _tag("v1.5.0", "sha-floor"),
            _tag("min-supported", "sha-floor"),
        ]
        monkeypatch.setattr(versionutil, "get_public_repo_tags", lambda *_: tags)
        monkeypatch.setattr(versionutil, "get_current_version", lambda _: "1.6.0")

        assert (
            versionutil.evaluate_version_status().status is VersionStatus.OUTDATED_WARN
        )

    def test_missing_min_supported_tag_mandates_upgrade(self, monkeypatch):
        """With no min-supported tag the floor is unknown, so we fall back to the pre-floor
        behavior and block any outdated client (a missing tag never softens a block)."""
        tags = [_tag("v2.0.0", "sha-latest")]
        monkeypatch.setattr(versionutil, "get_public_repo_tags", lambda *_: tags)
        monkeypatch.setattr(versionutil, "get_current_version", lambda _: "1.0.0")

        result = versionutil.evaluate_version_status()
        assert result.status is VersionStatus.BELOW_FLOOR
        assert result.floor_version == ""
        assert "granite.build.git@stable" in result.message
        # The block message says the upgrade is required, not just "available".
        assert "required" in result.message

    def test_missing_min_supported_tag_up_to_date_ok(self, monkeypatch):
        """No floor, but the client is current -> still UP_TO_DATE (no false block)."""
        tags = [_tag("v2.0.0", "sha-latest")]
        monkeypatch.setattr(versionutil, "get_public_repo_tags", lambda *_: tags)
        monkeypatch.setattr(versionutil, "get_current_version", lambda _: "2.0.0")

        assert versionutil.evaluate_version_status().status is VersionStatus.UP_TO_DATE

    def test_annotated_min_supported_resolves_via_commit_sha(self, monkeypatch):
        """Regression: min-supported and its vX.Y.Z tag are both *annotated* and point at
        the same commit. The /repos/.../tags endpoint reports each tag's peeled commit
        SHA, so they match and the floor resolves (with /git/refs/tags they would not,
        because annotated tags expose their tag-object SHA instead)."""
        tags = [
            _tag("v2.0.0", "sha-latest-commit"),
            _tag("v1.5.0", "sha-floor-commit"),
            _tag("min-supported", "sha-floor-commit"),  # same commit as v1.5.0
        ]
        monkeypatch.setattr(versionutil, "get_public_repo_tags", lambda *_: tags)
        monkeypatch.setattr(versionutil, "get_current_version", lambda _: "1.6.0")

        result = versionutil.evaluate_version_status()
        assert result.floor_version == "1.5.0"
        assert result.status is VersionStatus.OUTDATED_WARN

    def test_min_supported_commit_matches_no_version_tag_mandates_upgrade(
        self, monkeypatch
    ):
        """If min-supported's commit matches no vX.Y.Z tag, the floor is unresolved ->
        treated as no floor -> an outdated client is blocked, not softened to a warning.
        """
        tags = [
            _tag("v2.0.0", "sha-latest"),
            _tag("v1.5.0", "sha-floor-commit"),
            _tag("min-supported", "sha-orphan-commit"),  # matches no version tag
        ]
        monkeypatch.setattr(versionutil, "get_public_repo_tags", lambda *_: tags)
        monkeypatch.setattr(versionutil, "get_current_version", lambda _: "1.0.0")

        result = versionutil.evaluate_version_status()
        assert result.status is VersionStatus.BELOW_FLOOR
        assert result.floor_version == ""

    def test_up_to_date_at_latest(self, monkeypatch):
        """Current == latest (and == floor) is UP_TO_DATE with no message."""
        tags = [
            _tag("v2.0.0", "sha-latest"),
            _tag("min-supported", "sha-latest"),
        ]
        monkeypatch.setattr(versionutil, "get_public_repo_tags", lambda *_: tags)
        monkeypatch.setattr(versionutil, "get_current_version", lambda _: "2.0.0")

        result = versionutil.evaluate_version_status()
        assert result.status is VersionStatus.UP_TO_DATE
        assert result.message == ""


class TestResolveVersionsFromTags:
    def test_resolves_latest_and_floor(self):
        tags = [
            _tag("v2.0.0", "sha-latest"),
            _tag("v1.5.0", "sha-floor"),
            _tag("min-supported", "sha-floor"),
        ]
        assert versionutil._resolve_versions_from_tags(tags) == ("2.0.0", "1.5.0")

    def test_skips_malformed_tags(self):
        """Malformed (non-PEP440) tags are ignored, not fatal, and the highest valid wins."""
        tags = [
            _tag("v1.0.0"),
            _tag("not-a-version"),  # malformed: skipped
            _tag("v2.3.1"),
            _tag("latest"),  # malformed: skipped
            _tag("v2.0.0"),
        ]
        latest, floor = versionutil._resolve_versions_from_tags(tags)
        assert latest == "2.3.1"
        assert floor == ""

    def test_all_malformed_falls_back(self):
        """If no tag is a valid version, fall back to '0.0.0' rather than raising."""
        tags = [_tag("nightly"), _tag("release-candidate")]
        assert versionutil._resolve_versions_from_tags(tags) == ("0.0.0", "")


class TestGetLatestVersion:
    def test_get_latest_version_skips_malformed_tags(self, monkeypatch):
        tags = [_tag("v1.0.0"), _tag("not-a-version"), _tag("v2.3.1")]
        monkeypatch.setattr(versionutil, "get_public_repo_tags", lambda *_: tags)
        assert versionutil.get_latest_version("ibm-granite", "granite.build") == "2.3.1"

    def test_get_latest_version_all_malformed(self, monkeypatch):
        tags = [_tag("nightly"), _tag("release-candidate")]
        monkeypatch.setattr(versionutil, "get_public_repo_tags", lambda *_: tags)
        assert versionutil.get_latest_version("ibm-granite", "granite.build") == "0.0.0"
