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

"""The build's PR is optional: when GitHub keeps failing the build still runs,
and the run stops trying (and stops posting PR messages) after a few failures."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import gbserver.buildrunner.build_setup as build_setup
import gbserver.buildrunner.buildlogger as buildlogger
import gbserver.utils.git_retry as git_retry
from gbserver.buildrunner.build_setup import BuildSetup
from gbserver.spaces.db_space import get_pr_repo_uri
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_space import RepoSnapshot, StoredSpace
from gbserver.types.status import Status
from gbserver.utils.git_retry import end_build_git_run, get_build_git_breaker


class _Valid:
    def is_valid(self):
        return True


@pytest.fixture
def setup(tmp_path, monkeypatch):
    space = StoredSpace(name="s", git_repo_uri="https://github.example.com/o/space")
    storage = MagicMock()
    storage.space_storage.get_by_name.return_value = space
    monkeypatch.setattr(build_setup, "get_admin_storage", lambda: storage)
    monkeypatch.setattr(build_setup, "PR_CREATION_RETRY_INTERVAL", 0)
    monkeypatch.setattr(
        build_setup.BuildValidation,
        "validate_stored_build",
        staticmethod(lambda **_: _Valid()),
    )
    messages = MagicMock()
    monkeypatch.setattr(build_setup, "get_message_logger", lambda *_: messages)
    monkeypatch.setattr(build_setup, "post_validated_build_message", MagicMock())
    build = StoredBuild(
        name="b", space_name="s", source_uri="", username="u", build_archive="x"
    )
    yield SimpleNamespace(
        setup=BuildSetup(workspace_dir=tmp_path, event_source="t", gh_token="tok"),
        build=build,
        messages=messages,
    )
    end_build_git_run(build.uuid)


def test_pr_failure_does_not_fail_the_build(setup, monkeypatch):
    attempts = []

    def helper(self, stored_build, try_count=0):
        attempts.append(try_count)
        raise RuntimeError("github unreachable")

    monkeypatch.setattr(BuildSetup, "_BuildSetup__create_pr_for_build_helper", helper)
    success, updates = setup.setup.run(setup.build)

    assert success
    assert "status" not in updates and "source_uri" not in updates
    assert setup.build.status == Status.PENDING
    # Gave up after the breaker's failure budget, not MAX_PR_CREATION_TRIES.
    assert len(attempts) == git_retry.BUILD_GIT_MAX_CONSECUTIVE_FAILURES
    assert get_build_git_breaker(setup.build.uuid).is_open
    setup.messages.warning.assert_called_once()


def test_pr_success_is_recorded(setup, monkeypatch):
    url = "https://github.example.com/org/gbspace-x/pull/7"
    monkeypatch.setattr(
        BuildSetup, "_BuildSetup__create_pr_for_build_helper", lambda *a, **k: url
    )
    success, updates = setup.setup.run(setup.build)
    assert success and updates == {"source_uri": url}
    setup.messages.warning.assert_not_called()


def test_space_without_pr_repo_skips_pr(setup, monkeypatch):
    """A local space has no GitHub repo: no attempts, no breaker failures."""
    setup.setup.storage.space_storage.get_by_name.return_value = StoredSpace(
        name="s", git_repo_uri="file:///tmp/space"
    )
    helper = MagicMock()
    monkeypatch.setattr(BuildSetup, "_BuildSetup__create_pr_for_build_helper", helper)
    success, _ = setup.setup.run(setup.build)
    assert success
    helper.assert_not_called()
    assert get_build_git_breaker(setup.build.uuid).consecutive_failures == 0


def test_pr_messages_stop_once_given_up(monkeypatch):
    build = StoredBuild(
        name="b",
        space_name="s",
        source_uri="https://github.example.com/org/gbspace-x/pull/7",
        username="u",
        build_archive="x",
    )
    pr_logger = buildlogger.BuildPRLogger(stored_build=build, gh_token="tok")
    api = MagicMock()
    api.update_issue_comment.side_effect = RuntimeError("github down")
    pr_logger.github_api = api
    try:
        for _ in range(10):
            pr_logger.info("hello")  # never raises
        assert (
            api.update_issue_comment.call_count
            == git_retry.BUILD_GIT_MAX_CONSECUTIVE_FAILURES
        )
        assert not pr_logger.messages  # nothing queued up behind the failure
    finally:
        end_build_git_run(build.uuid)


@pytest.mark.parametrize(
    "row, expected",
    [
        (
            StoredSpace(name="g", git_repo_uri="https://github.example.com/o/space"),
            "https://github.example.com/o/space",
        ),
        (
            StoredSpace(
                name="g", git_repo_uri="git+ssh://github.example.com/o/space.git@cfg"
            ),
            "https://github.example.com/o/space",
        ),
        (
            StoredSpace(
                name="d",
                git_repo_uri="https://github.example.com/o/space",
                space_uri="db://spaces/d",
                space=RepoSnapshot(
                    archive="x", source_uri="https://github.example.com/o/space"
                ),
            ),
            "https://github.example.com/o/space",
        ),
        (StoredSpace(name="l", git_repo_uri="file:///tmp/space"), None),
    ],
)
def test_pr_repo(row, expected):
    """db-backed spaces keep their PRs in the snapshot's source repo."""
    assert get_pr_repo_uri(row) == expected
