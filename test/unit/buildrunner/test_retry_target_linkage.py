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

"""Failed->success target-run linkage for in-place build retry.

When a FAILED target re-runs and succeeds within the *same* build, the new
SUCCESS ``StoredTargetRun`` links back to the prior FAILED run via
``retry_of_target_id``. This is derived by ``__find_prior_failed_target_run`` and
applied in ``__create_and_store_target_run`` (the single creation point for target
runs). These tests exercise that lookup and its wiring directly, with the build
structure mocked.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gbserver.buildrunner.buildrunner import BuildRunner
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.artifact import ArtifactType
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventType,
    CreatedArtifactEventPayload,
    EntityRunMetadata,
)
from gbserver.types.status import Status

_BUILD_ID = "build-1"
_TARGET = "targetB"
_ENV_URI = "space://environments/bash"


def _make_runner(retry_count: int = 0) -> BuildRunner:
    """A BuildRunner with mocked storage, bypassing __init__.

    Args:
        retry_count: the stored build's retry_count. Linkage in
            __create_and_store_target_run must NOT depend on retry_count (a restart
            resets it to 0 yet leaves prior FAILED runs), so this is set on the mock
            to guard against a retry_count gate being re-introduced.
    """
    runner = object.__new__(BuildRunner)
    runner.storage = MagicMock()
    runner.build_run = SimpleNamespace()  # only a non-None sentinel is needed
    runner.stored_build = SimpleNamespace(retry_count=retry_count)
    return runner


def _failed_run(uuid: str, finished_at: datetime | None = None) -> StoredTargetRun:
    """A prior FAILED run of the target in this build.

    Args:
        uuid: the run's uuid.
        finished_at: when the run reached its terminal FAILED status; used to
            order multiple failed attempts (latest wins).
    """
    return StoredTargetRun(
        uuid=uuid,
        build_id=_BUILD_ID,
        environment_uri=_ENV_URI,
        name=_TARGET,
        status=Status.FAILED,
        finished_at=finished_at,
    )


class TestFindPriorFailedTargetRun:
    """``__find_prior_failed_target_run`` returns the FAILED run to link back to."""

    def test_returns_uuid_of_prior_failed_run(self):
        runner = _make_runner()
        runner.storage.target_storage.get_by_where.return_value = [
            _failed_run("failed-run-1")
        ]

        result = runner._BuildRunner__find_prior_failed_target_run(
            build_id=_BUILD_ID, target_name=_TARGET
        )

        assert result == "failed-run-1"
        # The lookup is scoped to the same build id, the target name, and FAILED.
        runner.storage.target_storage.get_by_where.assert_called_once_with(
            {"build_id": _BUILD_ID, "name": _TARGET, "status": Status.FAILED.name}
        )

    def test_links_to_most_recent_failed_run_with_multiple_failures(self):
        # With max_retries >= 2 a target can fail more than once; get_by_where's
        # ordering is undefined, so return the runs oldest-last to prove the
        # lookup orders by finished_at rather than trusting list position.
        runner = _make_runner()
        runner.storage.target_storage.get_by_where.return_value = [
            _failed_run("failed-run-1", finished_at=datetime(2026, 6, 17, 12, 0, 0)),
            _failed_run("failed-run-2", finished_at=datetime(2026, 6, 17, 12, 5, 0)),
        ][::-1]

        result = runner._BuildRunner__find_prior_failed_target_run(
            build_id=_BUILD_ID, target_name=_TARGET
        )

        assert result == "failed-run-2"

    def test_links_latest_when_a_failed_run_has_no_finished_at(self):
        # In production finished_at is tz-aware (get_time()). A defensively
        # unset value falls back to the tz-aware _SORT_EPOCH; mixing it with an
        # aware finished_at must not raise "can't compare offset-naive and
        # offset-aware datetimes", and the run with a real finished_at wins.
        runner = _make_runner()
        aware = datetime(2026, 6, 17, 12, 5, 0, tzinfo=timezone.utc)
        runner.storage.target_storage.get_by_where.return_value = [
            _failed_run("failed-run-no-ts", finished_at=None),
            _failed_run("failed-run-aware", finished_at=aware),
        ]

        result = runner._BuildRunner__find_prior_failed_target_run(
            build_id=_BUILD_ID, target_name=_TARGET
        )

        assert result == "failed-run-aware"

    def test_returns_empty_when_no_prior_failed_run(self):
        runner = _make_runner()
        runner.storage.target_storage.get_by_where.return_value = []

        result = runner._BuildRunner__find_prior_failed_target_run(
            build_id=_BUILD_ID, target_name=_TARGET
        )

        assert result == ""


class TestCreateAndStoreTargetRunLinkage:
    """``__create_and_store_target_run`` stamps retry_of_target_id from the lookup."""

    def _event(self, targetrun_id: str) -> BuildEvent:
        return BuildEvent(
            run_metadata=EntityRunMetadata(
                build_id=_BUILD_ID, target_name=_TARGET, targetrun_id=targetrun_id
            ),
            type=BuildEventType.STATUS_EVENT,
            payload=CreatedArtifactEventPayload(
                uri="", binding_id="", type=ArtifactType.FILESET
            ),
            timestamp=datetime(2026, 6, 17, 12, 0, 0),
            source="build-runner",
        )

    def _fake_build(self):
        """A minimal build whose single target resolves an environment uri."""
        env_asset = SimpleNamespace(uristr=_ENV_URI)
        target = SimpleNamespace(
            environment=SimpleNamespace(environment_asset=env_asset)
        )
        # config is only isinstance-checked; BuildConfig is patched to object so
        # any value passes, keeping this fake free of BuildConfig's required fields.
        return SimpleNamespace(config=object(), targets={_TARGET: target})

    def test_success_run_links_back_to_prior_failed_run(self):
        # retry_count == 0 deliberately: a restart reuses the build id and resets
        # retry_count to 0 while leaving pre-restart FAILED runs in place, so
        # linkage must NOT be gated on retry_count. A prior FAILED run exists, so
        # the new SUCCESS run links back to it regardless of retry_count.
        runner = _make_runner(retry_count=0)
        runner.storage.target_storage.get_by_where.return_value = [
            _failed_run("failed-run-1")
        ]

        with (
            patch(
                "gbserver.buildrunner.buildrunner.build_from_build_run",
                return_value=self._fake_build(),
            ),
            patch("gbserver.buildrunner.buildrunner.BuildConfig", object),
        ):
            created = runner._BuildRunner__create_and_store_target_run(
                self._event("success-run"),
                status=Status.SUCCESS,
                input_artifacts={},
            )

        assert created.retry_of_target_id == "failed-run-1"
        assert created.status == Status.SUCCESS
        # The created run is persisted.
        runner.storage.target_storage.add.assert_called_once_with(created)

    def test_first_run_has_no_linkage(self):
        # Genuine first attempt: the lookup runs but finds no prior FAILED run,
        # so retry_of_target_id stays "".
        runner = _make_runner()
        runner.storage.target_storage.get_by_where.return_value = []

        with (
            patch(
                "gbserver.buildrunner.buildrunner.build_from_build_run",
                return_value=self._fake_build(),
            ),
            patch("gbserver.buildrunner.buildrunner.BuildConfig", object),
        ):
            created = runner._BuildRunner__create_and_store_target_run(
                self._event("first-run"),
                status=Status.FAILED,
                input_artifacts={},
            )

        assert created.retry_of_target_id == ""
