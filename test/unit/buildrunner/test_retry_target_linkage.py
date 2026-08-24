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

from datetime import datetime
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


def _make_runner() -> BuildRunner:
    """A BuildRunner with mocked storage, bypassing __init__."""
    runner = object.__new__(BuildRunner)
    runner.storage = MagicMock()
    runner.build_run = SimpleNamespace()  # only a non-None sentinel is needed
    return runner


def _failed_run(uuid: str) -> StoredTargetRun:
    """A prior FAILED run of the target in this build."""
    return StoredTargetRun(
        uuid=uuid,
        build_id=_BUILD_ID,
        environment_uri=_ENV_URI,
        name=_TARGET,
        status=Status.FAILED,
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
        runner = _make_runner()
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
