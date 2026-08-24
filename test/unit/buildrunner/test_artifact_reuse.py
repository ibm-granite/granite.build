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

"""Tests for artifact-reuse handling in ``BuildRunner.__process_artifact_event``.

With in-place retry a build keeps a single build id across attempts, so the same
artifact URI can legitimately be re-emitted within the one build (by a retried
step or a re-run target). When that happens the existing registration is reused
(its status is preserved) and re-associated to the run that just produced it. An
artifact owned by a *different* build id is rejected.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from gbserver.buildrunner.buildrunner import BuildRunner
from gbserver.storage.artifact_registration import (
    ArtifactRegistration,
    ArtifactRegistrationStatus,
)
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.artifact import ArtifactType
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventType,
    CreatedArtifactEventPayload,
    EntityRunMetadata,
)

# A URI that normalizes to itself (no LH revision/version rewriting needed).
_TEST_URI = "lh://lake-staging.cloud/granite_dot_build.public/tables/digit_input"
_SPACE = "test-space"
_USER = "testuser"
_BINDING = "model_out"


def _make_runner(build_id):
    """Build a BuildRunner with mocked storage, bypassing __init__.

    Args:
        build_id: the single build id this runner is running (in-place retry
            reuses this id across attempts).

    Returns:
        A BuildRunner whose storage and target-linking side effect are mocked.
    """
    runner = object.__new__(BuildRunner)

    stored_build = MagicMock(spec=StoredBuild)
    stored_build.uuid = build_id
    stored_build.space_name = _SPACE
    stored_build.username = _USER
    runner.stored_build = stored_build

    runner.storage = MagicMock()
    runner.build_run = None
    runner.build_message_logger = MagicMock()
    # Isolate the target-linking side effect; exercised separately elsewhere.
    runner._BuildRunner__update_target_with_artifact = MagicMock()
    return runner


def _make_event(build_id, targetrun_id):
    return BuildEvent(
        run_metadata=EntityRunMetadata(build_id=build_id, targetrun_id=targetrun_id),
        type=BuildEventType.NEWARTIFACT_IN_ENVIRONMENT_EVENT,
        payload=CreatedArtifactEventPayload(
            uri=_TEST_URI, binding_id=_BINDING, type=ArtifactType.FILESET
        ),
        timestamp=datetime(2026, 6, 17, 12, 0, 0),
        source="build-runner",
    )


def _existing_artifact(created_by_build_id, created_by_target_id, status):
    return ArtifactRegistration(
        uri=_TEST_URI,
        space_name=_SPACE,
        username=_USER,
        name=_BINDING,
        type=ArtifactType.FILESET,
        created_by_build_id=created_by_build_id,
        created_by_target_id=created_by_target_id,
        status=status,
    )


class TestArtifactReuseWithinBuild:
    """``__process_artifact_event`` reuse behavior for the non-pushed path."""

    def test_reuses_same_build_artifact_and_preserves_success_status(self):
        """A re-run target re-emitting the same build's artifact reuses it without
        resetting a SUCCESS status back to PENDING."""
        build_id = "build-1"
        runner = _make_runner(build_id)

        existing = _existing_artifact(
            created_by_build_id=build_id,
            created_by_target_id="target-attempt-1",
            status=ArtifactRegistrationStatus.SUCCESS,
        )
        runner.storage.artifact_registry.get_by_uri.return_value = existing

        event = _make_event(build_id=build_id, targetrun_id="target-attempt-1")
        runner._BuildRunner__process_artifact_event(event, pushed=False)

        # The reused artifact keeps its SUCCESS status (a reset to PENDING would
        # never be restored since the step is not re-run).
        assert existing.status == ArtifactRegistrationStatus.SUCCESS
        # Same target: no re-association write, and no new record inserted.
        runner.storage.artifact_registry.update_fields.assert_not_called()
        runner.storage.artifact_registry.update.assert_not_called()
        # The current target must still be linked to it.
        runner._BuildRunner__update_target_with_artifact.assert_called_once()
        _, kwargs = runner._BuildRunner__update_target_with_artifact.call_args
        assert kwargs["artifact"] is existing

    def test_reassociates_artifact_to_the_rerun_target(self):
        """A re-run target that re-emits an artifact first produced by an earlier
        (failed) target run re-associates it via created_by_target_id."""
        build_id = "build-1"
        runner = _make_runner(build_id)

        existing = _existing_artifact(
            created_by_build_id=build_id,
            created_by_target_id="target-failed",
            status=ArtifactRegistrationStatus.PENDING,
        )
        runner.storage.artifact_registry.get_by_uri.return_value = existing
        reassociated = _existing_artifact(
            created_by_build_id=build_id,
            created_by_target_id="target-success",
            status=ArtifactRegistrationStatus.PENDING,
        )
        runner.storage.artifact_registry.update_fields.return_value = reassociated

        event = _make_event(build_id=build_id, targetrun_id="target-success")
        runner._BuildRunner__process_artifact_event(event, pushed=False)

        # The registration is re-pointed at the successful re-run target.
        runner.storage.artifact_registry.update_fields.assert_called_once_with(
            existing.uuid, {"created_by_target_id": "target-success"}
        )
        # No brand-new record is inserted.
        runner.storage.artifact_registry.update.assert_not_called()
        runner._BuildRunner__update_target_with_artifact.assert_called_once()
        _, kwargs = runner._BuildRunner__update_target_with_artifact.call_args
        assert kwargs["artifact"] is reassociated

    def test_rejects_artifact_from_another_build(self):
        """An existing artifact owned by a different build id is rejected."""
        build_id = "build-1"
        runner = _make_runner(build_id)

        existing = _existing_artifact(
            created_by_build_id="some-unrelated-build",
            created_by_target_id="target-x",
            status=ArtifactRegistrationStatus.SUCCESS,
        )
        runner.storage.artifact_registry.get_by_uri.return_value = existing

        event = _make_event(build_id=build_id, targetrun_id="target-1")
        with pytest.raises(ValueError, match="another build"):
            runner._BuildRunner__process_artifact_event(event, pushed=False)

        runner.storage.artifact_registry.update.assert_not_called()
        runner.storage.artifact_registry.update_fields.assert_not_called()

    def test_registers_new_artifact_when_none_exists(self):
        """With no existing record, a new artifact is created and persisted."""
        build_id = "build-1"
        runner = _make_runner(build_id)
        runner.storage.artifact_registry.get_by_uri.return_value = None

        event = _make_event(build_id=build_id, targetrun_id="target-1")
        runner._BuildRunner__process_artifact_event(event, pushed=False)

        # A brand-new record is written as PENDING.
        runner.storage.artifact_registry.update.assert_called_once()
        (created,), _ = runner.storage.artifact_registry.update.call_args
        assert isinstance(created, ArtifactRegistration)
        assert created.status == ArtifactRegistrationStatus.PENDING
        assert created.created_by_build_id == build_id
        runner._BuildRunner__update_target_with_artifact.assert_called_once()
