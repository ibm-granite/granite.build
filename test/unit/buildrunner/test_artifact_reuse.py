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

These cover the retry-chain reuse logic added in PR #103: an artifact already
registered by an *ancestor* build (a build retry) must be reused with its
existing status preserved, while an artifact owned by an unrelated build must be
rejected.
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


def _make_runner(retry_chain_ids):
    """Build a BuildRunner with mocked storage, bypassing __init__.

    ``retry_chain_ids[0]`` is treated as the current (retry) build; the rest are
    ancestors reachable via ``retry_of_build_id``.
    """
    runner = object.__new__(BuildRunner)

    current_id = retry_chain_ids[0]
    parent_id = retry_chain_ids[1] if len(retry_chain_ids) > 1 else ""

    stored_build = MagicMock(spec=StoredBuild)
    stored_build.uuid = current_id
    stored_build.space_name = _SPACE
    stored_build.username = _USER
    stored_build.retry_of_build_id = parent_id
    runner.stored_build = stored_build

    # build_storage walks the retry chain via get_by_uuid(retry_of_build_id).
    ancestors = {}
    for idx, bid in enumerate(retry_chain_ids[1:], start=1):
        ancestor = MagicMock(spec=StoredBuild)
        ancestor.uuid = bid
        ancestor.retry_of_build_id = (
            retry_chain_ids[idx + 1] if idx + 1 < len(retry_chain_ids) else ""
        )
        ancestors[bid] = ancestor

    storage = MagicMock()
    storage.build_storage.get_by_uuid.side_effect = lambda uuid: ancestors.get(uuid)
    runner.storage = storage

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


class TestArtifactReuseAcrossRetryChain:
    """``__process_artifact_event`` reuse behavior for the non-pushed path."""

    def test_reuses_ancestor_artifact_and_preserves_success_status(self):
        """A build retry re-emitting an ancestor's artifact reuses it without
        resetting a SUCCESS status back to PENDING."""
        retry_build = "build-retry-2"
        original_build = "build-original-1"
        runner = _make_runner([retry_build, original_build])

        existing = _existing_artifact(
            created_by_build_id=original_build,
            created_by_target_id="target-orig",
            status=ArtifactRegistrationStatus.SUCCESS,
        )
        runner.storage.artifact_registry.get_by_uri.return_value = existing

        event = _make_event(build_id=retry_build, targetrun_id="target-retry")
        runner._BuildRunner__process_artifact_event(event, pushed=False)

        # The reused artifact keeps its SUCCESS status (the bug was resetting it
        # to PENDING, which would never be restored since the step is skipped).
        assert existing.status == ArtifactRegistrationStatus.SUCCESS
        # A reused record must NOT be re-written to storage...
        runner.storage.artifact_registry.update.assert_not_called()
        # ...but the current target must still be linked to it.
        runner._BuildRunner__update_target_with_artifact.assert_called_once()
        _, kwargs = runner._BuildRunner__update_target_with_artifact.call_args
        assert kwargs["artifact"] is existing

    def test_reuses_same_build_artifact_from_retried_step(self):
        """A retried step within the same build reuses its own prior artifact."""
        build_id = "build-1"
        runner = _make_runner([build_id])
        existing = _existing_artifact(
            created_by_build_id=build_id,
            created_by_target_id="target-1",
            status=ArtifactRegistrationStatus.PENDING,
        )
        runner.storage.artifact_registry.get_by_uri.return_value = existing

        event = _make_event(build_id=build_id, targetrun_id="target-1")
        runner._BuildRunner__process_artifact_event(event, pushed=False)

        runner.storage.artifact_registry.update.assert_not_called()
        runner._BuildRunner__update_target_with_artifact.assert_called_once()

    def test_rejects_artifact_from_build_outside_retry_chain(self):
        """An existing artifact owned by an unrelated build is rejected."""
        runner = _make_runner(["build-retry-2", "build-original-1"])

        existing = _existing_artifact(
            created_by_build_id="some-unrelated-build",
            created_by_target_id="target-x",
            status=ArtifactRegistrationStatus.SUCCESS,
        )
        runner.storage.artifact_registry.get_by_uri.return_value = existing

        event = _make_event(build_id="build-retry-2", targetrun_id="target-retry")
        with pytest.raises(ValueError, match="not in this retry chain"):
            runner._BuildRunner__process_artifact_event(event, pushed=False)

        runner.storage.artifact_registry.update.assert_not_called()

    def test_registers_new_artifact_when_none_exists(self):
        """With no existing record, a new artifact is created and persisted."""
        build_id = "build-1"
        runner = _make_runner([build_id])
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
