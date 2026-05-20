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

"""
Utility functions for build finalization.
"""

from typing import Callable, Optional, Union

from gbserver.metrics.metrics_client import push_metrics
from gbserver.storage.artifact_registration import ArtifactRegistrationStatus
from gbserver.storage.singleton_storage import SingletonAdminStorage, get_admin_storage
from gbserver.storage.steprun_storage import IStoredStepRunStorage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_step_run import StoredStepRun
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.storage.target_run_storage import IStoredTargetRunStorage
from gbserver.types.metrics import Metric, MetricMetadata, MetricName
from gbserver.types.status import Status
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


def finalize_build_status(
    build_id: str, status: Status, failure_reason: str = ""
) -> Optional[StoredBuild]:
    """Finalize targets, steps, artifacts, and the build itself for a build that has reached a terminal state.

    This function ensures all child entities (targets, steps, artifacts) of a build
    are updated to reflect the build's final status, and then updates the build status itself.
    It can be called externally (e.g., from BuildRunnerJob) when a build needs to be
    finalized without going through the normal BuildRunner event flow.

    Args:
        build_id: The UUID of the build to finalize
        status: The final status of the build (should be a finished status like SUCCESS, FAILED, CANCELLED)
        failure_reason: Optional reason for failure (only used when status is FAILED)

    Return:
        The updated build if it was updated successfully.
    """
    if not status.is_finished():
        logger.warning(
            "finalize_build_status called with non-finished status %s for build %s, ignoring",
            status,
            build_id,
        )
        return  # type: ignore[return-value]

    storage = get_admin_storage()

    # Check if build exists and is not already finished
    build: Optional[StoredBuild] = storage.build_storage.get_by_uuid(build_id)  # type: ignore[assignment]
    if build is None:
        logger.warning("finalize_build_status: build %s not found", build_id)
        return None

    if build.status.is_finished():
        if build.status != status:
            logger.warning(
                "finalize_build_status: build %s already has finished status %s, skipping",
                build_id,
                build.status,
            )
            return build
        # Build already has the same finished status. Still re-run entity finalization to
        # catch any targets/steps stored concurrently with a previous finalization call
        # (race condition between the BuildWatcher monitoring thread calling stop() and the
        # build runner's event loop processing queued entity events).
        logger.info(
            "Build %s already has status %s, re-finalizing entities only",
            build_id,
            status,
        )
    else:
        logger.info("Finalizing build %s with status %s", build_id, status)

    # Finalize targets
    _finalize_target_or_step_status(
        build_id=build_id,
        storage=storage.target_storage,
        status=status,
        item_name="target",
    )

    # Finalize steps
    _finalize_target_or_step_status(
        build_id=build_id,
        storage=storage.step_storage,
        status=status,
        item_name="step",
    )

    # Finalize artifacts
    _finalize_artifact_status(
        build_id=build_id,
        storage=storage,
        status=status,
    )

    # Update the build status itself (only if not already in a finished state)
    if not build.status.is_finished():
        build = update_stored_build_status(build.uuid, status, failure_reason)
        logger.info("Build %s finalized with status %s", build_id, status)

    # Deactivate webhook subscriptions for completed builds
    try:
        from gbserver.types.constants import GBSERVER_WEBHOOKS_ENABLED

        if GBSERVER_WEBHOOKS_ENABLED:
            from gbserver.webhooks.sql_storage import create_webhook_storage

            webhook_storage = create_webhook_storage()
            count = webhook_storage.deactivate_for_build(build_id)
            if count > 0:
                logger.info(
                    "Deactivated %d webhook subscription(s) for completed build %s",
                    count,
                    build_id,
                )
    except Exception as e:
        logger.warning("Failed to deactivate webhook subscriptions: %s", e)

    return build


def update_stored_build_status(
    build_id: str,
    status: Status,
    failure_reason: str = "",
    should_update: Optional[Callable[[StoredBuild], bool]] = None,
) -> Optional[StoredBuild]:
    """Update the status of the stored build in storage.

    Returns:
        The updated build if successful, or None if required_current_status was specified
        and the current status didn't match.
    """
    updates = {}
    updates["status"] = status
    if failure_reason and status is Status.FAILED:
        updates["failure_reason"] = failure_reason  # type: ignore[assignment]
    build_storage = get_admin_storage().build_storage
    build = build_storage.update_fields(build_id, updates, should_update=should_update)
    return build


def _finalize_target_or_step_status(
    build_id: str,
    storage: Union[IStoredTargetRunStorage, IStoredStepRunStorage],
    status: Status,
    item_name: str,
) -> None:
    """For targets/steps that have not finished, apply the given build status.

    If given status is not one that represents a finished state (e.g, SUCCESS, CANCELLED, FAILED),
    then don't do anything.

    For all but PENDING target/step status values, the given status is applied directly.
    For PENDING targets/steps:
        - If status==SUCCESS, leave as PENDING (likely a bug, log warning)
        - Otherwise set to CANCELLED

    Args:
        build_id: ID of build to find targets or steps
        storage: One of step or target storage
        status: Status value from the build to apply
        item_name: One of "step" or "target", for logging
    """
    if not status.is_finished():
        return

    logger.info(
        "Build finished, updating status of stored %s as required.",
        item_name,
    )
    build_id_search = {"build_id": build_id}
    targets_or_steps = storage.get_by_where(build_id_search)
    for target_or_step in targets_or_steps:
        assert isinstance(
            target_or_step, (StoredStepRun, StoredTargetRun)
        ), f"invalid target_or_step: {target_or_step}"
        stored_status = target_or_step.status
        if not stored_status.is_finished() and stored_status != status:
            if stored_status is Status.PENDING:
                if status is Status.SUCCESS:
                    # This is likely a g.b bug, so leave the target/step as PENDING and log it.
                    logger.warning(
                        "Build succeeded with a PENDING %s: %s",
                        item_name,
                        target_or_step,
                    )
                    continue  # And leave the status as PENDING
                target_or_step.status = Status.CANCELLED
            else:
                target_or_step.status = status
            logger.info(
                "Correcting %s status based on build status %s: %s",
                item_name,
                status,
                target_or_step,
            )
            storage.update_fields(target_or_step.uuid, {"status": target_or_step.status})  # type: ignore[arg-type]


def _finalize_artifact_status(
    build_id: str, storage: SingletonAdminStorage, status: Status
) -> None:
    """For artifacts that are still PENDING, apply an appropriate final status based on the build status.

    When a build finishes, any artifacts that are still PENDING should be finalized:
    - If build succeeded (SUCCESS), leave artifacts as PENDING and log a warning (likely a bug)
    - If build failed (FAILED), set artifacts to FAILED
    - If build was cancelled (CANCELLED), set artifacts to CANCELLED

    Args:
        build_id: ID of the build whose artifacts should be finalized
        storage: Admin storage instance
        status: Final status of the build
    """
    if not status.is_finished():
        return

    logger.info("Build finished, updating status of PENDING artifacts as required.")
    artifacts = storage.artifact_registry.get_by_where(
        {"created_by_build_id": build_id}
    )
    for artifact in artifacts:
        if artifact.status == ArtifactRegistrationStatus.PENDING:
            if status is Status.SUCCESS:
                # This is likely a g.b bug, so leave the artifact as PENDING and log it.
                logger.warning(
                    "Build succeeded with a PENDING artifact: %s",
                    artifact,
                )
                continue  # Leave the status as PENDING
            elif status is Status.FAILED:
                artifact.status = ArtifactRegistrationStatus.FAILED
            else:  # CANCELLED or other terminal states
                artifact.status = ArtifactRegistrationStatus.CANCELLED
            logger.info(
                "Correcting artifact status based on build status %s: %s",
                status,
                artifact,
            )
            storage.artifact_registry.update_fields(
                artifact.uuid, {"status": artifact.status}
            )


def push_failed_status_update_metric(build_id: str, status_list: list[Status]):
    push_metrics(
        metrics=[
            Metric(
                name=MetricName.BUILD_STATUS_RACE_DETECTED,
                value=1,
                metadata=MetricMetadata(
                    build_id=build_id,
                    expected_status=str(status_list),
                ),
            )
        ]
    )
