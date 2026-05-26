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

# from .artifact import ArtifactStoreType, ArtifactType
# from .resources import ResourceSpec, ResourceType


"""
The Build runner and build event processor.
"""

import asyncio
import tempfile
import threading
import time
import traceback
from asyncio import Event, Queue
from base64 import b64decode
from pathlib import Path
from typing import Callable, List, Optional, Self, Union

from gbcommon.uri.git import GitURI
from gbcommon.uri.uri import URI
from gbcommon.uri.utils import get_artifact_type
from gbserver.build.build import Build
from gbserver.build.buildrun import BuildRun
from gbserver.build.run import RunFailed
from gbserver.build.space import Space
from gbserver.buildwatcher.abstractbuildrunner import AbstractBuildRunner
from gbserver.buildwatcher.build_setup import BuildSetup
from gbserver.buildwatcher.build_utils import (
    finalize_build_status,
    push_failed_status_update_metric,
    update_stored_build_status,
)
from gbserver.buildwatcher.buildlogger import (
    get_message_logger,
)
from gbserver.github.myghapi import MyGHApi
from gbserver.lineage.jobstats import get_lineage_store
from gbserver.metrics.metrics_client import push_metrics
from gbserver.storage.artifact_registration import (
    ArtifactRegistration,
    ArtifactRegistrationStatus,
)
from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_event import StoredEvent
from gbserver.storage.stored_step_run import StoredStepRun
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.buildconfig import BuildConfig, BuildTargetConfig
from gbserver.types.buildevent import (
    ArtifactPushedEventPayload,
    BuildEvent,
    BuildEventMessagePayload,
    BuildEventMetricsPayload,
    BuildEventStatusPayload,
    BuildEventTerminatePayload,
    BuildEventType,
    BuildEventWorkloadStatusPayload,
    CreatedArtifactEventPayload,
)
from gbserver.types.constants import (
    DEFAULT_DIR_PERMS,
    DEFAULT_GH_API_ENDPOINT,
    DEFAULT_ROOT_WORKSPACE_DIR,
    GBSERVER_GITHUB_TOKEN,
    GBSERVER_RAISE_BUILD_EXCEPTIONS,
    GBSERVER_WEBHOOKS_ENABLED,
    STARTING_BUILD_MESSAGE,
    WORKSPACE_BUILDS_DIR,
)
from gbserver.types.status import STATUS_TO_ICON, Status
from gbserver.utils.archive import extract_archive
from gbserver.utils.logger import get_logger
from gbserver.utils.unwrap_errors import get_readable_error_message
from gbserver.utils.utils import get_build_status_link

logger = get_logger(__name__)
# How many times the monitoring interval between checks for cancellation
_CANCEL_CHECK_MONITORING_INTERVAL_MULTIPLIER = 5
_BUILD_EVENT_SOURCE_NAME = "build-runner"


def build_from_build_run(build_run: BuildRun) -> Build:
    """Extracts the build and stored build objects from the build run."""
    build = build_run.entity
    assert isinstance(build, Build), f"invalid build_run.entity build: {build}"
    return build


class BuildRunner(AbstractBuildRunner):
    """Manager for monitoring Lakehouse and launching builds."""

    # Additional Constructor parameters
    space_uri: Optional[str]
    create_pr: bool
    enable_resume: bool = False

    # object managing a running build
    build_run: Optional[BuildRun]

    # Event to stop all threads
    stop_event: threading.Event

    # These go together to enable posting comments to the PR.
    github_api: Optional[MyGHApi]
    pr_id: Optional[str]

    def __init__(
        self: Self,
        build: StoredBuild,
        gh_token: str = GBSERVER_GITHUB_TOKEN,
        space_uri: Optional[str] = None,  # Overrides build's space
        workspace_dir: Union[str, Path] = DEFAULT_ROOT_WORKSPACE_DIR,
        monitoring_interval: int = 5,
        gh_api_endpoint: str = DEFAULT_GH_API_ENDPOINT,
        create_pr: bool = True,
        enable_resume: bool = False,
        dry_run: bool = False,
    ) -> None:
        super().__init__(
            build=build,
            gh_token=gh_token,
            workspace_dir=workspace_dir,
            monitoring_interval=monitoring_interval,
            gh_api_endpoint=gh_api_endpoint,
            dry_run=dry_run,
        )
        self.space_uri = space_uri
        self.create_pr = create_pr
        self.enable_resume = enable_resume
        self.build_run = None
        self.stop_event = threading.Event()
        self.build_message_logger = get_message_logger(
            build, _BUILD_EVENT_SOURCE_NAME
        )  # To be recreated later.
        self.event_storage = get_admin_storage().event_storage
        self._webhook_writer = None

    def stop(self: Self) -> None:
        """Stop the building thread if it was started."""
        logger.debug("BuildRunner.stop start")
        self.__cancel_build_run()
        logger.debug("BuildRunner.stop end")

    def start_and_wait(self: Self) -> None:
        """Run the inmemory build that was provided to the initializer in the current thread.
        Make sure the build is in build storage and add if not.
        In this case, if storing the build failed, then raise an exception.
        Returns after the build has completed/failed/cancelled or stop() has been called from another thread.

        Args:
            self (Self):
            stored_build (StoredBuild): a build which may or may not be already in admin build storage.

        """
        # Make sure the build is in the admin storage, which we need later.
        build_id = self.stored_build.uuid
        item = self.storage.build_storage.get_by_uuid(build_id)
        if item is None:
            logger.info("Storing unstored build: %s", build_id)
            self.storage.build_storage.add(self.stored_build)
            # This read-back is a double-check since we have seen records disappear from LH.
            item = self.storage.build_storage.get_by_uuid(build_id)
            if item is None:
                raise ValueError("Storage of build failed without error.")
            logger.info("Build successfuly stored build: %s", build_id)
        try:
            buildrunner_resume: bool = self.enable_resume and self.__should_resume()

            while True:
                if self.stored_build.status == Status.FAILED:
                    logger.info(
                        "Build %s is already FAILED; skipping run and proceeding to retry",
                        self.stored_build.uuid,
                    )
                else:
                    if buildrunner_resume:
                        logger.info(
                            "Build %s is already RUNNING; RESUMING instead of starting a new run",
                            self.stored_build.uuid,
                        )
                    else:
                        logger.info(
                            "Starting new build run for build %s",
                            self.stored_build.uuid,
                        )

                    asyncio.run(
                        self.__async_run_build(buildrunner_resume=buildrunner_resume)
                    )

                    logger.info(
                        "Build run completed (build_id=%s, resume=%s)",
                        self.stored_build.uuid,
                        buildrunner_resume,
                    )

                retry_build = self.__prepare_retry()
                if retry_build is None:
                    break
                self.stored_build = retry_build
                self.stop_event.clear()
                self.build_message_logger = get_message_logger(
                    retry_build, _BUILD_EVENT_SOURCE_NAME
                )
                buildrunner_resume = False

        except Exception as e:
            logger.error("build execution failed for build ID %s", build_id)
            raise e

    def __should_resume(self) -> bool:
        """
        Return True if this build looks like it was already started
        by a previous build-runner instance.
        """
        return self.stored_build.status == Status.RUNNING

    def __prepare_retry(self: Self) -> Optional[StoredBuild]:
        """If the finished build is eligible for retry, create, store, and return a new PENDING build.

        Returns the new retry build ready to run, or None if no retry should occur.
        """
        latest = self.storage.build_storage.get_by_uuid(self.stored_build.uuid)
        if latest is None or not isinstance(latest, StoredBuild):
            return None
        if not self._should_retry(latest):
            return None
        retry_build = StoredBuild(
            name=latest.name,
            space_name=latest.space_name,
            source_uri="",
            username=latest.username,
            build_archive=latest.build_archive,
            status=Status.PENDING,
            targets=latest.targets,
            description=latest.description,
            tags=latest.tags,
            retry_of_build_id=(
                latest.retry_of_build_id
                if latest.retry_of_build_id is not None
                else latest.uuid
            ),
            retry_count=latest.retry_count + 1,
        )
        self.storage.build_storage.add(retry_build)
        latest.retry_build_id = retry_build.uuid
        self.storage.build_storage.update(latest)
        logger.info(
            "Build %s failed; retrying as build %s (attempt %d of %d)",
            latest.uuid,
            retry_build.uuid,
            retry_build.retry_count,
            latest.get_build_config().retries.max_retries,
        )
        return retry_build

    def __comment_on_original_pr(self: Self) -> None:
        """Post a comment on the original failed build's PR linking to this retry build's PR."""
        original = self.storage.build_storage.get_by_uuid(
            self.stored_build.retry_of_build_id
        )
        if not isinstance(original, StoredBuild) or not original.source_uri:
            return
        owner, repo, pr_id = original.get_pr_info()
        if not pr_id:
            return
        try:
            myapi = MyGHApi(token=self.gh_token, owner=owner, repo=repo)
            build_config = original.get_build_config()
            body = (
                f"Build failed (attempt {self.stored_build.retry_count} of "
                f"{build_config.retries.max_retries}). "
                f"Automatically retrying: {self.stored_build.source_uri}"
            )
            myapi.update_issue_comment(body=body, pr_id=pr_id)
        except Exception as e:
            logger.warning(
                "Failed to post retry comment on original PR %s: %s", pr_id, e
            )

    def __get_targets_to_resume(self) -> Optional[List[str]]:
        """
        Returns the targets that are not yet completed and should be started/resumed.

        If any target is FAILED or INVALID, the build should be FAILED and NOT RESUMED.
        """
        target_runs = self.storage.target_storage.get_by_where(
            {"build_id": self.stored_build.uuid}
        )

        # Get the already failed targets from storage to not resume them again
        failed_targets = {
            tr.name
            for tr in target_runs
            if tr.status in (Status.FAILED, Status.INVALID)
        }

        if failed_targets:
            logger.error(
                "Build %s cannot be resumed: targets already FAILED/INVALID: %s",
                self.stored_build.uuid,
                sorted(failed_targets),
            )

            # Mark build failed
            if not self.stored_build.status.is_finished():
                self.__update_stored_build_status(
                    status=Status.FAILED,
                    failure_reason=(
                        "Cannot resume build; failed targets detected: "
                        + ", ".join(sorted(failed_targets))
                    ),
                )

            raise RuntimeError(
                f"Build {self.stored_build.uuid} has failed targets; resume aborted"
            )

        # Fetch the successful targets to NOT resume them again
        completed_targets = {
            tr.name for tr in target_runs if tr.status == Status.SUCCESS
        }

        if not self.stored_build.targets:
            return None

        resumable = [t for t in self.stored_build.targets if t not in completed_targets]

        logger.info(
            "Resuming build %s with targets: %s",
            self.stored_build.uuid,
            resumable,
        )

        return resumable

    async def __async_run_build(self, buildrunner_resume: bool) -> None:
        """Starts or Resumes the build depending on the buildrunner_resume flag."""

        stored_build = self.stored_build
        build_id = stored_build.uuid

        logger.info("Running build %s in resume=%s", build_id, buildrunner_resume)

        event_q: Queue[Event] = Queue()

        try:

            space = self.__get_build_space(stored_build)
            targets = stored_build.targets  # default: re-run all targets of the build
            force_fetch = True  # default: set to force fetch

            if not buildrunner_resume:

                workspace_dir = self.workspace_dir / WORKSPACE_BUILDS_DIR
                workspace_dir.mkdir(mode=DEFAULT_DIR_PERMS, parents=True, exist_ok=True)

                if not self.__setup(space):  # type: ignore[arg-type]
                    return  # Log messages were issued.

                build_dir = Path(tempfile.mkdtemp()) / build_id
                build_archive_bytes = stored_build.load_from_build_archive()
                extract_archive(build_archive_bytes, build_dir)

                force_fetch = True
                targets = stored_build.targets

            else:
                # RESUME
                # Resume an already-running build after a build-runner restarts.
                # - Do NOT recreate build artifacts
                # - Do NOT re-run setup

                workspace_dir = Path(self.workspace_dir)
                if not workspace_dir.exists():
                    raise RuntimeError(
                        f"Cannot resume build {build_id}: workspace_dir missing: {workspace_dir}"
                    )

                # not triggering  __setup() again

                build_dir = (
                    None  # DO NOT recreate build dir again - will get in workspace dir
                )
                force_fetch = False  # DO NOT refetch

                # Fetch the targets which has not COMPLETED - target-level resume in a build
                targets = self.__get_targets_to_resume()

            build = Build(
                build_dir=build_dir,
                space=space,
                build_id=build_id,
                username=stored_build.username,
                workspace_dir=workspace_dir,
                event_q=event_q,
                targets=targets,  # check which targets have finished (status == PENDING && RUNNING)
                force_fetch=force_fetch,
                target_already_run_fn=(
                    self.__is_target_already_run
                    if self.stored_build.retry_of_build_id
                    and self.stored_build.get_build_config().retries.target_reuse_enabled
                    else None
                ),
            )

            if not buildrunner_resume:
                if build.val_errors and not build.val_errors.is_valid(
                    check_warnings=True
                ):
                    self.build_message_logger.warning(str(build.val_errors))

                lineage_body = STARTING_BUILD_MESSAGE.format(
                    build_status_link=get_build_status_link(build_id)
                )
                self.build_message_logger.info(lineage_body)

            build_run = BuildRun(build=build, event_q=event_q)
            assert build_run.id == build_id
            self.build_run = build_run
            logger.info("starting the build: %s", build_id)
            build_task = asyncio.create_task(build_run.run_and_wait())

            # Create the monitoring task after the build_run to allow it to make
            # enough progress so that event monitoring, cancellation checking all work.
            event_monitoring_task = asyncio.create_task(
                self.__worker_task(event_q=event_q, build_id=build_id)
            )

            try:
                await build_task
            except asyncio.CancelledError:
                pass

            self.build_run = None

            self.stop_event.set()
            await event_monitoring_task

            logger.info(
                "Build %s finished (buildrunner_resume=%s)",
                build_id,
                buildrunner_resume,
            )

        except Exception as e:
            err_stack = traceback.format_exc()
            logger.error("%s", err_stack)
            if not stored_build.status.is_finished():
                logger.info(
                    "updating build status to failed for build %s", stored_build.uuid
                )
                self.__update_stored_build_status(
                    status=Status.FAILED, failure_reason=str(err_stack)
                )
            markdown = (
                f"build `{build_id}` status `{self.stored_build.status}`, error: {e}"
            )
            self.build_message_logger.error(markdown=markdown)

    def __setup(self: Self, space: Space):
        """Do any setup prior to actually running the build.  This includes
        1) validating the build.yaml
        2) Creating a PR for the build.

        Returns:
            True: build is valid and PR is created with gb_builds table updated to reflect PR location
            False: build is either invalid or we could not create the PR.  In the former case,
            the build is marked as INVALID, in the latter FAILED.
        """
        if self.stored_build.source_uri:
            # Build has already been validated and had a PR created for it.
            # Recreate the message logger to include the PR logger.
            self.build_message_logger = get_message_logger(
                self.stored_build, _BUILD_EVENT_SOURCE_NAME
            )
            return True

        build_setup = BuildSetup(
            workspace_dir=self.workspace_dir,
            event_source=_BUILD_EVENT_SOURCE_NAME,
            gh_token=self.gh_token,
            stop_event=self.stop_event,
            create_pr=self.create_pr,
            space=space,
        )

        # Also issues error messages to a build_message logger.
        success, updates = build_setup.run(self.stored_build)

        # Recreate the message logger to include the PR logger created in run() above.
        self.build_message_logger = get_message_logger(
            self.stored_build, _BUILD_EVENT_SOURCE_NAME
        )

        # Persist the build status and PR link, if any.
        # Don't update if status is other than PENDING
        update = self.storage.build_storage.update_fields(
            self.stored_build.uuid,
            updates,
            should_update=lambda item: item.status == Status.PENDING,
        )
        if update is not None:  # Build had pending status, all good.
            self.stored_build = update
            success = True
            logger.info(
                "Build update succeeded.  Build status is now %s",
                self.stored_build.status,
            )
            if self.stored_build.retry_of_build_id and self.stored_build.source_uri:
                self.__comment_on_original_pr()
        else:
            _build_result = self.storage.build_storage.get_by_uuid(
                self.stored_build.uuid
            )
            assert isinstance(_build_result, StoredBuild)
            self.stored_build = _build_result
            success = False
            logger.warning(
                "Build update failed.  Likely as the status was not %s (currently %s).",
                Status.PENDING,
                self.stored_build.status,
            )
            push_failed_status_update_metric(self.stored_build.uuid, [Status.PENDING])
        return success

    def __get_build_space(self: Self, stored_build: StoredBuild) -> Optional[Space]:
        """Get the Space object for the given build, if it defines a space name"""
        space = None
        username = stored_build.username
        if self.space_uri is not None:
            logger.info("using all_build_space_uri: %s", self.space_uri)
            # This is primarily for debugging from the command line when
            # we want the build to use a local git clone of a space.
            space = Space(self.space_uri, username=username)
            logger.info("space: %s", space)
        elif stored_build.space_name != "":
            space_name = stored_build.space_name
            stored_space = self.storage.space_storage.get_by_name(space_name)
            if stored_space is None:
                raise ValueError(f"Space {space_name} not found in space storage")
            s_uri = GitURI.get_gb_space_config_uri(
                uri=stored_space.git_repo_uri, token=self.gh_token
            )
            logger.info("for the space name %s using the URI: %s", space_name, s_uri)
            space = Space(uri=s_uri, username=username)
        return space

    def __is_build_cancelled(self) -> bool:
        # See if something else has signalled us to stop the build.
        build_id = self.stored_build.uuid
        try:
            build: StoredBuild = self.storage.build_storage.get_by_uuid(build_id)  # type: ignore
            if build is None:
                logger.warning(
                    "Build id %s unexpectedly not found on cancellation request check. Ignoring.",
                    build_id,
                )
                return False
            return build.status in (Status.CANCEL_REQUESTED, Status.CANCELLED)
        except Exception as e:
            logger.warning(
                "Exception checking if build %s is cancelled: %s. Assuming not cancelled.",
                build_id,
                e,
            )
            return False

    async def __worker_task(self: Self, event_q: Queue[Event], build_id: str) -> None:
        logger.debug("BuildRunner.__worker_task start build_id: %s", build_id)
        build_finished = False
        # last_cancel_check_time = time.time()
        while not self.stop_event.is_set():
            try:
                logger.info("build %s waiting for events...", build_id)
                event = await asyncio.wait_for(
                    event_q.get(), timeout=self.monitoring_interval
                )
                assert isinstance(event, BuildEvent), f"invalid event: {event}"
                logger.info(
                    "build %s got a new event: %s : %s", build_id, event.type, event
                )
                build_finished = self.__process_event(event=event)
                if build_finished:
                    logger.debug("build %s finished, exiting monitoring loop", build_id)
                    break
            # This is only available in Python 3.13
            # https://docs.python.org/3.13/library/asyncio-queue.html#asyncio.Queue.shutdown
            # except QueueShutDown:
            #     logger.debug("event_q has been shutdown!")
            #     break
            except TimeoutError:
                logger.debug("event_q.get() timeout!")
                continue
            except Exception as e:
                # TODO: do we need to exit the loop and mark the build as failed and cancel the build.
                err_stack_trace = traceback.format_exc()
                msg = f"build {build_id} failed to process the event: {event} , error: {e}"
                logger.error(msg)
                logger.error("Failed event processing:\n%s", err_stack_trace)
                reason = "failed to process the event"
                body = f"{reason}:\n```\n{event}\n```\n\nTrace:\n```\n{err_stack_trace}\n```\n"
                try:
                    self.build_message_logger.info(markdown=body)
                except Exception:
                    err_stack_trace = traceback.format_exc()
                    logger.error(
                        f"Ignoring failure to log msg via the build_message_logger\nmsg={body}\nbuild_message_logger exception=\n{err_stack_trace}"
                    )
                if (
                    GBSERVER_RAISE_BUILD_EXCEPTIONS
                ):  # Initially enabled only during build tests.
                    logger.error(
                        "Marking build %s as failed due to exception logged above",
                        build_id,
                    )
                    self.__cancel_and_fail_build(failure_reason=msg)
                    raise e
            finally:
                # Check for cancellation, every time through the loop to minimize time when
                # CANCEL_REQUESTED can be ignored via  a builds status event that sets to RUNNING.
                if self.__is_build_cancelled():
                    break  # and call cancel_build_run() below
        if self.stop_event.is_set():
            logger.warning("stop event has been set, stopping event monitoring...")

        if not build_finished:
            try:
                self.__cancel_build_run()
            except Exception as e:
                err_stack_trace = traceback.format_exc()
                logger.error(
                    "Failed build cancellation for build %s error: %s\n%s",
                    build_id,
                    e,
                    err_stack_trace,
                )

        logger.debug("BuildRunner.__worker_task end build_id: %s", build_id)

    def __cancel_and_fail_build(self: Self, failure_reason: str) -> None:
        """Stop/cancel the build and mark it as failed"""
        try:
            self.__cancel_build_run(update_status=False)
        except Exception as e:
            logger.error("Could not cancel build %s", self.stored_build.uuid)

        try:
            self.__update_stored_build_status(
                status=Status.FAILED, failure_reason=failure_reason
            )
        except Exception as e:
            logger.error("Could not mark build %s as failed", self.stored_build.uuid)

    def __cancel_build_run(self: Self, update_status: bool = True) -> None:
        """Cancel an inprogress build_run and mark the stored build as CANCELLED, unless it has finished."""

        self.stop_event.set()

        # This can be called both from the worker thread and stop(), so try and avoid duplication.
        if self.stored_build.status == Status.CANCELLED:
            if update_status:
                # Re-run entity finalization to catch any targets/steps stored concurrently
                # with a previous finalize_build_status() call (race between the BuildWatcher
                # monitoring thread and the build runner's event loop).
                finalize_build_status(self.stored_build.uuid, Status.CANCELLED)
            return

        if self.build_run is not None:
            status = self.build_run.status
            if status is Status.PENDING or status is Status.RUNNING:
                logger.info("cancelling the build %s", self.stored_build.uuid)
                self.build_run.cancel()
            else:
                logger.info(
                    "Not marking finished build as cancelled:  build id = %s",
                    self.stored_build.uuid,
                )
        else:
            logger.warning("self.build_run is None")
        if update_status and not self.stored_build.status.is_finished():
            self.__update_stored_build_status(Status.CANCELLED)

    def __process_workload_status_event(self: Self, event: BuildEvent) -> None:
        payload = event.payload
        assert isinstance(
            payload, BuildEventWorkloadStatusPayload
        ), f"payload is invalid: {event}"
        logger.info(
            "Workload status is %s for build %s",
            payload.status,
            event.run_metadata,
        )

    def __process_metrics_event(self: Self, event: BuildEvent) -> None:
        payload = event.payload
        assert isinstance(
            payload, BuildEventMetricsPayload
        ), f"payload is invalid: {event}"
        logger.info(
            "Got some metrics for the build '%s' : %s",
            event.run_metadata,
            payload.metrics,
        )
        push_metrics(metrics=payload.metrics)

    def __process_event(self: Self, event: Event) -> bool:
        """
        Process the event, making GitHub and Lakehouse updates.
        Returns True if the build finished running.
        """
        if self.stored_build.status.is_finished():
            logger.warning(
                "Still receiving events after build has finished (i.e. cancelled, success, failed, etc)."
                + " Ignoring event and terminating event processing."
            )
            return True
        logger.debug("BuildRunner.process_event start")
        logger.debug("event: %s", event)
        build_finished = False
        if not isinstance(event, BuildEvent):
            logger.error("only build events are supported right now")
            return build_finished

        # Add the BuildEvent to the history of events for this build
        self.event_storage.add(StoredEvent(build_event=event))

        # Dispatch to webhook subscribers
        self.__dispatch_to_webhooks(event)

        if event.type in (
            BuildEventType.NEWARTIFACT_IN_ENVIRONMENT_EVENT,
            BuildEventType.NEW_MULTIARTIFACT_IN_ENVIRONMENT_EVENT,
        ):
            logger.info(
                "Internal event %s comes along with a separate event %s, so is being ignored",
                event.type,
                BuildEventType.ARTIFACT_EVENT,
            )
        elif event.type is BuildEventType.ARTIFACT_PUSHED_EVENT:
            self.__process_artifact_event(event=event, pushed=True)
        elif event.type is BuildEventType.ARTIFACT_EVENT:
            self.__process_artifact_event(event=event, pushed=False)
        elif event.type is BuildEventType.MESSAGE_EVENT:
            self.__process_message_event(event=event)
        elif event.type is BuildEventType.TERMINATE_EVENT:
            self.__process_terminate_event(event=event)
        elif event.type is BuildEventType.STATUS_EVENT:
            build_finished = self.__process_build_status_event(event=event)
        elif event.type is BuildEventType.WORKLOAD_STATUS_EVENT:
            self.__process_workload_status_event(event=event)
        elif event.type is BuildEventType.METRICS_EVENT:
            self.__process_metrics_event(event=event)
        else:
            logger.error("unsupported event type: %s", event)
        logger.debug("BuildRunner.process_event end")
        return build_finished

    def __dispatch_to_webhooks(self: Self, event: BuildEvent) -> None:
        """Persist event to webhook storage for matching subscriptions."""
        if not GBSERVER_WEBHOOKS_ENABLED:
            return

        try:
            if self._webhook_writer is None:
                from gbserver.webhooks.event_writer import WebhookEventWriter

                writer = WebhookEventWriter(
                    build_id=self.stored_build.uuid,
                    space_name=self.stored_build.space_name,
                )
                subs = writer.start()
                if not subs:
                    return
                self._webhook_writer = writer
                logger.info(
                    "[BuildRunner] Webhook writer initialized with %d subscription(s)",
                    len(subs),
                )

            self._webhook_writer.accept_event(event)
        except Exception as e:
            logger.warning("[BuildRunner] Webhook persist error (non-fatal): %s", e)

    def __process_terminate_event(self: Self, event: BuildEvent) -> None:
        build_id = event.run_metadata.build_id
        assert build_id is not None, f"build_id is missing: {event}"
        assert (
            build_id == self.stored_build.uuid
        ), f"got an unknown build: {build_id} {event}"
        payload = event.payload
        assert isinstance(payload, BuildEventTerminatePayload), "payload is invalid"
        self.build_message_logger.info(markdown=payload.msg, triggering_event=event)

    def __process_message_event(self: Self, event: BuildEvent) -> None:
        run_meta = event.run_metadata
        build_id = run_meta.build_id
        assert build_id is not None, f"build_id is missing: {event}"
        assert (
            build_id == self.stored_build.uuid
        ), f"got an unknown build: {build_id} {event}"
        payload = event.payload
        assert isinstance(payload, BuildEventMessagePayload), "payload is invalid"
        target_name = run_meta.target_name
        step_uri = run_meta.targetstep_uri
        step_id = run_meta.targetsteprun_id
        markdown = f"""## ℹ️ Message Event

```
Target Name : {target_name}
Type        : step
Step URI    : {step_uri}
Step ID     : {step_id}
Build ID    : {build_id}
```

{payload.msg}
"""
        self.build_message_logger.info(markdown=markdown, triggering_event=event)

    def __process_build_status_event(self: Self, event: BuildEvent) -> bool:
        payload = event.payload
        run_info = event.run_metadata
        build_id = run_info.build_id
        assert (
            build_id == self.stored_build.uuid
        ), f"got an unknown build: {build_id} {event}"
        assert isinstance(build_id, str)
        build_finished = False
        assert isinstance(
            payload, BuildEventStatusPayload
        ), f"expected a status payload, actual: {payload}"
        logger.info("got a status: %s and a message: %s", payload.status, payload.msg)
        logger.info("run_info: %s", run_info)
        if self.build_run is None:
            logger.warning(
                "no run found for the build %s , assuming failure on creation", build_id
            )
            if run_info.type != "Build":
                logger.info("not a build status event: %s, ignoring...", run_info.type)
                return build_finished
            self.build_message_logger.info(markdown=payload.msg, triggering_event=event)
            build_finished = self.__process_build_info_type_event(event=event)
            return build_finished
        build_run = self.build_run
        logger.info("build_run: %s", build_run)
        self.build_message_logger.info(markdown=payload.msg, triggering_event=event)
        if run_info.type == "Build":
            build_finished = self.__process_build_info_type_event(event=event)
        elif run_info.type == "Target":
            self.__process_build_target_info_type_event(event=event)
        elif run_info.type == "TargetStep":
            self.__process_build_target_step_info_type_event(event=event)
        else:
            logger.error("unsupported run info type: %s", run_info)
        return build_finished

    def __process_artifact_event(
        self: Self, event: BuildEvent, pushed: bool = False
    ) -> None:
        logger.info("artifact event: %s", event)
        payload = event.payload
        if pushed:
            assert isinstance(
                payload, ArtifactPushedEventPayload
            ), f"expected ArtifactPushedEventPayload, actual: {payload}"
        else:
            assert isinstance(
                payload, CreatedArtifactEventPayload
            ), f"expected CreatedArtifactEventPayload, actual: {payload}"
        run_info = event.run_metadata
        logger.info("run_info: %s", run_info)
        build_id = run_info.build_id
        assert build_id is not None, f"build_id is missing: {event}"
        assert (
            build_id == self.stored_build.uuid
        ), f"got an unknown build: {build_id} {event}"
        stored_build = self.stored_build
        target_id = run_info.targetrun_id
        logger.info(
            "got an artifact %s for build %s target %s",
            payload.uri,
            build_id,
            target_id,
        )
        build_run = self.build_run
        logger.info("build_run: %s", build_run)
        username = stored_build.username
        logger.debug("GitHub username: %s", username)
        assert payload.uri is not None, "the artifact URI is None"
        logger.info("parsing the artifact uri: %s", payload.uri)
        # type = get_artifact_type(payload.uri)
        normalized_uri = BuildRunner._get_normalized_uri(payload.uri)
        artifact_type = get_artifact_type(normalized_uri)
        if not pushed:
            # Check if already registered: the same URI can arrive twice when a
            # step is retried (original job writes it before bkill, retried job
            # writes it again).  Reuse the existing record rather than inserting
            # a new one with a different UUID, which would violate the
            # (uri, space_name) unique constraint.
            # NOTE: this also means we should never use the artifact UUID for lookups and
            # instead only use the uri+spacename (as is done here).
            existing = self.storage.artifact_registry.get_by_uri(
                uri=normalized_uri, space_name=stored_build.space_name
            )
            if existing is not None:
                # Confirm that the artifact is really a retry from this same build.
                assert isinstance(existing, ArtifactRegistration)
                assert (
                    existing.created_by_build_id == build_id
                ), "Same artifact URI found for another build"
                assert (
                    existing.created_by_target_id == target_id
                ), "Same artifact URI found for another target"
                artifact = existing
                artifact.status = ArtifactRegistrationStatus.PENDING
            else:
                artifact = ArtifactRegistration(
                    uri=normalized_uri,
                    space_name=stored_build.space_name,
                    username=username,
                    name=payload.binding_id,
                    type=artifact_type,
                    created_by_build_id=build_id,
                    created_by_target_id=target_id,
                    created_at=event.timestamp,
                    status=ArtifactRegistrationStatus.PENDING,
                )
            logger.info("registering the artifact as pending: %s", artifact)
            self.storage.artifact_registry.update(artifact)
            self.__update_target_with_artifact(event=event, artifact=artifact)
        else:
            art_store = self.storage.artifact_registry
            _art_result = art_store.get_by_uri(
                uri=normalized_uri, space_name=stored_build.space_name
            )
            assert isinstance(
                _art_result, ArtifactRegistration
            ), f"failed to find an artifact registered for uri: {normalized_uri}"
            artifact = _art_result
            artifact.status = ArtifactRegistrationStatus.SUCCESS
            logger.info("updating the artifact as success: %s", artifact)
            _updated_art = self.storage.artifact_registry.update_fields(
                artifact.uuid, {"status": artifact.status}
            )
            assert isinstance(_updated_art, ArtifactRegistration)
            artifact = _updated_art
        icon = "📦" if pushed else "⬆️"
        download_msg = (
            f"gb artifact download -d <output directory> {normalized_uri}"
            if pushed
            else "(not available yet)"
        )
        body = f"""{icon}  New artifact {'push completed' if pushed else 'is being pushed...'}:
```
Status   : {artifact.status}
ID       : {artifact.uuid}
G.B URI  : {normalized_uri}
Download : {download_msg}
```
"""
        self.build_message_logger.info(markdown=body, triggering_event=event)

    # Left as protected method (i.e. with single _) for access by tests
    def _should_retry(self, stored_build: StoredBuild) -> bool:
        """Return True if the failed build is eligible for an automatic retry.

        A build is retryable when:
          - Its status is FAILED.
          - The build config specifies max_retries > 0.
          - The number of retries already attempted is below max_retries.
        """
        if stored_build.status != Status.FAILED:
            return False
        try:
            build_config = stored_build.get_build_config()
        except Exception as e:
            logger.warning(
                "Could not read build config for build %s, skipping retry: %s",
                stored_build.uuid,
                e,
            )
            return False
        if build_config.retries.max_retries <= 0:
            return False
        return stored_build.retry_count < build_config.retries.max_retries

    @staticmethod
    def _get_normalized_uri(uri_str: str) -> str:
        """Let the URI normalize itself, as necessary.
        For example, LH uris always make sure there is a model/fileset revision/version.
        """
        uri = URI.get_uri(uri_str)
        uri_str = uri.get_uristr(uri)
        return uri_str

    def __update_target_with_artifact(
        self: Self,
        event: BuildEvent,
        artifact: ArtifactRegistration,
    ) -> None:
        run_info = event.run_metadata
        target_id = run_info.targetrun_id
        assert isinstance(target_id, str)
        binding_id = self.__get_output_target_name(event)
        assert binding_id is not None and isinstance(
            binding_id, str
        ), f"binding_id {binding_id} was not a string"
        assert len(binding_id) > 0, f"binding id {binding_id} was an empty string"
        output_artifacts = {binding_id: [artifact.uuid]}
        target = self.storage.target_storage.get_by_uuid(target_id)
        if target is None:
            # TODO: Not sure we ever get here since it seems like we call this on an earlier event.
            logger.info("target %s doesn't exist, creating...", target_id)
            self.__create_and_store_target_run(
                event=event, output_artifacts=output_artifacts
            )
        else:
            logger.info("updating the target %s with artifact: %s", target_id, artifact)
            assert isinstance(target, StoredTargetRun)
            output_artifacts = self.__merge_output_artifacts(
                target.output_artifacts, output_artifacts
            )
            target.output_artifacts = output_artifacts
            self.storage.target_storage.update(target)

    def __get_output_target_name(self: Self, event: BuildEvent) -> Optional[str]:
        payload = event.payload
        if event.type is BuildEventType.ARTIFACT_PUSHED_EVENT:
            assert isinstance(
                payload, ArtifactPushedEventPayload
            ), f"expected ArtifactPushedEventPayload, actual: {payload}"
        else:
            assert isinstance(
                payload, CreatedArtifactEventPayload
            ), f"expected CreatedArtifactEventPayload, actual: {payload}"
        return payload.binding_id

    def __merge_output_artifacts(
        self: Self, a1: dict[str, list[str]], a2: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        out = {}
        processed_keys = []
        for key, v1 in a1.items():
            processed_keys.append(key)
            assert isinstance(v1, list), "Expected a list of output uuids here"
            v2 = a2.get(key, None)
            if v2 is not None:
                assert isinstance(v2, list), "Expected a list of output uuids here"
                v1 = list(set(v1 + v2))
            out[key] = v1

        for key, v2 in a2.items():
            assert isinstance(v2, list), "Expected a list of output uuids here"
            if not key in processed_keys:
                out[key] = v2.copy()

        return out

    def __create_and_store_target_run(
        self: Self,
        event: BuildEvent,
        status: Optional[Status] = None,
        input_artifacts: Optional[dict[str, str]] = None,
        output_artifacts: Optional[dict[str, list[str]]] = None,
    ) -> StoredTargetRun:
        run_info = event.run_metadata
        build_id = run_info.build_id
        assert isinstance(build_id, str)
        target_id = run_info.targetrun_id
        build_run = self.build_run
        assert build_run is not None, "build_run is None"
        build = build_from_build_run(build_run)
        build_config = build.config
        assert isinstance(build_config, BuildConfig)
        target_name = run_info.target_name
        assert isinstance(target_name, str)
        my_build_target = build.targets[target_name]
        env_asset = my_build_target.environment.environment_asset
        environment_uri = ""
        if env_asset is None:
            logger.warning("the environment_asset is None")
        else:
            environment_uri = env_asset.uristr
        stored_target = StoredTargetRun(
            uuid=target_id,
            build_id=build_id,
            environment_uri=environment_uri,
            name=target_name,
            # target_hash intentionally omitted — written only on SUCCESS via UPDATE
        )
        if status is not None:
            stored_target.status = status
        if input_artifacts is None:
            input_artifacts = self.__get_target_input_ids_from_event(
                event, target_name, warn_no_inputs=True, auto_register=True
            )
        logger.info("set  input artifact ids to %s", input_artifacts)
        stored_target.input_artifacts = input_artifacts
        if output_artifacts is not None:
            stored_target.output_artifacts = self.__merge_output_artifacts(
                stored_target.output_artifacts, output_artifacts
            )
            logger.info("set output artifacts to %s", stored_target.output_artifacts)

        self.storage.target_storage.add(stored_target)
        return stored_target

    def __update_stored_target_run(
        self: Self,
        stored_target_run: StoredTargetRun,
        event: BuildEvent,
        input_artifacts: dict[str, str],
    ) -> None:
        """Update an existing StoredTargetRun in storage from a status event.

        Sets started_at/finished_at timestamps on PENDING→RUNNING and RUNNING→* transitions,
        merges input artifacts, updates status, and writes target_hash only on SUCCESS
        (keeping the partial unique index invariant that only successful runs hold a hash).
        """
        payload = event.payload
        assert isinstance(payload, BuildEventStatusPayload)
        logger.info("stored_target_run %s", stored_target_run)
        if (
            stored_target_run.status is Status.PENDING
            and payload.status is Status.RUNNING
        ):
            logger.info("target started running at %s", event.timestamp)
            stored_target_run.started_at = event.timestamp
        if (
            stored_target_run.status is Status.RUNNING
            and payload.status is not Status.RUNNING
        ):
            logger.info("target started finished running at %s", event.timestamp)
            stored_target_run.finished_at = event.timestamp
        stored_target_run.status = payload.status
        for key, item in input_artifacts.items():
            stored_target_run.input_artifacts[key] = item
        # Only record target_hash on SUCCESS — non-successful runs have no meaningful hash
        if event.run_metadata.target_hash and payload.status == Status.SUCCESS:
            stored_target_run.target_hash = event.run_metadata.target_hash
        self.storage.target_storage.update(stored_target_run)

    def __get_retry_chain_build_ids(self: Self) -> list[str]:
        """Return all build UUIDs in the retry chain of the current build, from the current
        build back to the original (root) build, by following retry_of_build_id links.
        """
        build_ids = [self.stored_build.uuid]
        current_id = self.stored_build.retry_of_build_id
        while current_id:
            build_ids.append(current_id)
            ancestor = self.storage.build_storage.get_by_uuid(current_id)
            if not isinstance(ancestor, StoredBuild):
                break
            current_id = ancestor.retry_of_build_id
        return build_ids

    def __is_target_already_run(
        self: Self,
        target_hash: str,
    ) -> Optional[tuple[str, dict[str, list[str]]]]:
        """Return (target_uuid, resolved_outputs) if a prior successful run with the same
        target_hash exists within the current retry chain and all its output artifacts are
        fully registered, otherwise None. Only searches within the retry chain so that
        targets from unrelated builds are never skipped."""
        chain_build_ids = self.__get_retry_chain_build_ids()
        results = self.storage.target_storage.get_by_where(
            {
                "target_hash": target_hash,
                "status": Status.SUCCESS.name,
                "build_id": chain_build_ids,
            }
        )
        if not results:
            return None
        stored_target = results[0]
        logger.info(
            "Found previously run target %s from a retried/previous build %s",
            stored_target.uuid,
            stored_target.build_id,
        )
        resolved_outputs: dict[str, list[str]] = {}
        for binding_id, artifact_uuids in stored_target.output_artifacts.items():
            uris = []
            for artifact_uuid in artifact_uuids:
                artifact = self.storage.artifact_registry.get_by_uuid(artifact_uuid)
                if artifact is None:
                    logger.warning(
                        "target_already_run: artifact %s not found for target %s — not skipping",
                        artifact_uuid,
                        stored_target.uuid,
                    )
                    return None
                assert isinstance(artifact, ArtifactRegistration)
                if artifact.status != ArtifactRegistrationStatus.SUCCESS:
                    logger.warning(
                        "target_already_run: artifact %s has status %s for target %s — not skipping",
                        artifact_uuid,
                        artifact.status,
                        stored_target.uuid,
                    )
                    return None
                uris.append(artifact.uri)
            if uris:
                resolved_outputs[binding_id] = uris
        return (stored_target.uuid, resolved_outputs)

    def __get_target_input_ids_from_event(
        self: Self,
        event: BuildEvent,
        target_name,
        warn_no_inputs: bool = False,
        auto_register: bool = True,
    ) -> dict[str, str]:
        """Get the dictionary of target input names keyed to values of target input uuids.

        Args:
            event (BuildEvent): _description_
            target_name (_type_): _description_
            warn_no_inputs (bool, optional): _description_. Defaults to False.

        Returns:
            dict[str,str]: target input name mapped to uuid of a registered artifact.
        """
        input_artifacts = {}
        assert isinstance(event.payload, BuildEventStatusPayload)
        if event.payload.metadata is not None:
            if "inputs" in event.payload.metadata:
                target_inputs = event.payload.metadata["inputs"]
                assert isinstance(target_inputs, dict)
                input_artifacts = self.__get_target_input_ids(
                    target_name=target_name,
                    target_inputs=target_inputs,
                    auto_register=auto_register,
                )
            elif warn_no_inputs:
                logger.warning(
                    "the event payload is missing inputs for the target %s : %s",
                    target_name,
                    event,
                )

        return input_artifacts

    def __get_target_input_ids(
        self, target_name: str, target_inputs: dict[str, str], auto_register: bool
    ) -> dict[str, str]:
        """Get the uuids for the artifacts referenced in the dictionary of artifact uris.

        Args:
            target_name(str): name of the target, used only in exceptions.
            target_inputs (dict[str,str]): dictionary of input names to uris of artifacts

        Raises:
            ValueError: if a uri is not registered.
            ValueError: if a uri is registered more than once.

        Returns:
            dict[str, str]: dictionary of artifact uuids for each artifact.  keyed by the input name.  Never None.
        """
        input_artifacts = {}
        for input_name, unnormalized_uri in target_inputs.items():
            uri = BuildRunner._get_normalized_uri(unnormalized_uri)
            logger.info("unnormalized_uri: %s uri %s", unnormalized_uri, uri)
            # a list is always returned.
            items = self.storage.artifact_registry.get_by_where(
                {"uri": uri, "space_name": self.stored_build.space_name}
            )
            item = None
            if len(items) == 0:
                if auto_register:
                    item = self.__register_input_artifact(input_name, uri)
                else:
                    raise ValueError(
                        f"Registration not found for target {target_name} in space {self.stored_build.space_name}, input {input_name}={uri}"
                    )
            elif len(items) > 1:
                raise ValueError(
                    f"Target input uri {uri} has {len(items)} registrations"
                )
            else:
                item = items[0]
            assert isinstance(item, ArtifactRegistration)
            artifact_id = item.uuid
            logger.info("Adding artifact_id %s to target inputs", artifact_id)
            input_artifacts[input_name] = artifact_id

        return input_artifacts

    def __register_input_artifact(
        self, input_name: str, uri: str
    ) -> ArtifactRegistration:
        """Register a URI-based input artifact and get the registered ArtifactRegistration"""
        artifact_type = get_artifact_type(uri)
        artifact = ArtifactRegistration(
            uri=uri,
            space_name=self.stored_build.space_name,
            username=self.stored_build.username,
            name=input_name,
            type=artifact_type,
            created_by_build_id="",
            created_by_target_id="",
            created_at=time.time(),
            status=ArtifactRegistrationStatus.SUCCESS,
        )
        self.storage.artifact_registry.add(artifact)
        return artifact

    def __process_build_info_type_event(self: Self, event: BuildEvent) -> bool:
        build_id = event.run_metadata.build_id
        assert isinstance(build_id, str)
        payload = event.payload
        assert isinstance(payload, BuildEventStatusPayload)

        status = payload.status
        failure_reason = ""
        if status is Status.FAILED:
            failure_reason = (
                payload.msg if payload.msg else "got a build failed status event!"
            )

        # Update the build status as the last thing so JobStats and PR are updated before declaring the build complete - tests expect this.
        valid_status_values = [Status.PENDING, Status.RUNNING]
        valid_status = lambda item: item.status in valid_status_values
        updated = self.__update_stored_build_status(
            status, failure_reason=failure_reason, unfinished_should_update=valid_status
        )

        if not updated:
            logger.warning(
                "Build %s status update to status %s failed (may have been cancelled or deleted)",
                self.stored_build.uuid,
                str(status),
            )
            push_failed_status_update_metric(
                self.stored_build.uuid, valid_status_values
            )
        elif status == Status.SUCCESS:
            logger.info(
                "post the build status/lineage link once again upon build success..."
            )
            build_status_link = get_build_status_link(build_id=build_id)
            body = f"🥳🎉 Build status and lineage: {build_status_link}"
            self.build_message_logger.info(markdown=body, triggering_event=event)

        build_finished = status.is_finished()
        return build_finished

    def __update_stored_build_status(
        self: Self,
        status: Status,
        failure_reason: str = "",
        unfinished_should_update: Optional[Callable[[StoredBuild], bool]] = None,
    ) -> Optional[StoredBuild]:
        """Update the inmemory and instorage StoredBuild with the new status, with special handling
        for finished status value to update targets and steps in the associated build_run, if present.

        Args:
            status (Status): _description_
            failure_reason (str): applied if status is FAILED
            unfinished_should_update: use only for unfinished status values

        Return the updated build if the update was successful or None
        """
        if self.stored_build.status is status:
            return None
        logger.info(
            "Updating status %s->%s for build %s",
            self.stored_build.status,
            status,
            self.stored_build.uuid,
        )
        if (
            status == Status.RUNNING
            and self.__get_stored_build_status(self.stored_build.uuid)
            == Status.CANCEL_REQUESTED
        ):
            return None  # Don't do anything and process cancellation from the worker event loop.
        # Update the PR for failed and cancelled builds
        if status is Status.FAILED:
            body = f"{STATUS_TO_ICON[Status.FAILED]} Build failed."
        elif status is Status.CANCELLED:
            body = f"{STATUS_TO_ICON[Status.CANCELLED]} Build cancelled."
        else:
            body = None
        if body is not None:
            self.build_message_logger.info(markdown=body)
            # self.build_event_logger.info(markdown=body)

        # If the build is finished, use finalize_build_status to update targets, steps, artifacts, and the build itself.
        if status.is_finished():
            build = finalize_build_status(
                self.stored_build.uuid, status, failure_reason
            )
            if failure_reason:
                self.stored_build.failure_reason = failure_reason
                self.build_message_logger.error(failure_reason)
        else:
            # For non-finished statuses, just update the build status directly
            build = update_stored_build_status(
                self.stored_build.uuid,
                status,
                "",
                should_update=unfinished_should_update,
            )

        if build:
            # Update the in-memory stored_build to reflect the new status
            self.stored_build = build

        return build

    def __get_stored_build_status(self: Self, build_id: str) -> Status:
        build = self.storage.build_storage.get_by_uuid(build_id)
        assert build, f"Did not find build with id {build_id}"
        return build.status  # type: ignore[union-attr]

    def __process_build_target_info_type_event(self: Self, event: BuildEvent) -> None:
        logger.info("run_info is a Target")
        run_info = event.run_metadata
        payload = event.payload
        assert isinstance(payload, BuildEventStatusPayload)
        build_id = run_info.build_id
        assert isinstance(build_id, str)
        build_run = self.build_run
        assert build_run is not None, "build_run is None"
        build = build_from_build_run(build_run)
        build_target_name = run_info.target_name
        assert isinstance(build_target_name, str)
        build_target = build.targets[build_target_name]
        build_target_config = build_target.config
        assert isinstance(build_target_config, BuildTargetConfig)
        build_config = build.config
        assert isinstance(build_config, BuildConfig)
        targetrun_id = run_info.targetrun_id
        assert isinstance(targetrun_id, str)
        stored_target_run = self.storage.target_storage.get_by_uuid(targetrun_id)
        input_artifacts = self.__get_target_input_ids_from_event(
            event=event,
            target_name=build_target_name,
            warn_no_inputs=True,
            auto_register=True,
        )
        if stored_target_run is None:
            stored_target_run = self.__create_and_store_target_run(
                event=event,
                status=payload.status,
                input_artifacts=input_artifacts,
            )
            assert stored_target_run.uuid == targetrun_id
        else:
            assert isinstance(stored_target_run, StoredTargetRun)
            self.__update_stored_target_run(
                stored_target_run=stored_target_run,
                event=event,
                input_artifacts=input_artifacts,
            )
        if run_info.skipped_for_prerun_target_id:
            stored_target_run.skipped_for_prerun_target_id = (
                run_info.skipped_for_prerun_target_id
            )
            self.storage.target_storage.update(stored_target_run)
        if payload.status == Status.SUCCESS:
            # Target complete - record lineage here
            try:
                logger.info("create job stats for completed target %s", targetrun_id)
                get_lineage_store().add_jobstats_for_build_target(
                    self.storage,
                    build_id=build_id,
                    target_id=targetrun_id,
                )
            except Exception as e:
                logger.warning(
                    "failed to create job stats for completed build %s: %s",
                    build_id,
                    e,
                )

    def __process_build_target_step_info_type_event(self: Self, event: BuildEvent):
        logger.info("run_info is a TargetStep")
        payload = event.payload
        assert isinstance(payload, BuildEventStatusPayload)
        run_info = event.run_metadata
        build_id = run_info.build_id
        assert isinstance(build_id, str)
        build_run = self.build_run
        assert build_run is not None, "build_run is None"
        build = build_from_build_run(build_run)
        build_target_name = run_info.target_name
        assert isinstance(build_target_name, str)
        build_target = build.targets[build_target_name]
        build_target_config = build_target.config
        assert isinstance(build_target_config, BuildTargetConfig)
        found_step = None
        for step in build_target_config.steps:
            if step.step_uri == run_info.targetstep_uri:
                found_step = step
                break
        found_step_config = {} if found_step is None else found_step.config
        build_config = build.config
        assert isinstance(build_config, BuildConfig)
        targetsteprun_id = run_info.targetsteprun_id
        assert isinstance(targetsteprun_id, str)
        _step_result = self.storage.step_storage.get_by_uuid(targetsteprun_id)
        if _step_result:
            assert isinstance(_step_result, StoredStepRun)
            stored_step_run: StoredStepRun = _step_result
        else:
            stored_step_run = StoredStepRun(
                uuid=targetsteprun_id,
                build_id=build_id,
                target_id=run_info.targetrun_id,
                definition_uri=run_info.targetstep_uri,
                config=found_step_config,
                status=payload.status,
                status_msg=payload.msg,
                started_at=event.timestamp,
            )
        if (
            stored_step_run.status is Status.PENDING
            and payload.status is Status.RUNNING
        ):
            logger.info("step started running at %s", event.timestamp)
            stored_step_run.started_at = event.timestamp
        elif (
            stored_step_run.status is Status.RUNNING
            and payload.status is not Status.RUNNING
        ):
            logger.info("step started finished running at %s", event.timestamp)
            stored_step_run.finished_at = event.timestamp
        stored_step_run.status = payload.status
        stored_step_run.status_msg = payload.msg
        logger.info("stored_step_run %s", stored_step_run)
        assert (
            stored_step_run.uuid == targetsteprun_id
        ), f"expected {targetsteprun_id} actual {stored_step_run.uuid}"
        logger.info("creating/updating the step: %s", stored_step_run)
        self.storage.step_storage.update(stored_step_run)
