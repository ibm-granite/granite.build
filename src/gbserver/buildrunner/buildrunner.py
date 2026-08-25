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
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Self, Union

from gbcommon.uri.git import GitURI
from gbcommon.uri.uri import URI
from gbcommon.uri.utils import get_artifact_type
from gbserver.build.build import Build
from gbserver.build.buildrun import BuildRun
from gbserver.build.run import RunFailed
from gbserver.build.space import Space
from gbserver.buildrunner.abstractbuildrunner import AbstractBuildRunner
from gbserver.buildrunner.build_setup import BuildSetup
from gbserver.buildrunner.build_utils import (
    finalize_build_status,
    push_failed_status_update_metric,
    update_stored_build_status,
)
from gbserver.buildrunner.buildlogger import (
    get_message_logger,
)
from gbserver.github.myghapi import MyGHApi
from gbserver.metrics.metrics_client import push_metrics
from gbserver.storage.artifact_registration import (
    ArtifactRegistration,
    ArtifactRegistrationStatus,
)
from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_event import StoredEvent
from gbserver.storage.stored_step_run import StoredStepRun
from gbserver.storage.stored_target_run import (
    StoredTargetRun,
    latest_finished_target,
)
from gbserver.types.artifact import ArtifactType
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
    StepMetadataUpdateEventPayload,
)
from gbserver.types.constants import (
    DEFAULT_DIR_PERMS,
    DEFAULT_GH_API_ENDPOINT,
    DEFAULT_ROOT_WORKSPACE_DIR,
    GBSERVER_GITHUB_TOKEN,
    GBSERVER_RAISE_BUILD_EXCEPTIONS,
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
# Timezone-aware epoch used to order runs by a timestamp that may be unset.
# get_time() stamps finished_at as tz-aware, so the fallback must be tz-aware too
# or max() raises "can't compare offset-naive and offset-aware datetimes".
_SORT_EPOCH = datetime.fromtimestamp(0, tz=timezone.utc)


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
        self.build_run = None
        self.stop_event = threading.Event()
        # Set by the public stop()/stop_and_fail() entrypoints to signal an
        # explicit external termination. The start_and_wait retry loop checks this
        # to break instead of spawning a retry. Distinct from stop_event (which is
        # the worker-task lifecycle signal, set after every run and cleared between
        # retries) and from __is_build_cancelled() (which only detects a CANCELLED/
        # CANCEL_REQUESTED status): stop_and_fail marks the build FAILED, which is
        # retryable, so without this flag a SIGTERM would spawn a retry.
        self._stop_requested = threading.Event()
        self.build_message_logger = get_message_logger(
            build, _BUILD_EVENT_SOURCE_NAME
        )  # To be recreated later.
        self.event_storage = get_admin_storage().event_storage
        # Serializes status transitions (__update_stored_build_status). The build
        # runs on the worker thread while stop()/stop_and_fail() run on another
        # thread; without this, the worker's natural finalize (e.g. a concurrent
        # SUCCESS) can interleave with a stop-driven finalize and leave the build
        # and its targets/steps/artifacts in disagreeing states (the build status
        # is written unconditionally, but entity finalization only touches
        # unfinished entities).
        self._finalize_lock = threading.Lock()
        # Buffers step metadata (targetsteprun_id -> {key: value}) pushed by a
        # STEP_METADATA_UPDATE_EVENT that is processed before the status event which
        # creates the StoredStepRun row. These come from different producers (log
        # parsing vs. step lifecycle) with no ordering guarantee, so the value is held
        # here and flushed onto the row once it exists (see _apply_pending_step_metadata),
        # rather than dropped. The single serial worker loop is the only accessor, so
        # no lock is needed.
        self._pending_step_metadata: dict[str, dict[str, str]] = {}

    def stop(self: Self) -> None:
        """Stop the building thread if it was started."""
        logger.debug("BuildRunner.stop start")
        self._stop_requested.set()
        self.__cancel_build_run()
        logger.debug("BuildRunner.stop end")

    def stop_and_fail(self: Self, failure_reason: str) -> None:
        """Stop the running build and mark it FAILED.

        Public entrypoint intended to be called from another thread (e.g. a
        SIGTERM handler) while start_and_wait() runs. Flags the run as an explicit
        termination (so the retry loop does not treat the resulting FAILED status
        as a retryable failure and spawn a new run) and delegates the FAILED-then-
        cancel work to __cancel_and_fail_build (see its docstring for the ordering
        and locking that keep the terminal status consistent).

        Args:
            failure_reason (str): reason recorded on the build for the failure.
        """
        logger.debug("BuildRunner.stop_and_fail start")
        self._stop_requested.set()
        self.__cancel_and_fail_build(failure_reason=failure_reason)
        logger.debug("BuildRunner.stop_and_fail end")

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
            while True:
                if self.stored_build.status == Status.FAILED:
                    logger.info(
                        "Build %s is already FAILED; skipping run and proceeding to retry",
                        self.stored_build.uuid,
                    )
                else:
                    logger.info(
                        "Starting new build run for build %s",
                        self.stored_build.uuid,
                    )

                    asyncio.run(self.__async_run_build())

                    logger.info(
                        "Build run completed (build_id=%s)",
                        self.stored_build.uuid,
                    )

                # Stop if a cancellation was requested for this build. This runs
                # after every attempt regardless of how it finished — including the
                # exception path that bypasses the worker task — so the build is
                # marked CANCELLED and no further retry is created.
                if self.__is_build_cancelled():
                    self.__cancel_build_run()
                    break

                # An explicit termination via stop()/stop_and_fail() (e.g. a
                # SIGINT/SIGTERM handler) must not be retried. stop_and_fail marks
                # the build FAILED — which is otherwise retryable and is NOT caught
                # by __is_build_cancelled() above — so break here to end the chain.
                if self._stop_requested.is_set():
                    logger.info(
                        "Termination requested for build %s; not retrying",
                        self.stored_build.uuid,
                    )
                    break

                retry_build = self.__prepare_retry()
                if retry_build is None:
                    break
                # In-place retry: same build id, re-read to pick up the bumped
                # retry_count and RUNNING status. The message logger is keyed
                # on the build id, so it need not be recreated.
                self.stored_build = retry_build
                # Notify on the build's PR that an automatic retry is starting.
                # This must happen here: on the re-run __setup short-circuits
                # (source_uri is already set from the first attempt) and so never
                # reaches its own comment call. (That call still covers the rare
                # case where the first attempt failed before a PR was created.)
                self.__add_retry_comment_to_pr()
                self.stop_event.clear()

        except Exception as e:
            logger.error("build execution failed for build ID %s", build_id)
            raise e

    def __prepare_retry(self: Self) -> Optional[StoredBuild]:
        """If the finished build is eligible for retry, reset it in place for another attempt.

        The build keeps its same uuid: retry_count is bumped, the status is moved
        to RUNNING, and the failure_reason is cleared. No new StoredBuild is
        created.

        Returns the same (re-read) build ready to re-run, or None if no retry
        should occur.
        """
        latest = self.storage.build_storage.get_by_uuid(self.stored_build.uuid)
        if latest is None or not isinstance(latest, StoredBuild):
            return None
        if not self._should_retry(latest):
            return None
        # RUNNING (not PENDING): this build is re-run in place by the in-process
        # retry loop below, so it is genuinely in flight. A non-PENDING status also
        # keeps the BuildWatcher (which polls SUBMITTED/PENDING) from dispatching a
        # duplicate runner for it.
        #
        # Guard the flip on the build still being FAILED: a cancel arriving in the
        # brief FAILED window (see api.builds.request_cancellation) sets the build to
        # CANCELLED, and we must not overwrite that terminal state with RUNNING.
        # If the guard rejects the update, honor the cancel by not retrying.
        refreshed = self.storage.build_storage.update_fields(
            latest.uuid,
            {
                "retry_count": latest.retry_count + 1,
                "status": Status.RUNNING,
                "failure_reason": "",
            },
            should_update=lambda item: item.status == Status.FAILED,
        )
        if refreshed is None:
            # The guard rejected the flip: the build is no longer FAILED, which in
            # practice means a cancel landed in the FAILED window and set it to
            # CANCELLED. Honor that by not retrying.
            logger.info(
                "Build %s no longer FAILED (likely cancelled); skipping retry",
                latest.uuid,
            )
            return None
        logger.info(
            "Build %s failed; retrying in place (attempt %d of %d)",
            refreshed.uuid,
            refreshed.retry_count,
            refreshed.get_build_config().retries.max_retries,
        )
        return refreshed

    def __add_retry_comment_to_pr(self: Self) -> None:
        """Post a retry-attempt comment on this build's own PR.

        With in-place retry there is no separate "original" build; the build keeps
        its uuid and source_uri across attempts, so the comment is posted on its
        own PR.
        """
        if not self.stored_build.source_uri:
            return
        owner, repo, pr_id = self.stored_build.get_pr_info()
        if not pr_id:
            return
        try:
            myapi = MyGHApi(token=self.gh_token, owner=owner, repo=repo)
            build_config = self.stored_build.get_build_config()
            body = (
                f"Build failed (attempt {self.stored_build.retry_count} of "
                f"{build_config.retries.max_retries}). Automatically retrying."
            )
            myapi.update_issue_comment(body=body, pr_id=pr_id)
        except Exception as e:
            logger.warning("Failed to post retry comment on PR %s: %s", pr_id, e)

    async def __async_run_build(self) -> None:
        """Run the build: set up the workspace, extract the archive, and run all
        targets (targets already succeeded in this build are skipped by
        target_already_run_fn, not by resuming a prior runner)."""

        stored_build = self.stored_build
        build_id = stored_build.uuid

        logger.info("Running build %s", build_id)

        event_q: Queue[Event] = Queue()

        try:

            try:
                space = self.__get_build_space(stored_build)
            except ValueError as e:
                err_stack = traceback.format_exc()
                logger.error("%s", err_stack)
                self.__update_stored_build_status(
                    status=Status.INVALID, failure_reason=str(err_stack)
                )
                self.build_message_logger.error(
                    markdown=f"build `{build_id}` status `{self.stored_build.status}`, error: {e}"
                )
                return

            workspace_dir = self.workspace_dir / WORKSPACE_BUILDS_DIR
            workspace_dir.mkdir(mode=DEFAULT_DIR_PERMS, parents=True, exist_ok=True)

            if not self.__setup(space):  # type: ignore[arg-type]
                return  # Log messages were issued.

            build_dir = Path(tempfile.mkdtemp()) / build_id
            build_archive_bytes = stored_build.load_from_build_archive()
            extract_archive(build_archive_bytes, build_dir)

            build = Build(
                build_dir=build_dir,
                space=space,
                build_id=build_id,
                username=stored_build.username,
                workspace_dir=workspace_dir,
                event_q=event_q,
                targets=stored_build.targets,  # re-run all targets of the build
                force_fetch=True,
                target_already_run_fn=(
                    self.__is_target_already_run
                    if self.stored_build.get_build_config().retries.target_reuse_enabled
                    and self.__has_reusable_prior_runs()
                    else None
                ),
            )

            if build.val_errors and not build.val_errors.is_valid(check_warnings=True):
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

            logger.info("Build %s finished", build_id)

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
        # Don't update unless the build is still in a pre-PR state we own:
        # PENDING for a first run, RUNNING for a build-level retry run whose first
        # attempt failed before its PR was created (the retry loop set it RUNNING).
        # On the success path `updates` carries only source_uri, so this never
        # moves a RUNNING build backward.
        update = self.storage.build_storage.update_fields(
            self.stored_build.uuid,
            updates,
            should_update=lambda item: item.status in (Status.PENDING, Status.RUNNING),
        )
        if update is not None:  # Build had pending status, all good.
            self.stored_build = update
            success = True
            logger.info(
                "Build update succeeded.  Build status is now %s",
                self.stored_build.status,
            )
            if self.stored_build.retry_count and self.stored_build.source_uri:
                self.__add_retry_comment_to_pr()
        else:
            _build_result = self.storage.build_storage.get_by_uuid(
                self.stored_build.uuid
            )
            assert isinstance(_build_result, StoredBuild)
            self.stored_build = _build_result
            success = False
            logger.warning(
                "Build update failed.  Likely as the status was not %s/%s (currently %s).",
                Status.PENDING,
                Status.RUNNING,
                self.stored_build.status,
            )
            push_failed_status_update_metric(
                self.stored_build.uuid, [Status.PENDING, Status.RUNNING]
            )
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
        """Return True if a cancellation was requested for this build.

        Only the build's *status* is read from storage, since cancellation is set
        externally. Callers on the hot path (the worker loop) rate-limit how often
        they invoke this.
        """
        try:
            build = self.storage.build_storage.get_by_uuid(self.stored_build.uuid)
            if isinstance(build, StoredBuild) and build.status in (
                Status.CANCEL_REQUESTED,
                Status.CANCELLED,
            ):
                return True
        except Exception as e:
            logger.warning(
                "Exception checking if build %s is cancelled: %s. Assuming not cancelled.",
                self.stored_build.uuid,
                e,
            )
        return False

    async def __worker_task(self: Self, event_q: Queue[Event], build_id: str) -> None:
        logger.debug("BuildRunner.__worker_task start build_id: %s", build_id)
        build_finished = False
        # Rate-limit the cancellation check below: 0.0 means "check on the first
        # iteration", then at most once per monitoring_interval thereafter.
        last_cancel_check_time = 0.0
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
                # Check for cancellation, but at most once per monitoring_interval
                # so a build emitting events rapidly doesn't read the build's
                # status from storage on every iteration.
                now = time.time()
                if now - last_cancel_check_time >= self.monitoring_interval:
                    last_cancel_check_time = now
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
        """Mark the build FAILED and stop its in-flight workload.

        Single implementation shared by the public stop_and_fail() (external
        SIGTERM-style termination) and the worker task's own exception path.

        FAILED is committed *before* cancellation is signalled: __cancel_build_run
        sets stop_event, after which the worker loop tries to write its own
        CANCELLED status, so writing FAILED first lets that later write be skipped
        by finalize_build_status's is_finished() guard. _finalize_lock (held inside
        __update_stored_build_status) additionally serializes this against a
        concurrent natural finalize (e.g. a SUCCESS completing at the same instant).

        Args:
            failure_reason (str): reason recorded on the build for the failure.
        """
        try:
            self.__update_stored_build_status(
                status=Status.FAILED, failure_reason=failure_reason
            )
        except Exception:
            logger.error("Could not mark build %s as failed", self.stored_build.uuid)

        try:
            # update_status=False: FAILED is already set above; this only tears
            # down the in-flight workload and sets stop_event.
            self.__cancel_build_run(update_status=False)
        except Exception:
            logger.error("Could not stop build %s", self.stored_build.uuid)

    def __cancel_build_run(self: Self, update_status: bool = True) -> None:
        """Cancel the in-progress build_run and mark this build CANCELLED.

        With in-place retry there is a single build id across attempts, so a
        cancellation requested on it cancels the active (possibly retrying) run and
        marks the one build CANCELLED.
        """

        self.stop_event.set()

        # Cancel the in-progress workload for the current build, if still running.
        if self.build_run is not None:
            status = self.build_run.status
            if status is Status.PENDING or status is Status.RUNNING:
                logger.info("cancelling the build %s", self.stored_build.uuid)
                self.build_run.cancel()
            else:
                logger.info(
                    "Not cancelling already-finished build_run:  build id = %s",
                    self.stored_build.uuid,
                )
        else:
            logger.warning("self.build_run is None")

        if not update_status:
            return

        # Cancel the current build if it is still in flight (stop() semantics).
        if not self.stored_build.status.is_finished():
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
        elif event.type is BuildEventType.STEP_METADATA_UPDATE_EVENT:
            self.__process_step_metadata_update_event(event=event)
        else:
            logger.error("unsupported event type: %s", event)
        logger.debug("BuildRunner.process_event end")
        return build_finished

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
        # Prefer the type implied by the URI scheme (e.g. hf/lh). When the scheme
        # implies nothing (e.g. env_local's env://), fall back to the type the
        # build.yaml output set on the event payload.
        artifact_type = get_artifact_type(normalized_uri)
        if artifact_type == ArtifactType.UNDEFINED and payload.type is not None:
            artifact_type = payload.type
        if not pushed:
            # Check if already registered: the same URI can arrive twice when a
            # step is retried or a target only partially succeeded and a build retry
            # is being attempted.  Reuse the existing record rather than inserting
            # a new one with a different UUID, which would violate the
            # (uri, space_name) unique constraint.
            # NOTE: this also means we should never use the artifact UUID for lookups and
            # instead only use the uri+spacename (as is done here).
            existing = self.storage.artifact_registry.get_by_uri(
                uri=normalized_uri, space_name=stored_build.space_name
            )
            if existing is not None:
                # Confirm the artifact really originates from this build. With
                # in-place retry the same URI can be re-emitted within one build by
                # a retried step or a re-run target, but never by another build.
                assert isinstance(existing, ArtifactRegistration)
                if existing.created_by_build_id != self.stored_build.uuid:
                    raise ValueError(
                        "Same artifact URI found in space %s for another build in "
                        "this space: %s" % (existing.space_name, existing.uri)
                    )
                logger.info("Reusing artifact (%s) from earlier run", existing.uuid)
                # Re-associate the artifact to the run that just (re-)produced it,
                # so it points at the successful target run rather than a prior
                # failed/reused one.
                if target_id and existing.created_by_target_id != target_id:
                    _reassoc = self.storage.artifact_registry.update_fields(
                        existing.uuid, {"created_by_target_id": target_id}
                    )
                    if isinstance(_reassoc, ArtifactRegistration):
                        existing = _reassoc
                artifact = existing
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
        return stored_build.has_retries_remaining()

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
            # A run whose first reported status is already terminal (e.g. a target
            # that fails immediately) is created here and never passes through
            # __update_stored_target_run, so apply the started_at/finished_at
            # transition rules now against the pre-transition (PENDING) status —
            # otherwise finished_at is never stamped and retry linkage cannot order
            # multiple failed attempts. Mirrors the step-run status handler.
            self._apply_run_timestamps(
                stored_run=stored_target,
                new_status=status,
                timestamp=event.timestamp,
                run_label="target",
            )
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

        # Link a re-run to the prior FAILED run of the same target in this build,
        # so the build reads as an honest FAILED->SUCCESS history in one build id.
        # A prior FAILED run can exist both from an in-place retry (retry_count > 0)
        # and from a restart, which reuses the same build id and resets
        # retry_count to 0 while leaving the pre-restart FAILED runs in place.
        # retry_count therefore cannot gate this out, so always look up; the query
        # is a single indexed read and returns "" on a genuine first attempt.
        stored_target.retry_of_target_id = self.__find_prior_failed_target_run(
            build_id=build_id, target_name=target_name
        )
        self.storage.target_storage.add(stored_target)
        return stored_target

    def __find_prior_failed_target_run(
        self: Self, build_id: str, target_name: str
    ) -> str:
        """Return the uuid of the most-recent FAILED run of ``target_name`` in ``build_id``.

        With ``max_retries >= 2`` a target may fail more than once before it
        succeeds, so several FAILED runs can coexist in the same build. The new
        SUCCESS run should link to the *latest* failed attempt, so pick the run
        with the greatest ``finished_at`` (get_by_where's ordering is undefined,
        so we order explicitly here rather than trusting the provider).

        Args:
            build_id: the build being run (same id across in-place retries).
            target_name: the target whose earlier failed run should be linked.

        Returns:
            The most-recent FAILED StoredTargetRun's uuid, or "" if none exists.
        """
        prior = self.storage.target_storage.get_by_where(
            {
                "build_id": build_id,
                "name": target_name,
                "status": Status.FAILED.name,
            }
        )
        if not prior:
            return ""
        # A terminal FAILED run always has finished_at stamped; fall back to the
        # tz-aware _SORT_EPOCH so a (defensively handled) unset value never wins
        # and never mixes naive/aware datetimes in the comparison.
        latest = max(prior, key=lambda run: run.finished_at or _SORT_EPOCH)
        return latest.uuid

    def __update_stored_target_run(
        self: Self,
        stored_target_run: StoredTargetRun,
        event: BuildEvent,
        input_artifacts: dict[str, str],
    ) -> None:
        """Update an existing StoredTargetRun in storage from a status event.

        Stamps started_at on first entry into RUNNING and finished_at on first
        entry into a terminal status (via the shared _apply_run_timestamps),
        merges input artifacts, updates status, and writes target_hash only on
        SUCCESS (keeping the partial unique index invariant that only successful
        runs hold a hash).
        """
        payload = event.payload
        assert isinstance(payload, BuildEventStatusPayload)
        logger.info("stored_target_run %s", stored_target_run)
        self._apply_run_timestamps(
            stored_run=stored_target_run,
            new_status=payload.status,
            timestamp=event.timestamp,
            run_label="target",
        )
        stored_target_run.status = payload.status
        for key, item in input_artifacts.items():
            stored_target_run.input_artifacts[key] = item
        # Only record target_hash on SUCCESS — non-successful runs have no meaningful hash
        if event.run_metadata.target_hash and payload.status == Status.SUCCESS:
            stored_target_run.target_hash = event.run_metadata.target_hash
        self.storage.target_storage.update(stored_target_run)

    def __has_reusable_prior_runs(self: Self) -> bool:
        """Return True if this build already has SUCCESS target runs from an
        earlier attempt whose outputs a re-execution can reuse.

        This gates target reuse, covering both re-execution paths uniformly:
        an automatic in-place retry (``retry_count`` bumped) and a
        ``gb build restart`` (fresh runner, ``retry_count`` reset to 0). A
        build's first execution has no prior runs, so reuse stays off and every
        target runs. The per-target hash match still happens in
        ``__is_target_already_run``; this is only the cheap "are we re-running?"
        gate.

        Returns:
            bool: whether the current build has at least one prior SUCCESS run.
        """
        prior = self.storage.target_storage.get_by_where(
            {"build_id": self.stored_build.uuid, "status": Status.SUCCESS.name}
        )
        return bool(prior)

    def __is_target_already_run(
        self: Self,
        target_hash: str,
    ) -> Optional[dict[str, list[str]]]:
        """Return the target's resolved_outputs if a prior successful run with the
        same target_hash already exists in *this* build and all its output
        artifacts are fully registered; otherwise None.

        Searches only the current build (a re-run reuses the same build id), so a
        target that already succeeded on an earlier attempt is not re-run.
        """
        results = self.storage.target_storage.get_by_where(
            {
                "target_hash": target_hash,
                "status": Status.SUCCESS.name,
                "build_id": self.stored_build.uuid,
            }
        )
        # This runs only when target reuse is enabled (it is the skip function).
        # With max_retries >= 2, or across a restart, the target can already hold
        # more than one SUCCESS run sharing this hash: an earlier success whose
        # output artifacts had not fully registered is re-run and succeeds again,
        # and the old run is never demoted. get_by_where ordering is undefined, so
        # pick the newest — its outputs are the current ones to reuse — never an
        # arbitrary first row.
        stored_target = latest_finished_target(results)
        if stored_target is None:
            return None
        logger.info(
            "Found previously run target %s in build %s",
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
        return resolved_outputs

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
        # RUNNING is included so a build-level retry run (set RUNNING in place by the
        # retry loop) can transition RUNNING->RUNNING/SUCCESS/FAILED.
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
        """Update the stored build status, serialized against concurrent finalizes.

        Thin wrapper holding ``_finalize_lock`` so a status transition driven from
        another thread (stop()/stop_and_fail()) cannot interleave with the worker
        thread's natural finalize; see ``_finalize_lock`` in __init__. Delegates to
        ``__update_stored_build_status_locked``.

        Args:
            status (Status): the new build status.
            failure_reason (str): applied if status is FAILED.
            unfinished_should_update: use only for unfinished status values.

        Return the updated build if the update was successful or None.
        """
        with self._finalize_lock:
            return self.__update_stored_build_status_locked(
                status, failure_reason, unfinished_should_update
            )

    def __update_stored_build_status_locked(
        self: Self,
        status: Status,
        failure_reason: str = "",
        unfinished_should_update: Optional[Callable[[StoredBuild], bool]] = None,
    ) -> Optional[StoredBuild]:
        """Body of __update_stored_build_status. Must be called with _finalize_lock held.

        Updates the inmemory and instorage StoredBuild with the new status, with
        special handling for a finished status value to update targets and steps in
        the associated build_run, if present.

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
        if payload.status == Status.SUCCESS:
            # Lineage is now recorded via DB reconciliation, not from gb_events.
            pass

    @staticmethod
    def _apply_run_timestamps(
        stored_run: Union[StoredStepRun, StoredTargetRun],
        new_status: Status,
        timestamp: datetime,
        run_label: str = "step",
    ) -> None:
        """Stamp started_at/finished_at on a step or target run for a transition.

        Mutates ``stored_run`` in place based on its CURRENT (pre-transition)
        status and the incoming ``new_status``; the caller updates
        ``stored_run.status`` afterward. Shared by the step-run and target-run
        status handlers so both obey identical transition rules.

        started_at is stamped on the FIRST entry into RUNNING, regardless of the
        prior status. Keying off "was PENDING" would drop started_at on a
        non-linear first transition -- e.g. a target run created SUBMITTED (with
        started_at still None) that goes straight to RUNNING. The
        started_at-is-None guard alone provides the "first entry" semantics: a run
        can legitimately re-enter RUNNING (the Lsf monitor reports PENDING while a
        bsub job is queued or suspended, then RUNNING again once LSF dispatches or
        resumes it), and the guard keeps that from pushing started_at later than
        the run actually began.

        finished_at is stamped on the FIRST entry into any terminal status
        (``Status.is_finished()``), regardless of the prior status. Keying off
        "was RUNNING" was wrong in both directions: a RUNNING -> PENDING transition
        (a queued or suspended scheduler job) must NOT be marked finished, and a
        run that terminates straight from a PENDING-mapped state must still be
        marked finished -- e.g. the Lsf monitor reports PENDING for a suspended job
        (PSUSP/SSUSP/USUSP/UNKWN), the job is then killed while suspended, and the
        terminal FAILED arrives from PENDING rather than RUNNING. The
        finished_at-is-None guard keeps a redelivered terminal event from
        overwriting the original timestamp.

        :param stored_run: The step or target run to mutate; its ``status`` field
            still holds the pre-transition status.
        :param new_status: The status arriving with this event.
        :param timestamp: The event timestamp to record.
        :param run_label: Noun used in log lines ("step" or "target").
        """
        if new_status is Status.RUNNING and stored_run.started_at is None:
            logger.info("%s started running at %s", run_label, timestamp)
            stored_run.started_at = timestamp
        elif new_status.is_finished() and stored_run.finished_at is None:
            logger.info("%s finished running at %s", run_label, timestamp)
            stored_run.finished_at = timestamp

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
        self._apply_run_timestamps(
            stored_run=stored_step_run,
            new_status=payload.status,
            timestamp=event.timestamp,
            run_label="step",
        )
        stored_step_run.status = payload.status
        stored_step_run.status_msg = payload.msg
        logger.info("stored_step_run %s", stored_step_run)
        assert (
            stored_step_run.uuid == targetsteprun_id
        ), f"expected {targetsteprun_id} actual {stored_step_run.uuid}"
        logger.info("creating/updating the step: %s", stored_step_run)
        # Merge any step metadata buffered before this row existed (a log-parsed
        # STEP_METADATA_UPDATE_EVENT can race ahead of this status event) so the
        # single update below persists status and metadata together, rather than
        # re-fetching and writing a second time. The buffer is cleared only after
        # the write succeeds.
        pending_metadata = self._pending_step_metadata.get(targetsteprun_id)
        if pending_metadata:
            stored_step_run.metadata.update(pending_metadata)
        self.storage.step_storage.update(stored_step_run)
        if pending_metadata:
            del self._pending_step_metadata[targetsteprun_id]

    def __process_step_metadata_update_event(self: Self, event: BuildEvent) -> None:
        """Buffer a runtime key/value and merge it into the step's metadata.

        Correlates via event.run_metadata.targetsteprun_id. The value is recorded in
        _pending_step_metadata first, then flushed onto StoredStepRun.metadata by
        _apply_pending_step_metadata. If the row does not exist yet (this log-parsed
        event was processed before the status event that creates it), the value stays
        buffered and is flushed when the step status handler creates the row, so
        metadata is never lost to event-ordering.

        An event with no targetsteprun_id to correlate against — e.g. a stray marker
        in non-step output, or an uncorrelated monitor context — is dropped with a
        warning rather than raising, matching the buffering path's "no-op if the row
        isn't present" posture. A raise here would fail the whole build under
        GBSERVER_RAISE_BUILD_EXCEPTIONS over a benign lineage marker.

        :param event: a STEP_METADATA_UPDATE_EVENT carrying a
            StepMetadataUpdateEventPayload.
        """
        payload = event.payload
        assert isinstance(payload, StepMetadataUpdateEventPayload)
        targetsteprun_id = event.run_metadata.targetsteprun_id
        if not targetsteprun_id:
            logger.warning(
                "Ignoring STEP_METADATA_UPDATE_EVENT with no targetsteprun_id "
                "(key=%s); nothing to correlate it to.",
                payload.metadata_key,
            )
            return
        self._pending_step_metadata.setdefault(targetsteprun_id, {})[
            payload.metadata_key
        ] = payload.metadata_value
        self._apply_pending_step_metadata(targetsteprun_id)

    def _apply_pending_step_metadata(self: Self, targetsteprun_id: str) -> None:
        """Flush buffered step metadata onto its StoredStepRun row once it exists.

        No-op while the row is absent (a metadata event arrived before the status
        event that creates the row); the buffer is retried from here and from the step
        status handler after row creation. On success the buffered keys are merged into
        the row (existing keys preserved) and the buffer entry is cleared.

        :param targetsteprun_id: uuid of the step run whose buffered metadata to flush.
        """
        pending = self._pending_step_metadata.get(targetsteprun_id)
        if not pending:
            return
        stored = self.storage.step_storage.get_by_uuid(targetsteprun_id)
        if stored is None:
            return  # row not created yet; keep buffered for a later flush
        assert isinstance(stored, StoredStepRun)
        stored.metadata.update(pending)
        self.storage.step_storage.update(stored)
        del self._pending_step_metadata[targetsteprun_id]
