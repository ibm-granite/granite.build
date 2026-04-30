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
Watch the admin metadata tables for new builds, cancellations, etc.
"""

import shutil
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Self, Tuple, cast

from git import Repo
from tqdm import tqdm

from gbserver.buildwatcher.abstractbuildrunner import AbstractBuildRunner
from gbserver.buildwatcher.build_setup import BuildSetup
from gbserver.buildwatcher.build_utils import (
    push_failed_status_update_metric,
    update_stored_build_status,
)
from gbserver.buildwatcher.buildrunner import BuildRunner
from gbserver.buildwatcher.buildrunnerprocess import BuildRunnerProcess
from gbserver.metrics.metrics_client import (
    push_metrics,
)
from gbserver.storage.singleton_storage import SingletonAdminStorage, get_admin_storage
from gbserver.storage.storage import CREATED_TIME_FIELD_NAME, QueryControl, SortOrder
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.buildwatcherconfig import BuildWatcherConfig
from gbserver.types.constants import (
    COMMAND_RUN_BUILD_WATCH_BUILD_NAME,
    DEFAULT_DIR_PERMS,
    GBSERVER_GITHUB_TOKEN,
    WORKSPACE_REPOS_DIR,
)
from gbserver.types.metrics import (
    Metric,
    MetricMetadata,
    MetricName,
)
from gbserver.types.prwatcherconfig import get_uri_parts
from gbserver.types.status import Status
from gbserver.utils.filesystem import create_temp_subdir
from gbserver.utils.git_retry import git_clone_retry
from gbserver.utils.logger import get_logger
from gbserver.utils.utils import get_utc_time, normalize_to_filename

logger = get_logger(__name__)

BUILD_ONLY_THIS_NAME = ""

_BUILD_EVENT_SOURCE_NAME = "build-watcher"


def get_progress_callback():
    """For use with git library during clone and push for indicating the progress."""
    pbar = tqdm()

    # pylint: disable=unused-argument
    def progress_callback(op_code, cur_count, max_count=None, message=""):
        pbar.total = max_count
        pbar.n = cur_count
        pbar.refresh()

    return progress_callback


# def build_from_build_run(build_run: BuildRun) -> Tuple[Build, StoredBuild]:
#     """Extracts the build and stored build objects from the build run."""
#     build = build_run.entity
#     assert isinstance(build, Build), f"failed to get the build from build run"
#     stored_build = build.stored
#     assert isinstance(stored_build, StoredBuild)
#     return build, stored_build


class BuildWatcher:
    """Manager for monitoring admin tables and launching builds."""

    # Metrics
    exp_mov_avg_processing_delay: float = 0.0

    # Constructor parameters
    gh_token: str = GBSERVER_GITHUB_TOKEN
    config_path: Optional[Path] = None
    watch_for_config_changes: bool
    config: BuildWatcherConfig

    # list of build ids we've seen as pending.
    active_pending_builds: List[str]
    # list of build ids we've seen as cancelled.
    active_cancelled_builds: List[str]
    # build_id -> thread for a running build
    build_threads: Dict[str, threading.Thread]
    # build_id -> thread for a creating a PR for a build
    build_pr_threads: Dict[str, threading.Thread]
    # build_id -> BuildRunner for a running build
    build_runners: Dict[str, AbstractBuildRunner]

    # Use to lock access to build_threads/runners when there might be contention.
    _builds_lock: threading.Lock
    # Use to lock access to _wait_for_completion() method.
    # Do NOT lock while holding the _builds_lock, since we lock the build_lock inside this lock.
    _wait_lock: threading.Lock

    # Lakehouse storage
    storage: SingletonAdminStorage
    # thread that monitor Lakehouse, start/manage builds, etc.
    worker_thread: Optional[threading.Thread] = None

    # Event to stop all threads
    stop_event: threading.Event

    # This is primarily for debugging from the command line when
    # we want the build to use a local git clone of a space.
    all_build_space_uri: Optional[str] = None

    # workspace dir used by the build watcher
    watcher_workspace_dir: Path

    active_submitted_builds: List[str]

    def __init__(
        self: Self,
        config_path: Optional[Path] = None,
        watch_for_config_changes: bool = True,
        gh_token: str = GBSERVER_GITHUB_TOKEN,
        all_build_space_uri: Optional[str] = None,
    ) -> None:
        self.exp_mov_avg_processing_delay = 0
        self.gh_token = gh_token
        self.config_path = config_path
        self.all_build_space_uri = all_build_space_uri
        self.watch_for_config_changes = watch_for_config_changes

        self.active_cancelled_builds = []
        self.active_pending_builds = []
        self.build_threads = {}
        self.build_pr_threads = {}
        self.build_runners = {}
        self.stop_event = threading.Event()
        self._builds_lock = threading.Lock()
        self._wait_lock = threading.Lock()

        self.storage = get_admin_storage()

        self.config = BuildWatcherConfig()
        self.__reload_config_file(first=True)
        logger.debug("using the config: %s", self.config)
        # Allows multiple watchers or others using the default dir to run in the same file system at the same time.
        # This is good at least for parallel test runs.
        work_dir = create_temp_subdir(self.config.watcher_workspace_dir)
        self.watcher_workspace_dir = work_dir.resolve()
        self.active_submitted_builds = []

    def __reload_config_file(self: Self, first: bool = False) -> None:
        """If the config file is provided, reload it"""
        if self.config_path is None:
            if first:
                self.config.initialize(spaces_storage=self.storage.space_storage)
            return
        logger.info("loading config from %s", self.config_path)
        prev_config = self.config
        self.config = BuildWatcherConfig.from_yaml(self.config_path)
        self.config.initialize(spaces_storage=self.storage.space_storage, prev_config=prev_config)

    def start_and_wait(self: Self) -> None:
        """Start and wait for the build watcher to stop.
        This will return when the main event loop is stopped, via a call to stop(),
        or when there is an unhandled exception in the loop.
        """
        logger.debug("BuildWatcher.start_and_wait start")
        self.__start_worker_thread()
        self.__wait_for_completion()  # generally this will require stop() to be called to get this method to return.
        logger.debug("BuildWatcher.start_and_wait end")

    def __start_worker_thread(self: Self) -> None:
        """
        Start a thread to monitor Lakehouse for pending builds
        and threads to run those builds.
        """
        logger.debug("BuildWatcher.start start")
        if self.worker_thread is not None:
            logger.error("lakehouse monitoring thread is already running")
            return
        self.worker_thread = threading.Thread(target=self.__worker_thread_run)
        self.worker_thread.start()
        logger.debug("BuildWatcher.start end")

    def __wait_for_completion(self: Self) -> None:
        """Wait for the monitoring thread to complete and then stop all jobs.
        This requires a lock since it can be called from both start_and_wait() and stop() when stop() is called.
        """
        logger.debug("BuildWatcher.__wait_for_completion start")
        with self._wait_lock:
            if self.worker_thread is None:
                # Need to check this since we could be called asynchrhonously from both stop() and start_and_wait().
                # One of them gets through before the other, so ignore the 2nd.
                return

            # Wait for main monitoring thread.  When that is done, we're all done.
            self.worker_thread.join()

            # Once the main monitoring thread is done, we can assume we're shutting down
            # and so we'll stop all BuildRunners and wait for them to finish.
            with self._builds_lock:
                for build_id, build_runner in self.build_runners.items():
                    logger.info("stopping build runner for build %s", build_id)
                    build_runner.stop()
                for build_id, build_thread in self.build_threads.items():
                    logger.info("waiting on thread for build %s", build_id)
                    build_thread.join()
                for build_id, build_pr_thread in self.build_pr_threads.items():
                    logger.info("waiting on PR creation thread for build %s", build_id)
                    build_pr_thread.join()
            self.__clean_finished_builds()  # Not required, but may help to clean up memory.
            self.worker_thread = None

        logger.debug("BuildWatcher.__wait_for_completion end")

    def stop(self: Self) -> None:
        """Stop the monitoring thread, and wait for it to complete, the stop() all builds ."""
        logger.debug("BuildWatcher.stop start")
        self.stop_event.set()  # Get the monitoring thread to complete
        self.__wait_for_completion()
        logger.debug("BuildWatcher.stop end")

    def __process_cancel_requested_build(self, build: StoredBuild):
        """Update internal data structures, stop the given build if it is in our build_runners, and join the thread of it has a thread."""
        with self._builds_lock:
            if build.uuid in self.build_runners:
                self.build_runners[build.uuid].stop()
                del self.build_runners[build.uuid]
            if build.uuid in self.build_threads:
                self.build_threads[build.uuid].join()
                del self.build_threads[build.uuid]
            if build.uuid in self.build_pr_threads:
                self.build_pr_threads[build.uuid].join()
                del self.build_pr_threads[build.uuid]

    def __clean_finished_builds(self: Self) -> None:
        """See which threads are no longer alive and clean up our references to a) the threads and b) the associated build run.
        This should generally only be called from __start_worker_thread() or when self.build_* can be safely edited.
        """
        finished_builds = []
        with self._builds_lock:
            for build_id, thread in self.build_threads.items():
                if not thread.is_alive():
                    finished_builds.append(build_id)

            if len(finished_builds) > 0:
                logger.info("Cleaning up finished builds %s", finished_builds)
                for build_id in finished_builds:
                    del self.build_threads[build_id]
                    del self.build_runners[build_id]
                    self.build_pr_threads.pop(build_id, None)

    def __process_cancel_requested_builds(self: Self) -> None:
        cancelled_builds = []
        try:
            logger.info("fetching cancelled builds...")
            cancelled_builds = (
                self.__get_newly_cancelled_builds()
            )  # only those from the assigned space(s)
        except Exception as e:
            logger.error("%s", traceback.format_exc())
            logger.error("failed to fetch the new cancelled builds error: %s", e)
        to_show_cancelled = [b.source_uri for b in cancelled_builds]
        logger.debug("all cancelled builds: %d %s", len(to_show_cancelled), to_show_cancelled)
        for b in cancelled_builds:
            self.__process_cancel_requested_build(b)

    def __log_submission_delay(self: Self, stored_build: StoredBuild) -> None:
        created_at = stored_build.created_time
        now = get_utc_time()
        processing_delay = (now - created_at).total_seconds()
        if processing_delay < 0:
            logger.warning("negative processing_delay %s", processing_delay)
        if self.exp_mov_avg_processing_delay <= 0:
            self.exp_mov_avg_processing_delay = processing_delay
        else:
            self.exp_mov_avg_processing_delay = (
                self.exp_mov_avg_processing_delay * 0.9 + processing_delay * 0.1
            )
        build_id = stored_build.uuid
        logger.info(
            "Processing submitted build %s submitted %s seconds ago. Current average delay %s seconds)",
            build_id,
            processing_delay,
            self.exp_mov_avg_processing_delay,
        )
        push_metrics(
            metrics=[
                Metric(
                    name=MetricName.PROCESSING_DELAY,
                    value=processing_delay,
                    metadata=MetricMetadata(build_id=build_id),
                ),
                Metric(
                    name=MetricName.EXP_MOV_AVG_PROCESSING_DELAY,
                    value=self.exp_mov_avg_processing_delay,
                ),
            ]
        )

    def __process_submitted_build(self: Self, stored_build: StoredBuild) -> None:
        """Update the status of the build to PENDING"""
        logger.info(
            f"Enter process_submitted_build({stored_build.uuid}, status={stored_build.status})"
        )

        # 0. Log the time it took between the build creation in the DB and the time to get there.
        self.__log_submission_delay(stored_build=stored_build)

        # Make this the last thing we do to signal that we're done and ready for processing a PENDING build.
        # Require the build to be SUBMITTED and not some other state before moving to PENDING state
        valid_status = lambda item: item.status == Status.SUBMITTED
        build = update_stored_build_status(
            stored_build.uuid,
            Status.PENDING,
            failure_reason="",
            should_update=valid_status,
        )
        if not build:
            logger.warning(
                "Build %s status update from SUBMITTED to PENDING failed (may have been cancelled or deleted)",
                stored_build.uuid,
            )
            push_failed_status_update_metric(stored_build.uuid, [Status.SUBMITTED])

        logger.info(f"Exit process_submitted_build({stored_build.uuid})")

    def __update_existing_repo_clone(
        self: Self,
        clone_path: Path,
        branch: str,
        clone_url_with_creds: str,
        clone_url_redacted: str,
    ) -> Optional[Tuple[Path, Repo]]:
        """Reset and pull an existing local clone to bring it up to date.

        Updates the remote URL if credentials have changed. Returns (path, repo) on
        success, or None if the existing clone could not be used (caller should reclone).
        """
        try:
            repo = Repo(path=clone_path)
            logger.info("repo already cloned at path %s , pulling latest changes...", clone_path)
            repo.git.reset("--hard")
            repo.git.clean("-xdf")
            repo.git.checkout(branch)
            if repo.git.remote("get-url", "origin") != clone_url_with_creds:
                logger.warning(
                    "setting the remote to '%s' (maybe the creds changed)",
                    clone_url_redacted,
                )
                repo.git.remote("set-url", "origin", clone_url_with_creds)
            repo.git.pull()
            return clone_path, repo
        except Exception as e:
            logger.error(
                "delete and reclone because we failed to use the existing repo, error: %s",
                e,
            )
            return None

    def __clone_repo_for_pr(
        self: Self,
        repo_url: str,
        repo_branch: str = "main",
        force: bool = False,
        shallow: bool = True,
    ) -> Tuple[Path, Repo]:
        """Clone the repo fork and branch used in the PR."""
        logger.info("BuildWatcher.clone_repo_for_pr start repo_url: %s", repo_url)
        assert self.gh_token != "", "The GitHub token is empty"
        pr_html_url_parts = get_uri_parts(uri=repo_url)
        pr_clone_path = (
            self.watcher_workspace_dir
            / WORKSPACE_REPOS_DIR
            / str(threading.get_ident())
            / normalize_to_filename(pr_html_url_parts[2])
            / normalize_to_filename(pr_html_url_parts[3])
            / repo_branch
        )
        repo_url_no_prefix = repo_url.removeprefix("https://")
        clone_url_with_creds = "https://" + self.gh_token + "@" + repo_url_no_prefix
        clone_url_redacted = "https://****@" + repo_url_no_prefix
        logger.info("pr_clone_path: %s clone url: %s", pr_clone_path, clone_url_redacted)
        if pr_clone_path.is_dir():
            logger.info("the repo directory already exists at path: %s", pr_clone_path)
            if force:
                logger.warning("force is specified, delete and reclone")
            else:
                result = self.__update_existing_repo_clone(
                    pr_clone_path, repo_branch, clone_url_with_creds, clone_url_redacted
                )
                if result is not None:
                    return result
        logger.info("delete path %s", pr_clone_path)
        shutil.rmtree(pr_clone_path, ignore_errors=True)
        pr_clone_path.mkdir(mode=DEFAULT_DIR_PERMS, parents=True, exist_ok=False)
        clone_kwargs: dict = {
            "url": clone_url_with_creds,
            "to_path": pr_clone_path,
            "progress": get_progress_callback(),
            "branch": repo_branch,
            "no_checkout": True,
        }
        if shallow:
            clone_kwargs["depth"] = 1
        logger.info("cloning the repo")
        repo = self._clone_repo_with_retry(**clone_kwargs)
        repo.git.clean("-xdf")  # Not necessary, just to be sure of git status clean
        repo.git.checkout(repo_branch)
        logger.info("cloned pull request repo %s", repo)
        logger.info("BuildWatcher.clone_repo_for_pr end")
        return pr_clone_path, repo

    @git_clone_retry
    def _clone_repo_with_retry(self, url: str, to_path: Path, **kwargs) -> Repo:
        """Clone repository with retry logic."""
        return Repo.clone_from(url, to_path, **kwargs)

    def __process_submitted_builds(self: Self) -> None:
        submitted_builds = []
        try:
            logger.info("fetching submitted builds...")
            submitted_builds = (
                self.__get_newly_submitted_builds()
            )  # only those from the assigned space(s)
        except Exception as e:
            logger.error("failed to fetch the submitted builds error: %s", e)
            return
        logger.info("submitted builds: %d", len(submitted_builds))
        for submitted_build in submitted_builds:
            build_id = submitted_build.uuid
            # Do the processing, especially build long-running validation, in threads so we don't
            # 1) hold up other main event loop processing or
            # 2) serialize build submission processing (i.e. let them be processed independent of each other)
            thread = threading.Thread(
                name=f"Submitted build processor for build {build_id}",
                target=self.__process_submitted_build,
                args=(submitted_build,),
            )
            thread.daemon = True
            thread.start()

    def __process_pending_builds(self: Self) -> None:
        pending_builds = []
        try:
            logger.info("fetching pending builds...")
            pending_builds = (
                self.__get_newly_pending_builds()
            )  # only those from the assigned space(s)
            logger.info(f"Found {len(pending_builds)} pending builds in our managed spaces")
        except Exception as e:
            logger.error("%s", traceback.format_exc())
            logger.error("failed to fetch the new pending builds error: %s", e)
            return

        for b in pending_builds:
            try:
                # Start the build
                logger.info("starting build id %s", b.uuid)
                self.__start_build(b)
            except Exception as e:
                logger.error("failed to start build %s, error: %s", b.uuid, e)

    def __worker_thread_run(self: Self) -> None:
        """
        This is the main processing loop that:
            1) reloads the config, if configured to do so
            2) cancels builds
            3) starts pending builds on separate threads.
            4) garbage collects finished builds
        """
        logger.info("BuildWatcher.__worker_thread_run start")
        while not self.stop_event.is_set():
            logger.info("Begin processing loop")
            try:
                if self.watch_for_config_changes:
                    self.__reload_config_file()
                self.__clean_finished_builds()
                self.__process_cancel_requested_builds()
                self.__process_submitted_builds()
                self.__process_pending_builds()
                if not self.stop_event.is_set():
                    logger.info("End processing loop. Sleeping...")
                    time.sleep(self.config.monitoring_interval)
            except Exception as e:
                msg = traceback.format_exc()
                logger.error("Ignoring exception in BuildWatcher: %s\n%s", e, msg)
        if self.stop_event.is_set():
            logger.warning("stop event has been set, stopping __worker_thread_run...")

        logger.info("BuildWatcher.__worker_thread_run end")

    def __get_newly_submitted_builds(self: Self) -> List[StoredBuild]:
        """Get a list of builds that are new and in pending state."""
        return self.__get_unseen_builds_matching_status(
            Status.SUBMITTED, self.active_submitted_builds
        )

    def __get_newly_cancelled_builds(self: Self) -> List[StoredBuild]:
        """Get a list of builds that are new and in pending state."""
        return self.__get_unseen_builds_matching_status(
            Status.CANCEL_REQUESTED, self.active_cancelled_builds
        )

    def __get_newly_pending_builds(self: Self) -> List[StoredBuild]:
        """Get a list of builds that are new and in pending state."""
        return self.__get_unseen_builds_matching_status(Status.PENDING, self.active_pending_builds)

    def __get_unseen_builds_matching_status(
        self: Self, status: Status, active_build_ids: list[str]
    ) -> List[StoredBuild]:
        """Get a list of builds that are not yet currently active and have the given status.
        Also remove any builds that no longer have the given status from the active_build_ids list.
        """
        logger.debug("BuildWatcher.__get_unseen_builds_matching_status status={status} start")
        builds = self.__get_builds_matching_status(status)
        new_builds = []
        for b in builds:
            build_id = b.uuid
            if build_id in active_build_ids:
                logger.debug("already seen build: %s , skipping", b)
                continue
            active_build_ids.append(build_id)
            if b.name == COMMAND_RUN_BUILD_WATCH_BUILD_NAME:
                logger.info("local build: %s , skipping", b)
                continue
            new_builds.append(b)

        # Do some garbage collection on the list of active builds.
        build_ids_with_status = []
        for b in builds:
            build_ids_with_status.append(b.uuid)
        to_remove = []
        for build_id in active_build_ids:
            if (
                not build_id in build_ids_with_status
            ):  # active build no longer has the requested state, so we can stop tracking it.
                to_remove.append(build_id)
        for build_id in to_remove:
            active_build_ids.remove(build_id)

        logger.debug("BuildWatcher.__get_unseen_builds_matching_status status={status} end")
        return new_builds

    def __get_builds_matching_status(self: Self, status: Status) -> List[StoredBuild]:
        """Get a sorted list of builds that are in the given state and in one of the spaces we're assigned to watch
        Sorting is by creation time with oldest first.
        """
        where = {"status": status.name}
        sort_order = SortOrder(column=CREATED_TIME_FIELD_NAME, ascending=True)
        query_control = QueryControl(pagination=None, sort_orders=[sort_order])
        pending_builds = cast(
            List[StoredBuild],
            self.storage.build_storage.get_by_where(where=where, query_control=query_control),
        )
        our_pending_builds = []
        for b in pending_builds:  # TODO: should really do this in the query above
            if b.space_name in self.config._spaces.keys():
                our_pending_builds.append(b)
        if BUILD_ONLY_THIS_NAME == "":
            return our_pending_builds
        logger.info("filtering by BUILD_ONLY_THIS_NAME: %s", BUILD_ONLY_THIS_NAME)
        return [b for b in pending_builds if b.name == BUILD_ONLY_THIS_NAME]

    def __start_build(self: Self, build: StoredBuild) -> None:
        build_id = build.uuid

        build_runner = self.__create_build_runner(build)

        build_thread = threading.Thread(
            target=build_runner.start_and_wait,
            args=(),
            name=f"BuildRunner: build_id={build_id}",
        )
        with self._builds_lock:
            self.build_runners[build_id] = build_runner
            self.build_threads[build_id] = build_thread
        build_thread.start()

    def __warn_space_uri_not_supported(self: Self, runner_type: str) -> None:
        """Log a warning when all_build_space_uri is set but the runner type does not support it."""
        if self.all_build_space_uri is not None:
            logger.warning(
                "Build runner type %s does not support space uri. Ignoring %s",
                runner_type,
                self.all_build_space_uri,
            )

    def __create_build_runner(self: Self, build: StoredBuild) -> AbstractBuildRunner:
        """Return the AbstractBuildRunner implementation configured for this instance (thread, process, or job)."""
        runner_type = self.config.buildrunner_type
        if runner_type == "thread":
            return BuildRunner(
                build=build,
                gh_token=self.gh_token,
                gh_api_endpoint=self.config.gh_api_endpoint,
                monitoring_interval=self.config.monitoring_interval,
                workspace_dir=self.config.workspace_dir,
                space_uri=self.all_build_space_uri,
                create_pr=bool(self.gh_token),
            )
        if runner_type == "process":
            self.__warn_space_uri_not_supported(runner_type)
            return BuildRunnerProcess(
                build=build,
                gh_token=self.gh_token,
                gh_api_endpoint=self.config.gh_api_endpoint,
                monitoring_interval=self.config.monitoring_interval,
                workspace_dir=self.config.workspace_dir,
            )
        # Default to job runner
        if runner_type != "job":
            logger.warning("Build runner type %s not recognized. Using job type", runner_type)
        self.__warn_space_uri_not_supported(runner_type)
        from gbserver.buildwatcher.buildrunnerjob import BuildRunnerJob

        return BuildRunnerJob(
            build=build,
            gh_token=self.gh_token,
            gh_api_endpoint=self.config.gh_api_endpoint,
            monitoring_interval=self.config.monitoring_interval,
            workspace_dir=self.config.workspace_dir,
        )
