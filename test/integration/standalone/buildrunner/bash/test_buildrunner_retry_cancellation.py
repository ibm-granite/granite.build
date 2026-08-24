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

"""Cancellation of an in-place build retry.

With in-place retry a build keeps a single build id across attempts. Cancelling
the build while a retry is in flight must stop the active run, mark the one build
CANCELLED, and prevent any further retries. There is no retry chain to walk.
"""

import threading
from time import sleep, time

import pytest
from fastapi import HTTPException
from libgbtest.buildrunner.buildtest import (
    AbstractBuildTest,
    BuildTestSpecification,
    get_test_data_dir_for,
)
from libgbtest.buildrunner.utils import ExceptionRaisingThread
from libgbtest.constants import GBTEST_SPACE_NAME, GBTEST_USER_NAME

from gbserver.api.builds import request_cancellation
from gbserver.buildrunner.buildrunner import BuildRunner
from gbserver.buildwatcher.buildwatcher import BuildWatcher
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.status import Status

pytestmark = pytest.mark.standalone

_IN_FLIGHT = {
    Status.SUBMITTED,
    Status.PENDING,
    Status.RUNNING,
    Status.CANCEL_REQUESTED,
}


@pytest.mark.xdist_group(name="buildwatcher_bash_cancel")
class TestInPlaceRetryCancellation(AbstractBuildTest):
    """Cancelling an in-place retry stops it and marks the one build CANCELLED."""

    def setup_method(self, method):
        self.run_locally = True
        super().setup_method(method)

    def _get_spec(self) -> BuildTestSpecification:
        return BuildTestSpecification.from_yaml(
            get_test_data_dir_for(__file__) / "retry-cancel" / "buildtest.yaml"
        )

    def _make_build(self, status, retry_count) -> StoredBuild:
        """Create a bare StoredBuild for cancellation-routing assertions (not executed)."""
        return StoredBuild(
            name="test",
            space_name=GBTEST_SPACE_NAME,
            source_uri="",
            username=GBTEST_USER_NAME,
            status=status,
            retry_count=retry_count,
        )

    def _bare_runner(self, build: StoredBuild) -> BuildRunner:
        """A BuildRunner wired just enough to exercise __cancel_build_run / stop()."""
        runner = object.__new__(BuildRunner)
        runner.stored_build = build
        runner.storage = self.storage
        runner.build_run = None
        runner.stop_event = threading.Event()
        runner._stop_requested = threading.Event()
        runner._finalize_lock = threading.Lock()
        return runner

    def test_stop_after_success_does_not_cancel(self):
        """Stopping the runner as cleanup must not flip a finished build.

        The harness (and BuildWatcher shutdown) call runner.stop() after a build
        completes. With no cancellation requested, a SUCCESS build must stay
        SUCCESS — __cancel_build_run must not relabel finished builds.
        """
        build = self._make_build(Status.SUCCESS, 0)
        self.storage.build_storage.add(build)
        self._bare_runner(build).stop()
        assert (
            self.storage.build_storage.get_by_uuid(build.uuid).status == Status.SUCCESS
        ), "A cleanup stop() must not cancel a build that already succeeded"

    def test_request_cancellation_routes_by_status(self):
        """request_cancellation maps each build status to the right outcome.

        In-place retry keeps one build id, so cancellation is decided purely by
        the build's current status: an in-flight RUNNING build (including one the
        retry loop re-ran in place) becomes CANCEL_REQUESTED for the runner to act
        on; a not-yet-started (SUBMITTED/PENDING) build is cancelled outright; a
        finished build (a FAILED build with retries exhausted, or SUCCESS/CANCELLED)
        is not cancellable.
        """
        # In-flight: RUNNING defers to the runner via CANCEL_REQUESTED.
        build = self._make_build(Status.RUNNING, 1)
        self.storage.build_storage.add(build)
        updated = request_cancellation(self.storage.build_storage, build)
        assert (
            updated.status == Status.CANCEL_REQUESTED
        ), f"RUNNING should route to CANCEL_REQUESTED, got {updated.status}"

        # Not yet started: cancelled outright.
        for pre_run in (Status.SUBMITTED, Status.PENDING):
            build = self._make_build(pre_run, 0)
            self.storage.build_storage.add(build)
            updated = request_cancellation(self.storage.build_storage, build)
            assert (
                updated.status == Status.CANCELLED
            ), f"{pre_run} should route to CANCELLED, got {updated.status}"

        # Finished (retries exhausted / already done): not cancellable -> 412.
        for finished in (Status.FAILED, Status.SUCCESS, Status.CANCELLED):
            build = self._make_build(finished, 2)
            self.storage.build_storage.add(build)
            with pytest.raises(HTTPException) as exc_info:
                request_cancellation(self.storage.build_storage, build)
            assert exc_info.value.status_code == 412

    def test_cancel_stops_in_flight_retry(self):
        """E2E: cancel the build mid-retry; the one build ends CANCELLED."""
        spec = self._get_spec()
        space = self._check_and_setup_space(spec)

        stored_build = StoredBuild.create(
            name="test",
            space_name=space.name,
            source_uri="",
            username=GBTEST_USER_NAME,
            build_yaml_path=spec.build_yaml,
            status=Status.SUBMITTED,
        )
        build_id = stored_build.uuid
        self.storage.build_storage.add(stored_build)

        watcher = BuildWatcher(gh_token="", all_build_space_uri=spec.space_uri)
        watcher.config.buildrunner_type = "thread"
        watcher.config.monitoring_interval = 1

        thread = ExceptionRaisingThread(
            name="BuildWatcher", target=watcher.start_and_wait, args=()
        )
        thread.start()
        try:
            timeout = spec.timeout_minutes * 60
            # Wait until the first attempt has failed and a retry is in flight.
            self._wait_for_active_retry(build_id, timeout)
            build = self.storage.build_storage.get_by_uuid(build_id)
            request_cancellation(self.storage.build_storage, build)
            self._wait_until_settled(build_id, timeout)
        finally:
            watcher.stop()
            thread.join(timeout=60)

        builds = self.storage.build_storage.get_by_uuid(None) or []
        assert (
            len(builds) == 1
        ), f"In-place retry must reuse one build id, found {len(builds)} builds"
        build = builds[0]
        assert build.uuid == build_id
        assert (
            build.status == Status.CANCELLED
        ), f"Build should be CANCELLED after cancellation, got {build.status}"
        # Cancellation stopped it well short of exhausting max_retries (5).
        assert (
            build.retry_count < 5
        ), f"Build kept retrying after cancellation: retry_count={build.retry_count}"

    def _wait_for_active_retry(self, build_id: str, timeout_seconds: float) -> None:
        """Block until the build has retried at least once and is in flight."""
        start = time()
        while time() - start <= timeout_seconds:
            builds = self.storage.build_storage.get_by_uuid(None) or []
            build = next((b for b in builds if b.uuid == build_id), None)
            if (
                build is not None
                and build.retry_count >= 1
                and build.status in _IN_FLIGHT
            ):
                return
            sleep(1)
        assert False, f"No active retry appeared within {timeout_seconds}s."

    def _wait_until_settled(self, build_id: str, timeout_seconds: float) -> None:
        """Block until the build is no longer in flight."""
        poll = 2.0
        start = time()
        while time() - start <= timeout_seconds:
            builds = self.storage.build_storage.get_by_uuid(None) or []
            build = next((b for b in builds if b.uuid == build_id), None)
            if build is not None and build.status not in _IN_FLIGHT:
                return
            sleep(poll)
        assert False, f"Build {build_id} did not settle within {timeout_seconds}s."
