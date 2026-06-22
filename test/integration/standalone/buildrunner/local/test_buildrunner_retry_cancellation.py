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

"""Cancellation of a build-level retry chain.

Cancelling any member of a retry chain — including the already-FAILED original —
must stop the active retry and mark every build in the chain CANCELLED, and the
chain must not spawn further retries.
"""

import threading
from pathlib import Path
from time import sleep, time

import pytest
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
from gbserver.storage.stored_build import StoredBuild, get_retry_chain_members
from gbserver.types.status import Status

pytestmark = pytest.mark.standalone

_IN_FLIGHT = {
    Status.SUBMITTED,
    Status.PENDING,
    Status.RUNNING,
    Status.CANCEL_REQUESTED,
    Status.RETRY_PENDING,
}


@pytest.mark.xdist_group(name="buildwatcher_bash_cancel")
class TestRetryChainCancellation(AbstractBuildTest):
    """Cancelling a retry chain stops it and marks all members CANCELLED."""

    def setup_method(self, method):
        self.run_locally = True
        super().setup_method(method)

    def _get_spec(self) -> BuildTestSpecification:
        return BuildTestSpecification.from_yaml(
            get_test_data_dir_for(__file__) / "retry-cancel" / "buildtest.yaml"
        )

    def _make_build(self, status, retry_count, retry_of_build_id=None) -> StoredBuild:
        """Create a bare StoredBuild for chain-routing assertions (not executed)."""
        return StoredBuild(
            name="test",
            space_name=GBTEST_SPACE_NAME,
            source_uri="",
            username=GBTEST_USER_NAME,
            status=status,
            retry_count=retry_count,
            retry_of_build_id=retry_of_build_id,
        )

    def _bare_runner(self, build: StoredBuild) -> BuildRunner:
        """A BuildRunner wired just enough to exercise __cancel_build_run / stop()."""
        runner = object.__new__(BuildRunner)
        runner.stored_build = build
        runner.storage = self.storage
        runner.build_run = None
        runner.stop_event = threading.Event()
        runner._retry_chain_build_ids = [build.uuid]
        runner._retry_chain_lock = threading.Lock()
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

    def test_cancel_failed_root_is_accepted_when_chain_active(self):
        """Deterministic: a FAILED root with an active retry can be cancelled.

        The request sets CANCEL_REQUESTED on the root itself (a durable signal the
        runner detects chain-wide); the active retry is left for the runner to act
        on. With no active member, cancellation is rejected.
        """
        root = self._make_build(Status.FAILED, 0)
        retry1 = self._make_build(Status.FAILED, 1, retry_of_build_id=root.uuid)
        retry2 = self._make_build(Status.RUNNING, 2, retry_of_build_id=root.uuid)
        root.retry_build_id = retry1.uuid
        retry1.retry_build_id = retry2.uuid
        for b in (root, retry1, retry2):
            self.storage.build_storage.add(b)

        members = get_retry_chain_members(self.storage.build_storage, root)
        assert [m.uuid for m in members] == [root.uuid, retry1.uuid, retry2.uuid]

        # Cancelling the FAILED root is accepted because retry2 is still active.
        updated = request_cancellation(self.storage.build_storage, root)
        assert updated.uuid == root.uuid
        assert updated.status == Status.CANCEL_REQUESTED

        # With every member finished (no active retry), cancellation is rejected.
        self.storage.build_storage.update_fields(root.uuid, {"status": Status.FAILED})
        self.storage.build_storage.update_fields(retry2.uuid, {"status": Status.FAILED})
        done_root = self.storage.build_storage.get_by_uuid(root.uuid)
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            request_cancellation(self.storage.build_storage, done_root)

    def test_cancel_stops_retry_chain(self):
        """E2E: cancel the FAILED root mid-chain; all chain builds end CANCELLED."""
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
        root_id = stored_build.uuid
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
            self._wait_for_active_retry(timeout)
            # Cancel via the FAILED root — routes to the active retry.
            root = self.storage.build_storage.get_by_uuid(root_id)
            request_cancellation(self.storage.build_storage, root)
            self._wait_until_chain_settles(timeout)
        finally:
            watcher.stop()
            thread.join(timeout=60)

        builds = self.storage.build_storage.get_by_uuid(None) or []
        statuses = [b.status for b in builds]
        assert all(s == Status.CANCELLED for s in statuses), (
            f"Every build in the chain should be CANCELLED after cancellation, "
            f"got {statuses}"
        )
        # The chain stopped well short of exhausting max_retries (5).
        assert max(b.retry_count for b in builds) < 5, (
            f"Chain kept retrying after cancellation: retry_counts="
            f"{sorted(b.retry_count for b in builds)}"
        )

    def _wait_for_active_retry(self, timeout_seconds: float) -> None:
        """Block until a build with retry_count >= 1 is in flight (a retry is running)."""
        start = time()
        while time() - start <= timeout_seconds:
            builds = self.storage.build_storage.get_by_uuid(None) or []
            if any(b.retry_count >= 1 and b.status in _IN_FLIGHT for b in builds):
                return
            sleep(1)
        assert False, f"No active retry appeared within {timeout_seconds}s."

    def _wait_until_chain_settles(self, timeout_seconds: float) -> None:
        """Block until no build is in flight and the count is stable across two polls."""
        poll = 2.0
        prev_count = -1
        start = time()
        while time() - start <= timeout_seconds:
            builds = self.storage.build_storage.get_by_uuid(None) or []
            in_flight = [b for b in builds if b.status in _IN_FLIGHT]
            if builds and not in_flight and len(builds) == prev_count:
                return
            prev_count = len(builds)
            sleep(poll)
        assert False, f"Retry chain did not settle within {timeout_seconds}s."
