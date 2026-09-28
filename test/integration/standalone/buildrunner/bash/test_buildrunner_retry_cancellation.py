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

import os
import signal
import threading
import uuid
from pathlib import Path
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

# The orphan-reap test generates its build.yaml at run time (see
# _write_orphan_build_yaml): the workload writes its own shell PID to a per-run
# unique file, then sleeps. The test reads that PID and asserts it is reaped
# after cancellation via os.kill(pid, 0) — no argv reading, so it works on both
# Linux and macOS. The build.yaml and PID path are generated per run (not a fixed
# /tmp path) so concurrent pytest sessions on a shared host cannot collide.


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

    def _get_orphan_spec(self) -> BuildTestSpecification:
        return BuildTestSpecification.from_yaml(
            get_test_data_dir_for(__file__) / "retry-cancel" / "buildtest-orphan.yaml"
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
        """Block until the build is no longer in flight.

        On timeout, report the statuses actually observed and how many build records
        exist. Without that the failure surfaces downstream as a stale assertion
        (e.g. "workload still alive"), which reads like a reaping bug rather than a
        build that never settled.
        """
        poll = 2.0
        start = time()
        seen: list[str] = []
        while time() - start <= timeout_seconds:
            builds = self.storage.build_storage.get_by_uuid(None) or []
            build = next((b for b in builds if b.uuid == build_id), None)
            if build is not None:
                name = build.status.name
                if not seen or seen[-1] != name:
                    seen.append(name)
                if build.status not in _IN_FLIGHT:
                    return
            sleep(poll)
        assert False, (
            f"Build {build_id} did not settle within {timeout_seconds}s. "
            f"Status sequence observed: {' -> '.join(seen) or 'none'}. "
            f"{len(self.storage.build_storage.get_by_uuid(None) or [])} build "
            "record(s) exist; more than one for this id means it was dispatched twice."
        )

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        """True if pid is a live process. os.kill(pid, 0) is OS-agnostic and needs
        no argv reading (unlike psutil cmdline, which is unreliable on macOS)."""
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Exists but owned by another user — still alive for our purposes.
            return True

    @staticmethod
    def _read_workload_pid(pid_file: Path, timeout_seconds: float):
        """Poll for the PID file the workload writes; return its pid, or None."""
        start = time()
        while time() - start <= timeout_seconds:
            try:
                text = pid_file.read_text(encoding="utf-8").strip()
                if text:
                    return int(text)
            except (FileNotFoundError, ValueError):
                pass
            sleep(0.5)
        return None

    @staticmethod
    def _write_orphan_build_yaml(dest_dir: Path, pid_file: Path) -> Path:
        """Write a per-run orphan build.yaml whose workload records its PID.

        The command writes its shell PID to ``pid_file`` then sleeps far longer
        than the test. Generating this per run (unique pid_file) rather than using
        a fixed path keeps concurrent pytest sessions on a shared host from
        colliding. retries.max_retries has headroom so the chain would keep going
        if cancellation didn't stop it.
        """
        build_yaml = dest_dir / "build-orphan.yaml"
        build_yaml.write_text(
            "granite.build:\n"
            "  name: bash-retry-cancel-orphan-test\n"
            "  retries:\n"
            "    max_retries: 5\n"
            "  targets:\n"
            "    orphan-target:\n"
            "      allow_unknown: true\n"
            "      environment_uri: space://environments/bash\n"
            "      steps:\n"
            "        - step_uri: space://steps/command\n"
            "          config:\n"
            "            command_config:\n"
            f"              command: 'echo $$ > {pid_file}; sleep 600'\n"
            "            compute_config:\n"
            "              num_nodes: 1\n",
            encoding="utf-8",
        )
        return build_yaml

    def test_cancel_reaps_workload_child(self, tmp_path):
        """Cancelling a build reaps the bash workload process (no orphan).

        Regression for the SIGTERM shutdown flake: without cleanup_nohup the
        cancelled build's workload (a session leader via start_new_session) was
        never killed, so shutdown could wait out the workload and the process
        leaked. The workload records its PID; after cancellation that PID must be
        dead.
        """
        # Per-run unique PID file + generated build.yaml (no shared /tmp path), so
        # concurrent pytest sessions on the same host cannot collide.
        pid_file = tmp_path / f"gborphan-{uuid.uuid4().hex}.pid"

        spec = self._get_orphan_spec()
        spec.build_yaml = str(self._write_orphan_build_yaml(tmp_path, pid_file))
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
        workload_pid = None
        try:
            timeout = spec.timeout_minutes * 60
            # The workload must actually be running before we cancel (sanity).
            # Bound this independently of the (much longer) build timeout so a
            # miss fails fast instead of burning the whole budget.
            workload_pid = self._read_workload_pid(pid_file, timeout_seconds=90)
            assert workload_pid is not None, "workload never wrote its PID file"
            assert self._pid_alive(workload_pid), "workload PID not alive after launch"

            build = self.storage.build_storage.get_by_uuid(build_id)
            request_cancellation(self.storage.build_storage, build)
            self._wait_until_settled(build_id, timeout)

            # cleanup_nohup allows a 5s SIGTERM grace + SIGKILL; give ample margin.
            deadline = time() + 30
            while time() < deadline and self._pid_alive(workload_pid):
                sleep(0.5)
            assert not self._pid_alive(workload_pid), (
                f"workload pid {workload_pid} still alive after cancellation "
                "(cleanup_nohup did not reap it)"
            )
        finally:
            watcher.stop()
            thread.join(timeout=60)
            # Belt-and-suspenders: never let a failing test leak a 600s sleep.
            # Only kill the pid itself (not its group): after a reap the pid may
            # be gone and its number reused, so signalling a whole group by pgid
            # could hit an unrelated process. os.kill of a stale pid is a no-op
            # error we swallow.
            if workload_pid is not None and self._pid_alive(workload_pid):
                try:
                    os.kill(workload_pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            # pid_file lives under pytest's tmp_path, cleaned up automatically.

        builds = self.storage.build_storage.get_by_uuid(None) or []
        assert (
            len(builds) == 1
        ), f"In-place retry must reuse one build id, found {len(builds)}"
        assert builds[0].status == Status.CANCELLED
