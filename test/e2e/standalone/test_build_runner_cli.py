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

"""End-to-end tests for the ``gbserver build-runner`` CLI signal handling.

Unlike the harness-based build tests (which drive ``BuildRunner`` in-process),
these launch the real ``gbserver build-runner`` process against a minimal 1-step
bash build and assert ONLY the final build status for three scenarios:

* clean run                -> SUCCESS
* SIGINT (Ctrl+C)          -> CANCELLED
* SIGTERM (process kill)   -> FAILED

The build runs in STANDALONE mode against the repo's ``configurations/spaces/local``
space via the ``--space-dir`` option. The test extends
``AbstractSingletonStorageUsingTest`` so each test method gets a unique admin-table
prefix (dropped in teardown); that same prefix is handed to the subprocess via the
root ``--gb-admin-table-prefix`` flag, so the parent and the build-runner share the
same prefixed SQLite tables. The parent therefore:

* pre-registers the local space (as ``public``) in storage — ``load_build`` resolves
  the build's space by name, so it must exist before the runner starts; and
* reads the persisted final ``StoredBuild.status`` back from storage to assert on.

The build-runner is signalled once it reaches the running state, detected via a
marker line in the process's own log file (``GBSERVER_LOG_FILE``).
"""

import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import pytest
from libgbtest.utils import AbstractSingletonStorageUsingTest

from gbserver.storage.singleton_storage import reset_admin_storage
from gbserver.storage.sql.engine_cache import get_singleton_engine_cache
from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory
from gbserver.storage.storage_factory import StorageFactory
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_space import StoredSpace
from gbserver.types.constants import (
    COMMAND_RUN_BUILD_WATCH_BUILD_NAME,
    GB_BUILDS_TABLE_NAME,
    PUBLIC_SPACE_NAME,
)
from gbserver.types.status import Status

# test/e2e/standalone/<this file>  ->  repo root is three parents up from the dir.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SPACE_DIR = _REPO_ROOT / "configurations" / "spaces" / "local"
_BUILD_ROOT = (
    _REPO_ROOT / "test-data" / "e2e" / "standalone" / "build_runner_cli_signals"
)
_SUCCESS_BUILD_DIR = _BUILD_ROOT / "success"
_LONGRUNNING_BUILD_DIR = _BUILD_ROOT / "longrunning"
_LONGRUNNING_RETRY_BUILD_DIR = _BUILD_ROOT / "longrunning-retry"

# Log line the worker loop emits repeatedly once the build is actually running;
# used to time signal delivery so we interrupt a RUNNING build, not setup.
_RUNNING_MARKER = "waiting for events"

# Timeouts (seconds). First run copies the space + loads assetstores, so allow
# generous headroom for the running marker and for the build to complete.
_MARKER_TIMEOUT = 90
_SIGNAL_EXIT_TIMEOUT = 60
_SUCCESS_EXIT_TIMEOUT = 120

# Skip cleanly where the CLI / configurations tree is unavailable.
_skip_reason = (
    "gbserver CLI not on PATH"
    if shutil.which("gbserver") is None
    else (
        "configurations/spaces/local space not found" if not _SPACE_DIR.exists() else ""
    )
)
pytestmark = [
    pytest.mark.standalone,
    pytest.mark.skipif(bool(_skip_reason), reason=_skip_reason),
]


class TestBuildRunnerCliSignals(AbstractSingletonStorageUsingTest):
    """Drive the real build-runner CLI and assert the persisted build status.

    Uses SQLite storage so the parent and the STANDALONE subprocess share one
    backend. The base class supplies the per-method table prefix (handed to the
    subprocess via ``--gb-admin-table-prefix``) and the storage handle used to
    register the space and read the final status. GB_HOME_DIR is isolated to a
    private temp db per test; teardown discards that whole db (rmtree) rather than
    dropping tables one-by-one, because DROP TABLE wedges on a SQLite file the
    subprocess wrote to concurrently (a FileLock/lock interaction the base class's
    per-table teardown loop cannot clear).
    """

    @classmethod
    def _get_storage_factory(cls) -> StorageFactory:
        """Force SQLite so this process shares a backend with the subprocess."""
        return SqliteStorageFactory()

    def setup_method(self, method):
        """Set up the prefixed storage and register the local space as 'public'.

        Isolates GB_HOME_DIR to a private temp dir *before* the base class builds
        the SQLite storage, so this test uses its own db rather than the shared
        ``~/.granite.build`` one (which a running ``gbserver standalone`` server
        could otherwise hold locked, deadlocking teardown's DROP TABLE). Beyond
        the base setup (unique prefix + empty tables), it records the prefix to
        pass to the subprocess and registers the local space so the CLI's
        ``load_build`` can resolve it by name.
        """
        self._prev_gb_home = os.environ.get("GB_HOME_DIR")
        self._gb_home = tempfile.mkdtemp(prefix="gb_cli_sig_")
        os.environ["GB_HOME_DIR"] = self._gb_home

        super().setup_method(method)
        table = self.storage.build_storage.get_table_name()
        assert table.endswith(GB_BUILDS_TABLE_NAME)
        self.table_prefix = table[: -len(GB_BUILDS_TABLE_NAME)]
        self.storage.space_storage.add(
            [
                StoredSpace(
                    name=PUBLIC_SPACE_NAME,
                    git_repo_uri=f"file://{_SPACE_DIR}",
                    lakehouse_namespace="",
                )
            ]
        )

    def teardown_method(self, method):
        """Discard the whole isolated db and restore GB_HOME_DIR.

        Deliberately does NOT call the base class ``teardown_method`` (whose
        per-table DROP loop hangs on a db the subprocess wrote to). Instead it
        disposes pooled connections so no fds linger on the temp db, deletes the
        temp home (removing all prefixed tables at once), and restores the prior
        GB_HOME_DIR. ``self.storage`` is reset so the base class's next
        ``setup_method`` (which asserts it is None) is satisfied.
        """
        get_singleton_engine_cache().dispose_all()
        self.storage = None
        # Don't leave a singleton pointing at the temp db we're about to delete.
        reset_admin_storage()
        if self._prev_gb_home is None:
            os.environ.pop("GB_HOME_DIR", None)
        else:
            os.environ["GB_HOME_DIR"] = self._prev_gb_home
        shutil.rmtree(self._gb_home, ignore_errors=True)

    def _run_build_runner(
        self, build_dir: Path, tmp_path: Path, sig: Optional[int] = None
    ) -> int:
        """Run ``gbserver build-runner`` on a build dir, optionally signalling it.

        Launches the CLI against ``build_dir`` with the test's admin-table prefix
        and ``--space-dir`` pointed at the local space. When ``sig`` is given, waits
        for the build to reach the running state (marker in the log file) then
        delivers the signal. Blocks until the process exits.

        Args:
            build_dir: directory holding the build.yaml to run.
            tmp_path: pytest temp dir for the isolated workspace and log file.
            sig: optional signal number (SIGINT/SIGTERM) to send once running.

        Returns:
            The CLI process's exit code.

        Raises:
            AssertionError: if the build never reaches the running state.
            subprocess.TimeoutExpired: if the process does not exit in time.
        """
        log_file = tmp_path / "run.log"
        env = {
            **os.environ,
            "GB_ENVIRONMENT": "STANDALONE",
            # Share the parent's isolated SQLite db (same prefixed tables).
            "GB_HOME_DIR": self._gb_home,
            "GBSERVER_LOG_FILE": str(log_file),
            # No message broker runs in this test, so disable event publishing.
            "GBSERVER_EVENT_PUBLISHING_ENABLED": "false",
        }
        proc = subprocess.Popen(
            [
                "gbserver",
                "--gb-admin-table-prefix",
                self.table_prefix,
                "build-runner",
                "--build-dir",
                str(build_dir),
                "--space-dir",
                str(_SPACE_DIR),
                "--workspace-dir",
                str(tmp_path / "ws"),
                "--monitoring-interval",
                "5",
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            exit_timeout = _SUCCESS_EXIT_TIMEOUT
            if sig is not None:
                assert self._wait_for_running(
                    log_file, _MARKER_TIMEOUT
                ), "build never reached the running state"
                time.sleep(1.0)  # let the workload settle before interrupting
                proc.send_signal(sig)
                exit_timeout = _SIGNAL_EXIT_TIMEOUT
            proc.wait(timeout=exit_timeout)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=30)
        return proc.returncode

    @staticmethod
    def _wait_for_running(log_file: Path, timeout: float) -> bool:
        """Poll the gbserver log file until the build-is-running marker appears.

        Args:
            log_file: the GBSERVER_LOG_FILE the build-runner process writes to.
            timeout: max seconds to wait for the marker.

        Returns:
            True if the marker was seen within the timeout, else False.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if log_file.exists() and _RUNNING_MARKER in log_file.read_text(
                errors="ignore"
            ):
                return True
            time.sleep(0.5)
        return False

    def _final_build_status(self) -> Status:
        """Return the persisted status of the build produced by the CLI run.

        Returns:
            The single stored build's ``Status`` (from the prefixed tables shared
            with the subprocess).

        Raises:
            AssertionError: if not exactly one build was persisted.
        """
        builds = self.storage.build_storage.get_by_where(
            {"name": COMMAND_RUN_BUILD_WATCH_BUILD_NAME}
        )
        assert len(builds) == 1, f"expected exactly one build, got {len(builds)}"
        assert isinstance(builds[0], StoredBuild)
        return builds[0].status

    @pytest.mark.timeout(180)
    def test_cli_build_succeeds(self, tmp_path):
        """A clean 1-step bash build run via the CLI completes SUCCESS, exit 0."""
        rc = self._run_build_runner(_SUCCESS_BUILD_DIR, tmp_path)
        assert self._final_build_status() == Status.SUCCESS
        assert rc == 0

    @pytest.mark.timeout(180)
    def test_cli_sigint_cancels_build(self, tmp_path):
        """SIGINT (Ctrl+C) during a running build marks it CANCELLED, exit 0."""
        rc = self._run_build_runner(_LONGRUNNING_BUILD_DIR, tmp_path, sig=signal.SIGINT)
        assert self._final_build_status() == Status.CANCELLED
        # A deliberate cancel is a clean outcome.
        assert rc == 0

    @pytest.mark.timeout(180)
    def test_cli_sigterm_fails_build(self, tmp_path):
        """SIGTERM during a running build marks it FAILED and exits non-zero."""
        rc = self._run_build_runner(
            _LONGRUNNING_BUILD_DIR, tmp_path, sig=signal.SIGTERM
        )
        assert self._final_build_status() == Status.FAILED
        # A failed build must not report success to callers.
        assert rc != 0

    @pytest.mark.timeout(180)
    def test_cli_sigterm_does_not_retry_retryable_build(self, tmp_path):
        """SIGTERM on a retry-enabled build fails it WITHOUT spawning a retry.

        Regression: stop_and_fail() writes FAILED, which _should_retry() treats as
        retryable (max_retries > 0). Without the _stop_requested guard in the
        start_and_wait retry loop, SIGTERM would start a new build run instead of
        terminating. A spawned retry would create a second build with the same
        name, so asserting exactly one build (and it FAILED) proves no retry ran.
        """
        rc = self._run_build_runner(
            _LONGRUNNING_RETRY_BUILD_DIR, tmp_path, sig=signal.SIGTERM
        )
        builds = self.storage.build_storage.get_by_where(
            {"name": COMMAND_RUN_BUILD_WATCH_BUILD_NAME}
        )
        assert len(builds) == 1, f"SIGTERM spawned a retry: {len(builds)} builds exist"
        assert builds[0].status == Status.FAILED
        assert rc != 0
