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

"""Regression test for build-level retry exhaustion under the BuildWatcher.

With in-place retry a build keeps a single build id across attempts. A build
whose only target hard-fails should be retried exactly ``max_retries`` times —
all on the *same* build record — and then stop, leaving one FAILED build whose
``retry_count == max_retries``.

This must run through the *BuildWatcher* (not a directly-driven BuildRunner): the
bug being guarded against is that ``BuildRunner.__prepare_retry`` could leave the
build in a status the watcher re-dispatches. Retries are re-run in place as
``RUNNING`` (never ``PENDING``) precisely so the watcher — which polls
``SUBMITTED``/``PENDING`` only — does not launch a *second* runner for the build
the in-process retry loop is already running. A second runner would race the first
and push ``retry_count`` past ``max_retries``. Driving the build directly via a
BuildRunner would never expose this, because no watcher is polling.

The build runs in the local Bash environment, so no Docker/cluster is required.
"""

from time import sleep, time

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractBuildTest,
    BuildTestSpecification,
    get_test_data_dir_for,
)
from libgbtest.buildrunner.utils import ExceptionRaisingThread
from libgbtest.constants import GBTEST_USER_NAME

from gbserver.buildwatcher.buildwatcher import BuildWatcher
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.status import Status

pytestmark = pytest.mark.standalone

# Statuses that mean the build is still in flight; it has "settled" only once it
# is in none of these. A retry re-runs the same build in place as RUNNING, so
# RUNNING already covers the between-attempts window.
_IN_FLIGHT = {
    Status.SUBMITTED,
    Status.PENDING,
    Status.RUNNING,
    Status.CANCEL_REQUESTED,
}


@pytest.mark.xdist_group(name="buildwatcher_bash_retry")
class TestBuildWatcherRetryExhaustion(AbstractBuildTest):
    """The BuildWatcher must retry a failing build exactly ``max_retries`` times."""

    def setup_method(self, method):
        # Always run locally via the thread BuildRunner — no cluster login.
        self.run_locally = True
        super().setup_method(method)

    def _get_spec(self) -> BuildTestSpecification:
        return BuildTestSpecification.from_yaml(
            get_test_data_dir_for(__file__) / "retry-exhaust" / "buildtest.yaml"
        )

    def test_build_watcher_stops_after_max_retries(self):
        """Submit a build whose target always fails and let the BuildWatcher run it.

        Expectation: exactly ONE build record (in-place retry reuses the build
        id), FAILED, with ``retry_count == max_retries``. A count above that — or
        more than one build — indicates the watcher double-dispatched the build
        while its in-process retry loop was still running.
        """
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
        max_retries = stored_build.get_build_config().retries.max_retries
        assert max_retries > 0, "fixture must set retries.max_retries > 0"
        build_id = stored_build.uuid
        self.storage.build_storage.add(stored_build)

        watcher = BuildWatcher(gh_token="", all_build_space_uri=spec.space_uri)
        # Thread runner (no cluster) and fast (1s) polling so the watcher reliably
        # observes the build during its PENDING window. 1s is the minimum interval
        # (sub-second values busy-loop and are floored); the settle wait below
        # gives it up to spec.timeout_minutes to observe everything.
        watcher.config.buildrunner_type = "thread"
        watcher.config.monitoring_interval = 1

        thread = ExceptionRaisingThread(
            name="BuildWatcher", target=watcher.start_and_wait, args=()
        )
        thread.start()
        try:
            self._wait_until_retries_settle(
                build_id=build_id,
                timeout_seconds=spec.timeout_minutes * 60,
                max_retries=max_retries,
            )
        finally:
            watcher.stop()
            thread.join(timeout=60)

        builds = self.storage.build_storage.get_by_uuid(None) or []
        assert len(builds) == 1, (
            f"In-place retry must reuse one build id, but found {len(builds)} "
            f"builds. More than one indicates the watcher created/double-dispatched "
            f"extra build records."
        )
        build = builds[0]
        assert build.uuid == build_id
        assert (
            build.status == Status.FAILED
        ), f"Build should end FAILED after exhausting retries, got {build.status}"
        assert build.retry_count == max_retries, (
            f"Build should have retried exactly {max_retries} times, but "
            f"retry_count={build.retry_count}. A higher value indicates the "
            f"watcher double-dispatched the in-flight retry."
        )

    def _wait_until_retries_settle(
        self, build_id: str, timeout_seconds: float, max_retries: int
    ) -> None:
        """Block until the build has genuinely exhausted its retries.

        The build is settled only once it has reached ``retry_count ==
        max_retries``, is in a terminal state, and is no longer in flight.

        Keying on the exhausting attempt is required to avoid a race: between a
        build's ``RUNNING -> FAILED`` finalization and the next retry being re-run
        in place as ``RUNNING`` there is a brief window where the build looks
        terminal. A plain "not in flight" heuristic can return during that window —
        before the last retry runs — and the subsequent ``watcher.stop()`` then
        tears down the BuildRunner mid-retry, cancelling the pending attempt.
        Waiting for the ``max_retries`` attempt to become terminal removes that
        window, since no further retry can be staged.

        Args:
            build_id: The single build id being retried in place.
            timeout_seconds: Maximum time to wait for the build to settle.
            max_retries: The configured retry ceiling; retries are exhausted once
                the build reaches this ``retry_count`` and is terminal.

        Raises:
            AssertionError: if the build has not settled before the timeout.
        """
        poll = 1.0
        start = time()
        while time() - start <= timeout_seconds:
            builds = self.storage.build_storage.get_by_uuid(None) or []
            build = next((b for b in builds if b.uuid == build_id), None)
            if build is not None:
                exhausted = (
                    build.retry_count == max_retries and build.status.is_finished()
                )
                if exhausted and build.status not in _IN_FLIGHT:
                    return
            sleep(poll)
        assert False, (
            f"Build {build_id} did not exhaust {max_retries} retries within "
            f"{timeout_seconds}s."
        )
