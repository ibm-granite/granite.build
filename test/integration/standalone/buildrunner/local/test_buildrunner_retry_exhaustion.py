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

A build whose only target hard-fails should be retried exactly ``max_retries``
times — producing ``1 + max_retries`` build records total, all FAILED — and then
stop.

This must run through the *BuildWatcher* (not a directly-driven BuildRunner):
the bug being guarded against is that ``BuildRunner.__prepare_retry`` persists
each retry build as PENDING, while the BuildRunner's in-process retry loop *also*
runs it.  The watcher polls for PENDING builds and dispatches a *second* runner
for that same retry build, so each failure spawns more than one next-level retry
— a branching tree of retries instead of a linear chain.  Driving the build
directly via a BuildRunner would never expose this, because no watcher is polling.

The build runs in the local Bash environment, so no Docker/cluster is required.
"""

from pathlib import Path
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

# Statuses that mean a build (or its retry) is still in flight; the retry chain
# has "settled" only once no build is in any of these. RETRY is in-flight: it is
# a retry build the in-process loop is about to run.
_IN_FLIGHT = {
    Status.SUBMITTED,
    Status.PENDING,
    Status.RUNNING,
    Status.CANCEL_REQUESTED,
    Status.RETRY_PENDING,
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

        Expectation: exactly ``1 + max_retries`` build records exist, all FAILED,
        with retry_count values 0..max_retries (a linear chain).  The bug produces
        *more* than that because the watcher double-dispatches each PENDING retry.
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
        self.storage.build_storage.add(stored_build)

        watcher = BuildWatcher(gh_token="", all_build_space_uri=spec.space_uri)
        # Thread runner (no cluster) and fast (1s) polling so the watcher reliably
        # observes retry builds during their brief PENDING window. 1s is the minimum
        # interval (sub-second values busy-loop and are floored); the chain-settle
        # wait below gives it up to spec.timeout_minutes to observe everything.
        watcher.config.buildrunner_type = "thread"
        watcher.config.monitoring_interval = 1

        thread = ExceptionRaisingThread(
            name="BuildWatcher", target=watcher.start_and_wait, args=()
        )
        thread.start()
        try:
            self._wait_until_chain_settles(
                timeout_seconds=spec.timeout_minutes * 60,
                max_retries=max_retries,
            )
        finally:
            watcher.stop()
            thread.join(timeout=60)

        builds = self.storage.build_storage.get_by_uuid(None) or []
        retry_counts = sorted(b.retry_count for b in builds)
        statuses = [b.status for b in builds]

        assert len(builds) == 1 + max_retries, (
            f"Expected a linear retry chain of {1 + max_retries} builds "
            f"(1 original + {max_retries} retries), but found {len(builds)} "
            f"with retry_counts={retry_counts}. More than expected indicates the "
            f"BuildWatcher is double-dispatching PENDING retry builds."
        )
        assert all(
            s == Status.FAILED for s in statuses
        ), f"Every build in the chain should be FAILED, got statuses={statuses}"
        assert retry_counts == list(range(0, max_retries + 1)), (
            f"Retry chain should have one build per retry_count 0..{max_retries}, "
            f"got {retry_counts}"
        )

    def _wait_until_chain_settles(
        self, timeout_seconds: float, max_retries: int
    ) -> None:
        """Block until the retry chain has genuinely exhausted its retries.

        The chain is settled only once the final allowed attempt
        (``retry_count == max_retries``) has reached a terminal state and no
        build is still in flight.

        Keying on the exhausting attempt is required to avoid a race: between a
        build's ``RUNNING -> FAILED`` finalization and the next retry being
        persisted as ``RETRY_PENDING`` there is a brief window where every build
        looks terminal and the count is momentarily stable.  A "no in-flight +
        stable count" heuristic can return during that window — before the last
        retry runs — and the subsequent ``watcher.stop()`` then tears down the
        BuildRunner mid-chain, cancelling the pending retry (and re-marking the
        whole chain CANCELLED). Waiting for the ``max_retries`` attempt to become
        terminal removes that window, since no further retry can be spawned.

        Args:
            timeout_seconds: Maximum time to wait for the chain to settle.
            max_retries: The configured retry ceiling; the chain is exhausted
                once a build with this ``retry_count`` is terminal.

        Raises:
            AssertionError: if the chain has not settled before the timeout.
        """
        poll = 1.0
        start = time()
        while time() - start <= timeout_seconds:
            builds = self.storage.build_storage.get_by_uuid(None) or []
            in_flight = [b for b in builds if b.status in _IN_FLIGHT]
            exhausted = any(
                b.retry_count == max_retries and b.status.is_finished() for b in builds
            )
            if exhausted and not in_flight:
                return
            sleep(poll)
        assert False, (
            f"Retry chain did not exhaust {max_retries} retries within "
            f"{timeout_seconds}s; last seen {len(builds)} build(s)."
        )
