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

"""Unit tests for the BuildWatcher "stuck build" recovery logic.

These cover the removal of the old in-memory "seen" lists in favor of a live-thread
dispatch guard, plus the staleness watchdog and its bounded re-dispatch budget. No K8s
or storage backend is required — everything is patched.
"""

import datetime
import threading
from unittest.mock import AsyncMock, MagicMock, patch

from libgbtest.constants import requires_k8s

from gbserver.storage.stored_build import StoredBuild
from gbserver.types.metrics import MetricName
from gbserver.types.status import Status


def _make_watcher(
    max_stuck_redispatches=3, stuck_build_timeout_seconds=900, buildrunner_type="job"
):
    """Build a BuildWatcher instance without running its real __init__."""
    from gbserver.buildwatcher.buildwatcher import BuildWatcher

    with patch.object(BuildWatcher, "__init__", lambda self, *a, **kw: None):
        watcher = BuildWatcher.__new__(BuildWatcher)
    watcher.build_runners = {}
    watcher.build_threads = {}
    watcher.build_pr_threads = {}
    watcher.submitted_in_flight = set()
    watcher.cancel_in_flight = set()
    watcher.stuck_redispatch_counts = {}
    watcher._builds_lock = threading.Lock()
    watcher.exp_mov_avg_processing_delay = 0.0
    watcher.storage = MagicMock()
    config = MagicMock()
    config.max_stuck_redispatches = max_stuck_redispatches
    config.stuck_build_timeout_seconds = stuck_build_timeout_seconds
    config.buildrunner_type = buildrunner_type
    watcher.config = config
    return watcher


def _mock_build(uuid="b1", status=Status.PENDING, updated_time=None):
    build = MagicMock(spec=StoredBuild)
    build.uuid = uuid
    build.status = status
    build.name = "some-build"
    build.source_uri = ""
    build.updated_time = updated_time or datetime.datetime.now(datetime.timezone.utc)
    return build


def _live_thread():
    t = MagicMock()
    t.is_alive.return_value = True
    return t


def _dead_thread():
    t = MagicMock()
    t.is_alive.return_value = False
    return t


BW = "gbserver.buildwatcher.buildwatcher"


class TestPendingDispatchGuard:
    """__process_pending_builds dispatches based on the live-thread record, not a list."""

    def test_pending_with_live_thread_not_dispatched(self):
        watcher = _make_watcher()
        build = _mock_build()
        watcher.build_threads[build.uuid] = _live_thread()
        with (
            patch.object(
                watcher,
                "_BuildWatcher__get_builds_matching_status",
                return_value=[build],
            ),
            patch.object(watcher, "_BuildWatcher__start_build") as start,
        ):
            watcher._BuildWatcher__process_pending_builds()
        start.assert_not_called()

    def test_pending_with_no_thread_dispatched(self):
        watcher = _make_watcher()
        build = _mock_build()
        with (
            patch.object(
                watcher,
                "_BuildWatcher__get_builds_matching_status",
                return_value=[build],
            ),
            patch.object(watcher, "_BuildWatcher__start_build") as start,
        ):
            watcher._BuildWatcher__process_pending_builds()
        start.assert_called_once_with(build)

    def test_pending_with_dead_thread_dispatched(self):
        watcher = _make_watcher()
        build = _mock_build()
        # A dead thread reads as not-in-flight, so the build is re-dispatched.
        watcher.build_threads[build.uuid] = _dead_thread()
        with (
            patch.object(
                watcher,
                "_BuildWatcher__get_builds_matching_status",
                return_value=[build],
            ),
            patch.object(watcher, "_BuildWatcher__start_build") as start,
        ):
            watcher._BuildWatcher__process_pending_builds()
        start.assert_called_once_with(build)

    def test_start_build_exception_counts_but_does_not_fail_under_cap(self):
        watcher = _make_watcher(max_stuck_redispatches=3)
        build = _mock_build()
        with (
            patch.object(
                watcher,
                "_BuildWatcher__get_builds_matching_status",
                return_value=[build],
            ),
            patch.object(
                watcher, "_BuildWatcher__start_build", side_effect=RuntimeError("boom")
            ),
            patch(f"{BW}.finalize_build_status") as finalize,
            patch(f"{BW}.push_stuck_build_metric") as metric,
        ):
            watcher._BuildWatcher__process_pending_builds()
        assert watcher.stuck_redispatch_counts[build.uuid] == 1
        finalize.assert_not_called()
        # a fresh (non-stale) dispatch that throws must NOT emit a re-dispatch metric
        metric.assert_not_called()

    def test_start_build_exception_fails_at_cap(self):
        watcher = _make_watcher(max_stuck_redispatches=2)
        build = _mock_build()
        watcher.stuck_redispatch_counts[build.uuid] = 1  # one attempt already
        with (
            patch.object(
                watcher,
                "_BuildWatcher__get_builds_matching_status",
                return_value=[build],
            ),
            patch.object(
                watcher,
                "_BuildWatcher__start_build",
                side_effect=RuntimeError("missing yaml"),
            ),
            patch(f"{BW}.finalize_build_status") as finalize,
            patch(f"{BW}.push_stuck_build_metric") as metric,
        ):
            watcher._BuildWatcher__process_pending_builds()
        finalize.assert_called_once()
        args = finalize.call_args[0]
        assert args[0] == build.uuid
        assert args[1] == Status.FAILED
        metric.assert_called_once()
        # counter dropped so a resubmit starts fresh
        assert build.uuid not in watcher.stuck_redispatch_counts


class TestPendingStaleness:
    """The single PENDING pass also recovers builds stuck with no live runner."""

    def _stale_build(self, seconds=3600, uuid="b1"):
        old = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            seconds=seconds
        )
        return _mock_build(uuid=uuid, updated_time=old)

    def test_stale_no_thread_under_cap_cleans_and_redispatches(self):
        watcher = _make_watcher(
            max_stuck_redispatches=3, stuck_build_timeout_seconds=900
        )
        build = self._stale_build()
        watcher.build_runners[build.uuid] = MagicMock()
        watcher.build_threads[build.uuid] = _dead_thread()
        with (
            patch.object(
                watcher,
                "_BuildWatcher__get_builds_matching_status",
                return_value=[build],
            ),
            patch.object(watcher, "_BuildWatcher__cleanup_runner_resources") as cleanup,
            patch.object(watcher, "_BuildWatcher__start_build") as start,
            patch(f"{BW}.finalize_build_status") as finalize,
            patch(f"{BW}.push_stuck_build_metric") as metric,
        ):
            watcher._BuildWatcher__process_pending_builds()
        cleanup.assert_called_once_with(build.uuid)
        # cleaned AND re-dispatched in the same pass
        start.assert_called_once_with(build)
        finalize.assert_not_called()
        assert watcher.stuck_redispatch_counts[build.uuid] == 1
        # stale refs cleared before re-dispatch
        assert build.uuid not in watcher.build_threads
        assert build.uuid not in watcher.build_runners
        metric.assert_called_once()

    def test_stale_no_thread_at_cap_fails_no_redispatch(self):
        watcher = _make_watcher(
            max_stuck_redispatches=1, stuck_build_timeout_seconds=900
        )
        build = self._stale_build()
        with (
            patch.object(
                watcher,
                "_BuildWatcher__get_builds_matching_status",
                return_value=[build],
            ),
            patch.object(watcher, "_BuildWatcher__cleanup_runner_resources") as cleanup,
            patch.object(watcher, "_BuildWatcher__start_build") as start,
            patch(f"{BW}.finalize_build_status") as finalize,
            patch(f"{BW}.push_stuck_build_metric") as metric,
        ):
            watcher._BuildWatcher__process_pending_builds()
        finalize.assert_called_once()
        assert finalize.call_args[0][1] == Status.FAILED
        cleanup.assert_called_once_with(build.uuid)
        # at cap: do NOT re-dispatch, and no re-dispatch metric
        start.assert_not_called()
        for call in metric.call_args_list:
            assert call[0][1] != MetricName.STUCK_BUILD_REDISPATCHED

    def test_stale_with_live_thread_left_alone(self):
        watcher = _make_watcher(stuck_build_timeout_seconds=900)
        build = self._stale_build()
        watcher.build_threads[build.uuid] = _live_thread()
        with (
            patch.object(
                watcher,
                "_BuildWatcher__get_builds_matching_status",
                return_value=[build],
            ),
            patch.object(watcher, "_BuildWatcher__cleanup_runner_resources") as cleanup,
            patch.object(watcher, "_BuildWatcher__start_build") as start,
            patch(f"{BW}.finalize_build_status") as finalize,
        ):
            watcher._BuildWatcher__process_pending_builds()
        cleanup.assert_not_called()
        start.assert_not_called()
        finalize.assert_not_called()

    def test_charged_once_per_loop_when_stale_and_throwing(self):
        # regression for the double-charge bug: a stale build re-dispatch that throws
        # is charged exactly once in a single loop, not twice.
        watcher = _make_watcher(
            max_stuck_redispatches=5, stuck_build_timeout_seconds=900
        )
        build = self._stale_build()
        with (
            patch.object(
                watcher,
                "_BuildWatcher__get_builds_matching_status",
                return_value=[build],
            ),
            patch.object(watcher, "_BuildWatcher__cleanup_runner_resources"),
            patch.object(
                watcher,
                "_BuildWatcher__start_build",
                side_effect=RuntimeError("boom"),
            ),
            patch(f"{BW}.finalize_build_status"),
            patch(f"{BW}.push_stuck_build_metric"),
        ):
            watcher._BuildWatcher__process_pending_builds()
        assert watcher.stuck_redispatch_counts[build.uuid] == 1

    def test_thread_runner_cleanup_is_base_noop(self):
        # standalone/thread mode selects the base-class no-op cleanup (no K8s)
        watcher = _make_watcher(buildrunner_type="thread")
        from gbserver.buildrunner.abstractbuildrunner import AbstractBuildRunner
        from gbserver.buildrunner.buildrunner import BuildRunner

        cls = watcher._BuildWatcher__runner_class_for_type()
        assert cls is BuildRunner
        # BuildRunner does not override the no-op cleanup_resources
        assert (
            cls.cleanup_resources.__func__
            is AbstractBuildRunner.cleanup_resources.__func__
        )
        # ...and calling it does nothing / does not raise
        cls.cleanup_resources("any-build-id")

    def test_recovered_build_counter_pruned(self):
        # a build with a lingering counter that is no longer PENDING gets pruned
        watcher = _make_watcher()
        watcher.stuck_redispatch_counts = {"gone": 2, "still-pending": 1}
        still = _mock_build(uuid="still-pending")  # fresh, stays PENDING
        with (
            patch.object(
                watcher,
                "_BuildWatcher__get_builds_matching_status",
                return_value=[still],
            ),
            patch.object(watcher, "_BuildWatcher__start_build"),
        ):
            watcher._BuildWatcher__process_pending_builds()
        assert "gone" not in watcher.stuck_redispatch_counts
        assert watcher.stuck_redispatch_counts["still-pending"] == 1


class TestSubmittedProcessing:
    """SUBMITTED->PENDING flip is guarded by submitted_in_flight, cleared on completion."""

    def test_in_flight_prevents_duplicate_thread(self):
        watcher = _make_watcher()
        build = _mock_build(status=Status.SUBMITTED)
        watcher.submitted_in_flight.add(build.uuid)
        with (
            patch.object(
                watcher,
                "_BuildWatcher__get_builds_matching_status",
                return_value=[build],
            ),
            patch(f"{BW}.threading.Thread") as thread_cls,
        ):
            watcher._BuildWatcher__process_submitted_builds()
        thread_cls.assert_not_called()

    def test_not_in_flight_spawns_thread_and_marks_in_flight(self):
        watcher = _make_watcher()
        build = _mock_build(status=Status.SUBMITTED)
        with (
            patch.object(
                watcher,
                "_BuildWatcher__get_builds_matching_status",
                return_value=[build],
            ),
            patch(f"{BW}.threading.Thread") as thread_cls,
        ):
            watcher._BuildWatcher__process_submitted_builds()
        thread_cls.assert_called_once()
        assert build.uuid in watcher.submitted_in_flight

    def test_process_submitted_build_clears_in_flight_on_success(self):
        watcher = _make_watcher()
        build = _mock_build(status=Status.SUBMITTED)
        watcher.submitted_in_flight.add(build.uuid)
        with (
            patch.object(watcher, "_BuildWatcher__log_submission_delay"),
            patch(f"{BW}.update_stored_build_status", return_value=build),
        ):
            watcher._BuildWatcher__process_submitted_build(build)
        assert build.uuid not in watcher.submitted_in_flight

    def test_process_submitted_build_clears_in_flight_on_exception(self):
        watcher = _make_watcher()
        build = _mock_build(status=Status.SUBMITTED)
        watcher.submitted_in_flight.add(build.uuid)
        with (
            patch.object(watcher, "_BuildWatcher__log_submission_delay"),
            patch(
                f"{BW}.update_stored_build_status", side_effect=RuntimeError("db down")
            ),
        ):
            # Must not raise; the build stays SUBMITTED for a later retry.
            watcher._BuildWatcher__process_submitted_build(build)
        assert build.uuid not in watcher.submitted_in_flight

    def test_rejection_already_pending_is_benign(self):
        watcher = _make_watcher()
        build = _mock_build(status=Status.SUBMITTED)
        current = _mock_build(uuid=build.uuid, status=Status.PENDING)
        watcher.storage.build_storage.get_by_uuid.return_value = current
        with (
            patch.object(watcher, "_BuildWatcher__log_submission_delay"),
            patch(f"{BW}.update_stored_build_status", return_value=None),
            patch(f"{BW}.push_failed_status_update_metric") as metric,
        ):
            watcher._BuildWatcher__process_submitted_build(build)
        # Already PENDING: no race metric, no warning.
        metric.assert_not_called()

    def test_rejection_unexpected_status_pushes_metric(self):
        watcher = _make_watcher()
        build = _mock_build(status=Status.SUBMITTED)
        current = _mock_build(uuid=build.uuid, status=Status.CANCELLED)
        watcher.storage.build_storage.get_by_uuid.return_value = current
        with (
            patch.object(watcher, "_BuildWatcher__log_submission_delay"),
            patch(f"{BW}.update_stored_build_status", return_value=None),
            patch(f"{BW}.push_failed_status_update_metric") as metric,
        ):
            watcher._BuildWatcher__process_submitted_build(build)
        metric.assert_called_once()

    def test_thread_start_failure_clears_in_flight(self):
        # If thread.start() raises, the flip thread never runs its finally, so the guard
        # must be cleared here or the build is stranded SUBMITTED forever.
        watcher = _make_watcher()
        build = _mock_build(status=Status.SUBMITTED)
        with (
            patch.object(
                watcher,
                "_BuildWatcher__get_builds_matching_status",
                return_value=[build],
            ),
            patch(f"{BW}.threading.Thread") as thread_cls,
        ):
            thread_cls.return_value.start.side_effect = RuntimeError("thread limit")
            # Must not raise
            watcher._BuildWatcher__process_submitted_builds()
        assert build.uuid not in watcher.submitted_in_flight


class TestCancelInFlight:
    """A cancel we initiated must not be re-processed by the orphan branch."""

    def _cancel_build(self, uuid="c1"):
        return _mock_build(uuid=uuid, status=Status.CANCEL_REQUESTED)

    def test_tracked_cancel_marks_in_flight_no_force_cancel(self):
        watcher = _make_watcher()
        build = self._cancel_build()
        watcher.build_runners[build.uuid] = MagicMock()
        watcher.build_threads[build.uuid] = MagicMock()  # join() is a no-op mock
        admin = MagicMock()
        with (
            patch(f"{BW}.get_admin_storage", return_value=admin),
            patch.object(watcher, "_BuildWatcher__cleanup_runner_resources") as cleanup,
        ):
            watcher._BuildWatcher__process_cancel_requested_build(build)
        assert build.uuid in watcher.cancel_in_flight
        # runner owns the CANCELLED write; watcher must not force it or clean up
        admin.build_storage.update_fields.assert_not_called()
        cleanup.assert_not_called()

    def test_orphan_suppressed_when_cancel_in_flight(self):
        watcher = _make_watcher()
        build = self._cancel_build()
        watcher.cancel_in_flight.add(build.uuid)  # we already stopped it earlier
        admin = MagicMock()
        with (
            patch(f"{BW}.get_admin_storage", return_value=admin),
            patch.object(watcher, "_BuildWatcher__cleanup_runner_resources") as cleanup,
        ):
            watcher._BuildWatcher__process_cancel_requested_build(build)
        cleanup.assert_not_called()
        admin.build_storage.update_fields.assert_not_called()

    def test_genuine_orphan_forces_cancel(self):
        # empty cancel_in_flight, not tracked -> real orphan (e.g. watcher restart)
        watcher = _make_watcher()
        build = self._cancel_build()
        admin = MagicMock()
        with (
            patch(f"{BW}.get_admin_storage", return_value=admin),
            patch.object(watcher, "_BuildWatcher__cleanup_runner_resources") as cleanup,
        ):
            watcher._BuildWatcher__process_cancel_requested_build(build)
        cleanup.assert_called_once_with(build.uuid)
        admin.build_storage.update_fields.assert_called_once()
        assert (
            admin.build_storage.update_fields.call_args[1]["fields"]["status"]
            == Status.CANCELLED
        )

    def test_cancel_in_flight_pruned_when_no_longer_cancel_requested(self):
        watcher = _make_watcher()
        watcher.cancel_in_flight = {"gone", "still"}
        still = self._cancel_build(uuid="still")
        with (
            patch.object(
                watcher,
                "_BuildWatcher__get_builds_matching_status",
                return_value=[still],
            ),
            patch.object(watcher, "_BuildWatcher__process_cancel_requested_build"),
        ):
            watcher._BuildWatcher__process_cancel_requested_builds()
        assert watcher.cancel_in_flight == {"still"}


JOB = "gbserver.buildrunnerjob.buildrunnerjob"


@requires_k8s
class TestBuildRunnerJobCleanupResources:
    """BuildRunnerJob.cleanup_resources reaps AppWrapper/RayCluster/Job/pods by name+label.

    It imports kubernetes_asyncio (optional 'ibm' extra), so these are skipped when that
    extra is not installed (e.g. the default CI test job).
    """

    def _run_cleanup(self, custom_api, batch_api, core_api):
        from gbserver.buildrunnerjob.buildrunnerjob import BuildRunnerJob

        with (
            patch(f"{JOB}.AtomicApiClient.create_api_client") as mock_api_cls,
            patch(f"{JOB}.BUILDRUNNERJOB_NAMESPACE", "test-ns"),
            patch(f"{JOB}.client.CustomObjectsApi", return_value=custom_api),
            patch(f"{JOB}.client.BatchV1Api", return_value=batch_api),
            patch(f"{JOB}.client.CoreV1Api", return_value=core_api),
        ):
            mock_api = AsyncMock()
            mock_api_cls.return_value = mock_api
            mock_api.__aenter__ = AsyncMock(return_value=mock_api)
            mock_api.__aexit__ = AsyncMock(return_value=False)
            BuildRunnerJob.cleanup_resources("build-xyz")

    def test_deletes_job_and_pods_alongside_aw_and_rc(self):
        custom_api = AsyncMock()
        custom_api.list_namespaced_custom_object = AsyncMock(
            side_effect=[
                {"items": [{"metadata": {"name": "aw-1"}}]},  # AppWrappers
                {"items": [{"metadata": {"name": "rc-1"}}]},  # RayClusters
            ]
        )
        custom_api.delete_namespaced_custom_object = AsyncMock()

        batch_api = AsyncMock()
        batch_api.delete_namespaced_job = AsyncMock()

        core_api = AsyncMock()
        pod = MagicMock()
        pod.metadata.name = "pod-1"
        core_api.list_namespaced_pod = AsyncMock(return_value=MagicMock(items=[pod]))
        core_api.delete_namespaced_pod = AsyncMock()

        self._run_cleanup(custom_api, batch_api, core_api)

        # AppWrapper + RayCluster deleted
        assert custom_api.delete_namespaced_custom_object.await_count == 2
        # Job deleted by deterministic name
        batch_api.delete_namespaced_job.assert_awaited_once()
        assert (
            batch_api.delete_namespaced_job.await_args.kwargs["name"]
            == "gb-build-runner-build-xyz"
        )
        # Pods deleted by label
        core_api.delete_namespaced_pod.assert_awaited_once()

    def test_job_delete_404_does_not_break_pod_cleanup(self):
        custom_api = AsyncMock()
        custom_api.list_namespaced_custom_object = AsyncMock(
            side_effect=[{"items": []}, {"items": []}]
        )
        batch_api = AsyncMock()
        batch_api.delete_namespaced_job = AsyncMock(side_effect=Exception("404"))
        core_api = AsyncMock()
        core_api.list_namespaced_pod = AsyncMock(return_value=MagicMock(items=[]))

        # Must not raise despite the job-delete failure.
        self._run_cleanup(custom_api, batch_api, core_api)

        core_api.list_namespaced_pod.assert_awaited_once()
