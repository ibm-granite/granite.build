#!/usr/bin/env python3

"""Unit tests for AppWrapperMonitor time-based API failure threshold."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes_asyncio import client

from gbserver.monitoring.appwrapper_monitor import AppWrapperMonitor
from gbserver.types.buildevent import BuildEventType, EntityRunMetadata


def _make_monitor(poll=1.0):
    """Create an AppWrapperMonitor with mocked dependencies."""
    entityrun_metadata = EntityRunMetadata(
        build_id="test-build-id",
        username="test-user",
        type="TargetStepRun",
        target_name="test-target",
        targetrun_id="test-targetrun-id",
        targetsteprun_id="test-targetsteprun-id",
        targetstep_uri="space://steps/test",
        target_step_index=0,
    )
    event_queue = asyncio.Queue()
    stop_event = asyncio.Event()
    monitor = AppWrapperMonitor(
        name="test-appwrapper",
        namespace="test-ns",
        poll=poll,
        launch_id="test-launch-id",
        entityrun_metadata=entityrun_metadata,
        event_queue=event_queue,
        stop_event=stop_event,
    )
    return monitor, event_queue, stop_event


class TestTimeBasedThreshold:
    """Tests for time-based API failure threshold."""

    def test_api_failure_start_time_initialized_none(self):
        """_api_failure_start_time starts as None."""
        monitor, _, _ = _make_monitor()
        assert monitor._api_failure_start_time is None

    @pytest.mark.asyncio
    async def test_single_timeout_records_start_time(self):
        """First API failure records the start time."""
        monitor, _, _ = _make_monitor()
        monitor.custom_api = MagicMock()
        monitor.custom_api.get_namespaced_custom_object = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )
        result = await monitor._get_appwrapper_status()
        assert result == "Running"
        assert monitor._api_failure_start_time is not None

    @pytest.mark.asyncio
    async def test_successful_call_resets_start_time(self):
        """Successful API call resets _api_failure_start_time to None."""
        monitor, _, _ = _make_monitor()
        monitor._api_failure_start_time = 100.0
        monitor.custom_api = MagicMock()
        monitor.custom_api.get_namespaced_custom_object = AsyncMock(
            return_value={"status": {"phase": "Running"}}
        )
        result = await monitor._get_appwrapper_status()
        assert result == "Running"
        assert monitor._api_failure_start_time is None

    def test_is_api_failure_timeout_false_when_none(self):
        """No timeout if no failures recorded."""
        monitor, _, _ = _make_monitor()
        assert monitor._is_api_failure_timeout() is False

    def test_is_api_failure_timeout_false_before_threshold(self):
        """No timeout if duration hasn't exceeded threshold."""
        monitor, _, _ = _make_monitor()
        monitor._api_failure_start_time = time.monotonic() - 10  # 10s ago
        assert monitor._is_api_failure_timeout() is False

    def test_is_api_failure_timeout_true_after_threshold(self):
        """Timeout if duration exceeds threshold."""
        monitor, _, _ = _make_monitor()
        monitor._api_failure_start_time = (
            time.monotonic() - 400
        )  # 400s ago (> 300 default)
        assert monitor._is_api_failure_timeout() is True

    def test_unpause_resets_api_failure_start_time(self):
        """unpause() resets _api_failure_start_time to None."""
        monitor, _, _ = _make_monitor()
        monitor._api_failure_start_time = 100.0
        monitor.unpause()
        assert monitor._api_failure_start_time is None


class TestPodLivenessCheck:
    """Tests for pod liveness verification before declaring fatal."""

    @pytest.mark.asyncio
    async def test_pods_running_returns_true(self):
        """If any pod is Running, returns True."""
        monitor, _, _ = _make_monitor()
        monitor.v1 = MagicMock()

        mock_pod = MagicMock()
        mock_pod.status.phase = "Running"
        mock_pod.metadata.name = "test-pod-1"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        monitor.v1.list_namespaced_pod = AsyncMock(return_value=mock_pod_list)

        result = await monitor._check_pod_liveness()
        assert result is True

    @pytest.mark.asyncio
    async def test_no_running_pods_returns_false(self):
        """If no pods are Running, returns False."""
        monitor, _, _ = _make_monitor()
        monitor.v1 = MagicMock()

        mock_pod = MagicMock()
        mock_pod.status.phase = "Failed"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        monitor.v1.list_namespaced_pod = AsyncMock(return_value=mock_pod_list)

        result = await monitor._check_pod_liveness()
        assert result is False

    @pytest.mark.asyncio
    async def test_pod_api_failure_returns_false(self):
        """If pod API also fails, returns False (allow fatal)."""
        monitor, _, _ = _make_monitor()
        monitor.v1 = MagicMock()
        monitor.v1.list_namespaced_pod = AsyncMock(side_effect=asyncio.TimeoutError())

        result = await monitor._check_pod_liveness()
        assert result is False

    @pytest.mark.asyncio
    async def test_empty_pod_list_returns_false(self):
        """If pod list is empty, returns False."""
        monitor, _, _ = _make_monitor()
        monitor.v1 = MagicMock()

        mock_pod_list = MagicMock()
        mock_pod_list.items = []
        monitor.v1.list_namespaced_pod = AsyncMock(return_value=mock_pod_list)

        result = await monitor._check_pod_liveness()
        assert result is False

    @pytest.mark.asyncio
    async def test_liveness_check_resets_timer_when_pods_alive(self):
        """When pods are alive, the timeout timer should be reset."""
        monitor, event_queue, stop_event = _make_monitor()
        monitor.v1 = MagicMock()
        monitor.custom_api = MagicMock()

        # Simulate pods alive
        mock_pod = MagicMock()
        mock_pod.status.phase = "Running"
        mock_pod.metadata.name = "test-pod-1"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        monitor.v1.list_namespaced_pod = AsyncMock(return_value=mock_pod_list)

        # Set timer as if it expired
        monitor._api_failure_start_time = time.monotonic() - 400

        # Call _check_pod_liveness and verify it returns True
        result = await monitor._check_pod_liveness()
        assert result is True


class TestWorkloadStatusTransientErrors:
    """_get_workload_status tolerates transient apiserver errors.

    This runs while assembling a state-change payload, outside the monitor
    loop's transient handling. A 429/5xx that outlived the transport-retry
    budget must not crash the build: it is recorded as an API failure and
    reported as empty status for that poll, rather than propagating.
    """

    @pytest.mark.asyncio
    async def test_transient_429_returns_empty_and_records_failure(self):
        """A 429 on the workload list is tolerated, not raised."""
        monitor, _, _ = _make_monitor()
        monitor.custom_api = MagicMock()
        monitor.custom_api.list_namespaced_custom_object = AsyncMock(
            side_effect=client.ApiException(status=429, reason="Too Many Requests")
        )
        result = await monitor._get_workload_status()
        assert result == []
        assert monitor._api_failure_start_time is not None

    @pytest.mark.asyncio
    async def test_transient_503_on_status_call_is_tolerated(self):
        """A 503 while fetching per-workload status is tolerated too."""
        monitor, _, _ = _make_monitor()
        monitor.custom_api = MagicMock()
        # List succeeds and returns one owned workload...
        monitor.custom_api.list_namespaced_custom_object = AsyncMock(
            return_value={
                "items": [
                    {
                        "metadata": {
                            "name": "wl-1",
                            "ownerReferences": [
                                {
                                    "apiVersion": "workload.codeflare.dev/v1beta2",
                                    "name": monitor.name,
                                }
                            ],
                        }
                    }
                ]
            }
        )
        # ...but the per-workload status call is throttled.
        monitor.custom_api.get_namespaced_custom_object_status = AsyncMock(
            side_effect=client.ApiException(status=503, reason="Service Unavailable")
        )
        result = await monitor._get_workload_status()
        assert result == []
        assert monitor._api_failure_start_time is not None

    @pytest.mark.asyncio
    async def test_non_transient_error_propagates(self):
        """A non-transient ApiException (e.g. 401) still propagates."""
        monitor, _, _ = _make_monitor()
        monitor.custom_api = MagicMock()
        monitor.custom_api.list_namespaced_custom_object = AsyncMock(
            side_effect=client.ApiException(status=401, reason="Unauthorized")
        )
        with pytest.raises(client.ApiException):
            await monitor._get_workload_status()


def _pod(name, phase="Failed", reason=None, conditions=()):
    """A minimal K8s pod object for the disruption checks."""
    pod = MagicMock()
    pod.metadata.name = name
    pod.status.phase = phase
    pod.status.reason = reason
    pod.status.container_statuses = []
    pod.status.conditions = [
        MagicMock(type=ctype, status=cstatus, reason=creason)
        for ctype, cstatus, creason in conditions
    ]
    return pod


class TestDisruptionTracking:
    """The monitor records, per pod, whether the cluster disrupted it, so a
    `Failed` verdict can tell a disruption from a workload failure."""

    @pytest.mark.parametrize(
        "pod, expected",
        [
            (
                _pod(
                    "p",
                    conditions=[("DisruptionTarget", "True", "PreemptionByScheduler")],
                ),
                "PreemptionByScheduler",
            ),
            (
                _pod(
                    "p",
                    conditions=[("DisruptionTarget", "True", "EvictionByEvictionAPI")],
                ),
                "EvictionByEvictionAPI",
            ),
            (_pod("p", reason="Evicted"), "Evicted"),
            (_pod("p", conditions=[("DisruptionTarget", "False", "x")]), None),
            (_pod("p", conditions=[("Ready", "False", "PodFailed")]), None),
        ],
    )
    def test_pod_disruption_reason(self, pod, expected):
        assert AppWrapperMonitor._pod_disruption_reason(pod) == expected

    @pytest.mark.asyncio
    async def test_failed_pods_poll_records_disruption(self):
        monitor, _, _ = _make_monitor()
        monitor.v1 = MagicMock()
        pods = MagicMock()
        pods.items = [
            _pod(
                "p-1", conditions=[("DisruptionTarget", "True", "TerminationByKubelet")]
            ),
            _pod("p-2"),
        ]
        monitor.v1.list_namespaced_pod = AsyncMock(return_value=pods)
        monitor._collect_pod_logs = AsyncMock(return_value={})
        await monitor._get_appwrapper_failed_pods()
        assert monitor._disrupted_pods == {"p-1": "TerminationByKubelet"}
        assert set(monitor.failed_pods) == {"p-1", "p-2"}

    @pytest.mark.asyncio
    async def test_preemption_event_records_disruption(self):
        """A pod deleted before a poll saw it leaves only its Preempted event."""
        monitor, _, _ = _make_monitor()
        monitor.v1 = MagicMock()
        event = MagicMock()
        event.type = "Normal"
        event.reason = "Preempted"
        event.message = "Preempted by pod x on node y"
        event.involved_object.name = "gbtest-0"
        event.involved_object.kind = "Pod"
        event.metadata.uid = "uid-1"
        event.last_timestamp = None
        event.event_time = None
        events_list = MagicMock()
        events_list.items = [event]
        monitor.v1.list_namespaced_event = AsyncMock(return_value=events_list)
        monitor.launched_pods = {"gbtest-0": MagicMock()}

        await monitor._get_new_events([])
        assert monitor._disrupted_pods == {"gbtest-0": "Preempted"}

    def test_unpause_resets_disruptions_and_bumps_generation(self):
        monitor, _, _ = _make_monitor()
        monitor._disrupted_pods = {"p": "Evicted"}
        generation = monitor._generation
        monitor.unpause()
        assert monitor._disrupted_pods == {}
        assert monitor._generation == generation + 1

    @pytest.mark.asyncio
    async def test_state_change_payload(self):
        """Payload `max_retries` is the real retryLimit, not the constant
        'unlimited', and the payload carries the disrupted pods."""
        monitor, _, _ = _make_monitor()
        monitor.custom_api = MagicMock()
        monitor.custom_api.get_namespaced_custom_object = AsyncMock(
            return_value={
                "metadata": {
                    "annotations": {"workload.codeflare.dev.appwrapper/retryLimit": "3"}
                },
                "status": {"phase": "Failed", "resettingCount": 1},
            }
        )
        await monitor._get_appwrapper_status()
        monitor._get_workload_status = AsyncMock(return_value=[])
        monitor._get_appwrapper_failed_pods = AsyncMock(return_value=None)
        monitor._get_new_events = AsyncMock(return_value=[])
        monitor._disrupted_pods = {"p-1": "PreemptionByScheduler"}

        msg = await monitor._get_state_change("Failed", get_state_for_exception=True)
        data = json.loads(msg.split("```json\n", 1)[1].rsplit("\n```", 1)[0])

        assert data["max_retries"] == "3"
        assert data["disrupted_pods"] == {"p-1": "PreemptionByScheduler"}


class TestFailedGracePeriod:
    """On `Failed` the monitor stops after the grace period -- unless the
    RetryHandler relaunched the workload meanwhile, which it must keep watching."""

    async def _run(self, monitor, statuses, on_failed=None):
        """Run monitor() over `statuses`, calling on_failed when Failed is seen."""
        seen = []

        async def status():
            state = statuses.pop(0) if statuses else "Running"
            seen.append(state)
            if state == "Failed" and on_failed:
                on_failed()
            return state

        api_cm = MagicMock()
        api_cm.__aenter__ = AsyncMock(return_value=MagicMock())
        api_cm.__aexit__ = AsyncMock(return_value=False)
        monitor._get_appwrapper_status = status
        monitor._get_appwrapper_failed_pods = AsyncMock(return_value=None)
        monitor._publish_state_change = AsyncMock(return_value=None)
        with (
            patch(
                "gbserver.monitoring.appwrapper_monitor.AtomicApiClient.create_api_client",
                AsyncMock(return_value=api_cm),
            ),
            patch(
                "gbserver.monitoring.appwrapper_monitor.GBSERVER_MONITORING_GRACE_PERIOD",
                0.2,
            ),
        ):
            await asyncio.wait_for(monitor.monitor(), timeout=5.0)
        return seen

    @pytest.mark.asyncio
    async def test_failed_without_relaunch_stops(self):
        monitor, _, stop_event = _make_monitor(poll=0.01)
        seen = await self._run(monitor, ["Running", "Failed", "Running"])
        assert seen == ["Running", "Failed"]
        assert stop_event.is_set()

    @pytest.mark.asyncio
    async def test_relaunch_during_grace_keeps_watching(self):
        monitor, _, _ = _make_monitor(poll=0.01)
        loop = asyncio.get_running_loop()

        def relaunch():  # what K8s.retry_workload does: pause, reinstall, unpause
            loop.call_later(0.02, monitor.pause)
            loop.call_later(0.1, monitor.unpause)

        seen = await self._run(
            monitor, ["Running", "Failed", "Running", "Succeeded"], on_failed=relaunch
        )
        assert seen == ["Running", "Failed", "Running", "Succeeded"]

    @pytest.mark.asyncio
    async def test_relaunch_still_in_progress_after_grace_waits(self):
        monitor, _, _ = _make_monitor(poll=0.01)
        loop = asyncio.get_running_loop()

        def slow_relaunch():  # reinstall outlasts the grace period
            loop.call_later(0.02, monitor.pause)
            loop.call_later(0.5, monitor.unpause)

        seen = await self._run(
            monitor, ["Failed", "Running", "Succeeded"], on_failed=slow_relaunch
        )
        assert seen == ["Failed", "Running", "Succeeded"]
