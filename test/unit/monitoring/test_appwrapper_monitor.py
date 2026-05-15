#!/usr/bin/env python3

"""Unit tests for AppWrapperMonitor time-based API failure threshold."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
