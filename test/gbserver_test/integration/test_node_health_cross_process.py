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
End-to-end integration tests for node health cross-process communication.

Simulates the real-world scenario where:
1. BuildWatcher (Process 1) records failures via NodeHealthTracker
2. REST API Server (Process 2) queries failures via storage-backed API
3. Both processes communicate through shared persistent storage
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Self
from unittest.mock import Mock, patch

import pytest
import pytest_asyncio
from gbserver_test.test_utils import AbstractSingletonStorageUsingTest

from gbserver.api.node_health import (
    get_failure_summary,
    get_node_failures,
    get_problematic_nodes,
    health_check,
    resolve_node_failures,
)
from gbserver.resilience import NodeHealthTracker
from gbserver.storage.stored_node_failure import StoredNodeFailure


class TestNodeHealthCrossProcess(AbstractSingletonStorageUsingTest):
    """Test cross-process communication via persistent storage."""

    @pytest_asyncio.fixture
    async def tracker_with_storage(self: Self) -> NodeHealthTracker:
        """Create tracker with storage (simulates BuildWatcher process)."""
        tracker = NodeHealthTracker(
            node_failure_storage=self.storage.node_failure_storage,
            alert_threshold=3,
            alert_window_minutes=30,
        )
        await tracker.start()

        yield tracker

        await tracker.stop()

    @pytest.fixture
    def mock_request(self: Self) -> Mock:
        """Create a mock request object."""
        return Mock()

    @pytest.mark.asyncio
    async def test_write_in_tracker_read_in_api(
        self: Self, tracker_with_storage: NodeHealthTracker, mock_request: Mock
    ) -> None:
        """Test that failures recorded via tracker are readable via API."""
        await tracker_with_storage.record_failure(
            node_name="worker-node-cross-1",
            build_id="build-cross-1",
            launch_id="launch-1",
            failure_type="FailedMount",
            metadata={"test": "cross-process"},
        )
        await tracker_with_storage.record_failure(
            node_name="worker-node-cross-1",
            build_id="build-cross-2",
            launch_id="launch-2",
            failure_type="FailedAttachVolume",
        )

        result = await get_node_failures(
            mock_request, "worker-node-cross-1", minutes=30
        )

        assert result["node_name"] == "worker-node-cross-1"
        assert result["failure_count"] == 2
        failure_types = [f["failure_type"] for f in result["failures"]]
        assert "FailedMount" in failure_types
        assert "FailedAttachVolume" in failure_types

    @pytest.mark.asyncio
    async def test_summary_aggregates_data(
        self: Self, tracker_with_storage: NodeHealthTracker, mock_request: Mock
    ) -> None:
        """Test that summary endpoint aggregates data from storage."""
        nodes = ["worker-node-a", "worker-node-b", "worker-node-c"]
        for i, node in enumerate(nodes):
            for j in range(i + 1):
                await tracker_with_storage.record_failure(
                    node_name=node,
                    build_id=f"build-{i}-{j}",
                    launch_id=f"launch-{i}-{j}",
                    failure_type="FailedMount",
                )

        summary = await get_failure_summary(mock_request)

        assert summary["worker-node-a"]["total_failures"] == 1
        assert summary["worker-node-b"]["total_failures"] == 2
        assert summary["worker-node-c"]["total_failures"] == 3

    @pytest.mark.asyncio
    async def test_problematic_nodes_from_storage(
        self: Self, tracker_with_storage: NodeHealthTracker, mock_request: Mock
    ) -> None:
        """Test problematic nodes endpoint queries from storage."""
        for i in range(5):
            await tracker_with_storage.record_failure(
                node_name="worker-node-problematic",
                build_id=f"build-prob-{i}",
                launch_id=f"launch-prob-{i}",
                failure_type="FailedMount",
            )

        await tracker_with_storage.record_failure(
            node_name="worker-node-ok",
            build_id="build-ok-1",
            launch_id="launch-ok-1",
            failure_type="FailedMount",
        )

        result = await get_problematic_nodes(mock_request, threshold=3, minutes=30)

        assert "worker-node-problematic" in result["problematic_nodes"]
        assert "worker-node-ok" not in result["problematic_nodes"]

    @pytest.mark.asyncio
    async def test_resolve_marks_resolved(
        self: Self, tracker_with_storage: NodeHealthTracker, mock_request: Mock
    ) -> None:
        """Test that resolving failures marks them resolved (not deleted)."""
        for i in range(3):
            await tracker_with_storage.record_failure(
                node_name="worker-node-resolve",
                build_id=f"build-resolve-{i}",
                launch_id=f"launch-resolve-{i}",
                failure_type="FailedMount",
            )

        # Verify failures exist
        result_before = await get_node_failures(
            mock_request, "worker-node-resolve", minutes=30
        )
        assert result_before["failure_count"] == 3

        # Resolve via API
        with patch("gbserver.api.node_health.is_super_admin", return_value=True):
            resolve_result = await resolve_node_failures(
                mock_request, "worker-node-resolve"
            )
            assert resolve_result["status"] == "success"
            assert resolve_result["resolved_count"] == 3

        # Unresolved query should show zero
        result_after = await get_node_failures(
            mock_request, "worker-node-resolve", minutes=30
        )
        assert result_after["failure_count"] == 0

    @pytest.mark.asyncio
    async def test_health_check_verifies_storage(
        self: Self, tracker_with_storage: NodeHealthTracker
    ) -> None:
        """Test that health check verifies storage is set."""
        result = await health_check()
        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_api_reads_direct_storage_write(
        self: Self, tracker_with_storage: NodeHealthTracker, mock_request: Mock
    ) -> None:
        """Test that API reads data written directly to storage."""
        failure = StoredNodeFailure(
            node_name="worker-node-direct-write",
            build_id="build-direct",
            launch_id="launch-1",
            failure_type="FailedMount",
        )
        self.storage.node_failure_storage.add(failure)

        result = await get_node_failures(
            mock_request, "worker-node-direct-write", minutes=30
        )

        assert result["failure_count"] == 1
        assert result["failures"][0]["failure_type"] == "FailedMount"

    @pytest.mark.asyncio
    async def test_time_window_filtering(
        self: Self, tracker_with_storage: NodeHealthTracker, mock_request: Mock
    ) -> None:
        """Test that time window filtering works correctly."""
        nfs = self.storage.node_failure_storage

        # Add an old failure and backdate it
        old_failure = StoredNodeFailure(
            node_name="worker-node-time-test",
            build_id="build-old",
            launch_id="launch-old",
            failure_type="FailedMount",
        )
        nfs.add(old_failure)
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        nfs.update_fields(
            old_failure.uuid,
            {"created_time": old_time},
            update_updated_time=False,
        )

        # Record recent failure via tracker
        await tracker_with_storage.record_failure(
            node_name="worker-node-time-test",
            build_id="build-recent",
            launch_id="launch-recent",
            failure_type="FailedMount",
        )

        # 30-minute window should only get recent
        result = await get_node_failures(
            mock_request, "worker-node-time-test", minutes=30
        )
        assert result["failure_count"] == 1
        assert result["failures"][0]["build_id"] == "build-recent"

        # 3-hour window should get both
        result_all = await get_node_failures(
            mock_request, "worker-node-time-test", minutes=180
        )
        assert result_all["failure_count"] == 2
