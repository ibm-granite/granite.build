from datetime import timedelta
from typing import Self

import pytest
from gbserver_test.storage.storage import (
    AbstractExistingDataReadTest,
    AbstractStorageTest,
    AbstractStorageTestSupport,
)
from gbserver_test.test_utils import AbstractSingletonStorageUsingTest

from gbserver.storage import singleton_storage
from gbserver.storage.storage import BaseItemStorage
from gbserver.storage.stored_node_failure import StoredNodeFailure
from gbserver.utils.utils import get_utc_time


class NodeFailureStorageTestSupport(AbstractStorageTestSupport):

    def __init__(self):
        super().__init__(sort_column="node_name")

    def _get_test_item(self, index):
        obj = StoredNodeFailure(
            node_name=f"worker-node-{index}",
            build_id=f"build-{index}",
            launch_id=f"launch-{index}",
            failure_type=f"FailedMount-{index}",
            retry_count=index,
        )
        return obj


class BaseNodeFailureStorageTest(AbstractStorageTest):

    @classmethod
    def _get_test_config(cls) -> AbstractStorageTestSupport:
        return NodeFailureStorageTestSupport()

    def _get_tested_storage(self) -> BaseItemStorage:
        return self.storage.node_failure_storage


class BaseLegacyNodeFailureTest(AbstractExistingDataReadTest):

    def _get_tested_readonly_storage(
        self, storage: singleton_storage.SingletonAdminStorage
    ):
        return storage.node_failure_storage


class TestNodeFailureQueryMethods(AbstractSingletonStorageUsingTest):
    """Tests for query methods on BaseNodeFailureStorage."""

    def _add(self: Self, **kwargs) -> StoredNodeFailure:
        """Helper to add a failure to storage."""
        defaults = {
            "node_name": "node-1",
            "build_id": "build-1",
            "launch_id": "launch-1",
            "failure_type": "FailedMount",
        }
        defaults.update(kwargs)
        item = StoredNodeFailure(**defaults)
        self.storage.node_failure_storage.add(item)
        return item

    def test_get_recent_failures(self: Self) -> None:
        """Test get_recent_failures returns unresolved failures within window."""
        nfs = self.storage.node_failure_storage

        self._add(node_name="node-a", build_id="b1")
        self._add(node_name="node-a", build_id="b2")
        self._add(node_name="node-b", build_id="b3")

        result = nfs.get_recent_failures("node-a", minutes=30)
        assert len(result) == 2
        assert all(f.node_name == "node-a" for f in result)

    def test_get_recent_failures_excludes_old(self: Self) -> None:
        """Test that failures outside the time window are excluded."""
        nfs = self.storage.node_failure_storage

        old = self._add(node_name="node-a", build_id="b-old")
        # Backdate to 2 hours ago
        old_time = get_utc_time() - timedelta(hours=2)
        nfs.update_fields(
            old.uuid, {"created_time": old_time}, update_updated_time=False
        )

        self._add(node_name="node-a", build_id="b-recent")

        result = nfs.get_recent_failures("node-a", minutes=30)
        assert len(result) == 1
        assert result[0].build_id == "b-recent"

    def test_recent_excludes_resolved(self: Self) -> None:
        """Test that resolved failures are excluded."""
        nfs = self.storage.node_failure_storage

        self._add(node_name="node-a", build_id="b1")
        nfs.resolve_node_failures("node-a")

        self._add(node_name="node-a", build_id="b2")

        result = nfs.get_recent_failures("node-a", minutes=30)
        assert len(result) == 1
        assert result[0].build_id == "b2"

    def test_get_failure_summary(self: Self) -> None:
        """Test get_failure_summary groups by node with stats."""
        nfs = self.storage.node_failure_storage

        self._add(
            node_name="node-a",
            build_id="b1",
            failure_type="FailedMount",
            metadata={"namespace": "ns1", "cluster": "c1"},
        )
        self._add(
            node_name="node-a",
            build_id="b2",
            failure_type="NCCLError",
            metadata={"namespace": "ns1", "cluster": "c1"},
        )
        self._add(
            node_name="node-b",
            build_id="b3",
            failure_type="FailedMount",
            metadata={"namespace": "ns2"},
        )

        summary = nfs.get_failure_summary(alert_window_minutes=30)

        assert "node-a" in summary
        assert "node-b" in summary
        assert summary["node-a"]["total_failures"] == 2
        assert summary["node-a"]["unique_builds"] == 2
        assert summary["node-a"]["failure_types"]["FailedMount"] == 1
        assert summary["node-a"]["failure_types"]["NCCLError"] == 1
        assert summary["node-a"]["namespaces"] == ["ns1"]
        assert summary["node-a"]["clusters"] == ["c1"]
        assert summary["node-b"]["total_failures"] == 1
        assert summary["node-b"]["namespaces"] == ["ns2"]

    def test_summary_excludes_resolved(self: Self) -> None:
        """Test that resolved failures are excluded from summary."""
        nfs = self.storage.node_failure_storage

        self._add(node_name="node-a", build_id="b1")
        nfs.resolve_node_failures("node-a")

        summary = nfs.get_failure_summary()
        assert "node-a" not in summary

    def test_get_problematic_nodes(self: Self) -> None:
        """Test get_problematic_nodes returns nodes above threshold."""
        nfs = self.storage.node_failure_storage

        for i in range(5):
            self._add(node_name="bad-node", build_id=f"b-{i}")
        self._add(node_name="ok-node", build_id="b-ok")

        result = nfs.get_problematic_nodes(threshold=3, minutes=30)
        assert "bad-node" in result
        assert "ok-node" not in result

    def test_problematic_excludes_resolved(self: Self) -> None:
        """Test that resolved failures don't count toward threshold."""
        nfs = self.storage.node_failure_storage

        for i in range(5):
            self._add(node_name="node-a", build_id=f"b-{i}")
        nfs.resolve_node_failures("node-a")

        result = nfs.get_problematic_nodes(threshold=3, minutes=30)
        assert "node-a" not in result

    def test_resolve_node_failures(self: Self) -> None:
        """Test resolve marks failures as resolved, not deleted."""
        nfs = self.storage.node_failure_storage

        self._add(node_name="node-a", build_id="b1")
        self._add(node_name="node-a", build_id="b2")
        self._add(node_name="node-b", build_id="b3")

        count = nfs.resolve_node_failures("node-a")
        assert count == 2

        # Unresolved query should exclude node-a
        recent_a = nfs.get_recent_failures("node-a", minutes=30)
        assert len(recent_a) == 0

        # node-b should be unaffected
        recent_b = nfs.get_recent_failures("node-b", minutes=30)
        assert len(recent_b) == 1

        # Records should still exist (not deleted)
        all_items = nfs.get_by_where({"node_name": "node-a"})
        assert len(all_items) == 2
        assert all(item.resolved for item in all_items)

    def test_resolve_idempotent(self: Self) -> None:
        """Test that resolving already-resolved failures returns 0."""
        nfs = self.storage.node_failure_storage

        self._add(node_name="node-a", build_id="b1")
        nfs.resolve_node_failures("node-a")

        count = nfs.resolve_node_failures("node-a")
        assert count == 0

    def test_get_unresolved_since(self: Self) -> None:
        """Test get_unresolved_failures_for_node_since."""
        nfs = self.storage.node_failure_storage

        old = self._add(node_name="node-a", build_id="b-old")
        old_time = get_utc_time() - timedelta(hours=2)
        nfs.update_fields(
            old.uuid, {"created_time": old_time}, update_updated_time=False
        )

        self._add(node_name="node-a", build_id="b-recent")

        since = get_utc_time() - timedelta(minutes=30)
        result = nfs.get_unresolved_failures_for_node_since("node-a", since)
        assert len(result) == 1
        assert result[0].build_id == "b-recent"
