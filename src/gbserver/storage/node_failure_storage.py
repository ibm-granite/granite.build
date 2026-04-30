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
Base storage interface and implementation for node failure events.

Provides query methods for node failure data access. These methods are the
primary data access layer used by the REST API and other consumers.
"""

import datetime
from typing import Any, Dict, List

from gbserver.storage.storage import (
    CREATED_TIME_FIELD_NAME,
    BaseItemStorage,
    IItemStorage,
)
from gbserver.storage.stored_node_failure import StoredNodeFailure
from gbserver.types.constants import GB_NODE_FAILURES_TABLE_NAME
from gbserver.utils.utils import get_utc_time

_PAGE_SIZE = 100


class INodeFailureStorage(IItemStorage[StoredNodeFailure]):
    """Interface for node failure storage implementations.

    Extends base storage with domain-specific query methods for
    node failure analysis and monitoring.
    """

    def get_recent_failures(self, node_name: str, minutes: int = 30) -> List[StoredNodeFailure]:
        """Get recent unresolved failures for a specific node."""
        raise NotImplementedError

    def get_failure_summary(self, alert_window_minutes: int = 30) -> Dict[str, Dict[str, Any]]:
        """Get summary of unresolved failures across all nodes."""
        raise NotImplementedError

    def get_problematic_nodes(self, threshold: int = 5, minutes: int = 30) -> List[str]:
        """Get node names exceeding failure threshold within time window."""
        raise NotImplementedError

    def resolve_node_failures(self, node_name: str) -> int:
        """Mark all unresolved failures for a node as resolved. Returns count."""
        raise NotImplementedError

    def get_unresolved_failures_for_node_since(
        self, node_name: str, since: datetime.datetime
    ) -> List[StoredNodeFailure]:
        """Get unresolved failures for a node since a given time."""
        raise NotImplementedError


class BaseNodeFailureStorage(BaseItemStorage[StoredNodeFailure], INodeFailureStorage):
    """
    Base storage implementation for node failure events.

    Provides common functionality for storing and querying node failure data
    across different storage backends (SQL, SQLite, Lakehouse, etc.).
    """

    def __init__(self, **kwargs) -> None:
        kwargs["item_class"] = StoredNodeFailure
        if kwargs.get("table_name") is None:
            kwargs["table_name"] = GB_NODE_FAILURES_TABLE_NAME
        super().__init__(**kwargs)

    def _get_column_values(self, item: StoredNodeFailure) -> dict:
        """
        Extract column values for storage from a StoredNodeFailure item.

        Exposes key fields for querying:
        - node_name: For filtering by specific nodes
        - build_id: For finding failures in a specific build
        - launch_id: For finding failures in a specific step
        - failure_type: For analyzing types of failures
        - created_time: For time-based queries
        - resolved: For filtering resolved vs unresolved failures
        """
        fields_to_include = {
            "node_name",
            "build_id",
            "launch_id",
            "failure_type",
            "resolved",
        }

        json = item.model_dump(include=fields_to_include)

        json[CREATED_TIME_FIELD_NAME] = item.created_time

        # Include resolved_timestamp if present
        if item.resolved_timestamp:
            json["resolved_timestamp"] = item.resolved_timestamp

        return json

    @classmethod
    def _get_sample_item(cls) -> StoredNodeFailure:
        """
        Return a sample item for use by BaseItemStorage.

        This is used to initialize the storage schema.
        """
        item = StoredNodeFailure(
            node_name="worker-node-1",
            build_id="build-12345",
            launch_id="launch-67890",
            failure_type="FailedMount",
            retry_count=0,
        )
        return item

    # ── Query methods ────────────────────────────────────────────────

    def get_recent_failures(self, node_name: str, minutes: int = 30) -> List[StoredNodeFailure]:
        """Get recent unresolved failures for a specific node.

        Args:
            node_name: Name of the Kubernetes node
            minutes: Time window in minutes

        Returns:
            List of unresolved failures within the time window
        """
        cutoff = get_utc_time() - datetime.timedelta(minutes=minutes)
        result: List[StoredNodeFailure] = []
        for page in self.get_paged(
            {"node_name": node_name, "resolved": False}, page_size=_PAGE_SIZE
        ):
            result.extend(item for item in page if item.created_time > cutoff)
        return result

    def get_failure_summary(self, alert_window_minutes: int = 30) -> Dict[str, Dict[str, Any]]:
        """Get summary of unresolved failures across all nodes.

        Args:
            alert_window_minutes: Time window for counting recent failures

        Returns:
            Dict mapping node_name to summary with total_failures,
            recent_failures, failure_types, unique_builds, namespaces,
            clusters, oldest_failure, newest_failure
        """
        alert_cutoff = get_utc_time() - datetime.timedelta(minutes=alert_window_minutes)

        # Group by node, streaming page by page
        by_node: Dict[str, List[StoredNodeFailure]] = {}
        for page in self.get_paged({"resolved": False}, page_size=_PAGE_SIZE):
            for item in page:
                by_node.setdefault(item.node_name, []).append(item)

        summary: Dict[str, Dict[str, Any]] = {}
        for node_name, items in by_node.items():
            recent = [i for i in items if i.created_time > alert_cutoff]

            # Count failure types
            failure_types: Dict[str, int] = {}
            for item in items:
                failure_types[item.failure_type] = failure_types.get(item.failure_type, 0) + 1

            unique_builds = len(set(item.build_id for item in items))

            # Extract namespace/cluster from metadata
            namespaces = sorted(
                set(
                    item.metadata.get("namespace", "")
                    for item in items
                    if item.metadata.get("namespace")
                )
            )
            clusters = sorted(
                set(
                    item.metadata.get("cluster", "")
                    for item in items
                    if item.metadata.get("cluster")
                )
            )

            timestamps = [item.created_time for item in items]

            summary[node_name] = {
                "total_failures": len(items),
                "recent_failures": len(recent),
                "failure_types": failure_types,
                "unique_builds": unique_builds,
                "namespaces": namespaces,
                "clusters": clusters,
                "oldest_failure": (min(timestamps).isoformat() if timestamps else None),
                "newest_failure": (max(timestamps).isoformat() if timestamps else None),
            }

        return summary

    def get_problematic_nodes(self, threshold: int = 5, minutes: int = 30) -> List[str]:
        """Get node names exceeding failure threshold within time window.

        Args:
            threshold: Minimum failure count to be considered problematic
            minutes: Time window in minutes

        Returns:
            List of node names with failures >= threshold
        """
        cutoff = get_utc_time() - datetime.timedelta(minutes=minutes)

        node_counts: Dict[str, int] = {}
        for page in self.get_paged({"resolved": False}, page_size=_PAGE_SIZE):
            for item in page:
                if item.created_time > cutoff:
                    node_counts[item.node_name] = node_counts.get(item.node_name, 0) + 1

        return [node for node, count in node_counts.items() if count >= threshold]

    def resolve_node_failures(self, node_name: str) -> int:
        """Mark all unresolved failures for a node as resolved.

        Sets resolved=True and resolved_timestamp=now() on all unresolved
        failures for the given node. Does NOT delete records.

        Args:
            node_name: Name of the Kubernetes node

        Returns:
            Number of failures marked as resolved
        """
        now = get_utc_time()
        count = 0
        for page in self.get_paged(
            {"node_name": node_name, "resolved": False}, page_size=_PAGE_SIZE
        ):
            for item in page:
                self.update_fields(
                    item.uuid,
                    {"resolved": True, "resolved_timestamp": now},
                )
                count += 1
        return count

    def get_unresolved_failures_for_node_since(
        self, node_name: str, since: datetime.datetime
    ) -> List[StoredNodeFailure]:
        """Get unresolved failures for a node since a given time.

        Used by NodeHealthTracker for alert threshold evaluation.

        Args:
            node_name: Name of the Kubernetes node
            since: Cutoff datetime (UTC)

        Returns:
            List of unresolved failures since the cutoff time
        """
        result: List[StoredNodeFailure] = []
        for page in self.get_paged(
            {"node_name": node_name, "resolved": False}, page_size=_PAGE_SIZE
        ):
            result.extend(item for item in page if item.created_time > since)
        return result
