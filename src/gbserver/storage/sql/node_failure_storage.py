#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""
SQL storage implementation for node failure events.
"""

from gbserver.storage.node_failure_storage import (
    BaseNodeFailureStorage,
    INodeFailureStorage,
)
from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.storage.stored_node_failure import StoredNodeFailure


class SQLNodeFailureStorage(
    BaseSQLItemStorage[StoredNodeFailure],
    BaseNodeFailureStorage,
    INodeFailureStorage,
):
    """
    SQL-based storage implementation for node failure events.

    Schema optimizations:
    - Indexed node_name and resolved columns for filtered queries
    - Autoincrement index column for deterministic ordering (gb_events pattern)
    - Default sorting by created_time descending for recent-first queries
    """

    def __init__(self, **kwargs) -> None:
        # Index node_name and resolved for query methods that filter by both
        kwargs["indexed_columns"] = ["node_name", "resolved"]
        # Autoincrement index for deterministic ordering (follows gb_events pattern)
        kwargs["autoincr_column"] = "index"
        # Sort by index ascending by default (equivalent to created_time order, but
        # faster since index is an integer primary key)
        kwargs["default_pagination_sort_by_column"] = "index"
        super().__init__(**kwargs)
