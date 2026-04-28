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
