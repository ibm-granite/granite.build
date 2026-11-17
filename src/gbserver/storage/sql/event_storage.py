#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

from gbserver.storage.event_storage import BaseStoredEventStorage, IStoredEventStorage
from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.storage.storage import CREATED_TIME_FIELD_NAME
from gbserver.storage.stored_event import StoredEvent


class SQLEventStorage(
    BaseSQLItemStorage[StoredEvent], BaseStoredEventStorage, IStoredEventStorage
):

    def __init__(self, **kwargs) -> None:
        kwargs["indexed_columns"] = ["build_id"]
        kwargs["default_pagination_sort_by_column"] = CREATED_TIME_FIELD_NAME
        # We add an index for sorting since the timestamps may not be reliable.
        # See get_sorted_build_events() below.
        kwargs["autoincr_column"] = "index"
        super().__init__(**kwargs)
