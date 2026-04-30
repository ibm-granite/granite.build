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

from gbserver.storage.event_storage import BaseStoredEventStorage, IStoredEventStorage
from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.storage.storage import CREATED_TIME_FIELD_NAME
from gbserver.storage.stored_event import StoredEvent


class SQLEventStorage(BaseSQLItemStorage[StoredEvent], BaseStoredEventStorage, IStoredEventStorage):

    def __init__(self, **kwargs) -> None:
        kwargs["indexed_columns"] = ["build_id"]
        kwargs["default_pagination_sort_by_column"] = CREATED_TIME_FIELD_NAME
        # We add an index for sorting since the timestamps may not be reliable.
        # See get_sorted_build_events() below.
        kwargs["autoincr_column"] = "index"
        super().__init__(**kwargs)
