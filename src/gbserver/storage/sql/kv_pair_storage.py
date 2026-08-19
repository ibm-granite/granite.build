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
SQL storage implementation for the generic gb_kv_pairs key-value store.
"""

from gbserver.storage.kv_pair_storage import (
    BaseKeyValuePairStorage,
    IKeyValuePairStorage,
)
from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.storage.stored_kv_pair import StoredKeyValuePair


class SQLKeyValuePairStorage(
    BaseSQLItemStorage[StoredKeyValuePair],
    BaseKeyValuePairStorage,
    IKeyValuePairStorage,
):
    """SQL-based storage implementation for the generic gb_kv_pairs key-value store."""

    def __init__(self, **kwargs) -> None:
        # Keys are looked up individually (get_value/set_value resolve `key` to
        # a row), so `key` carries a unique index: it is the store's lookup
        # identity, and the uniqueness is what lets set_value treat a found row
        # as *the* row for that key. Follows the sibling SQL storages' **kwargs
        # constructor pattern, which also keeps the factory's
        # `SQLKeyValuePairStorage(table_name=...)` call type-checkable.
        kwargs["unique_columns"] = {"key": None}
        super().__init__(**kwargs)
