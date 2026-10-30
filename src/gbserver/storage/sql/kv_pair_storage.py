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

from typing import Any, Dict, Optional

from sqlalchemy.exc import IntegrityError

from gbserver.storage.kv_pair_storage import (
    BaseKeyValuePairStorage,
    IKeyValuePairStorage,
)
from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.storage.storage import JSON_COLUMN_NAME, UUID_COLUMN_NAME
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

    def compare_and_set(
        self, key: str, expected: Optional[Dict[str, Any]], value: Dict[str, Any]
    ) -> bool:
        """Optimistic: the update matches the row only while its stored JSON is the
        one read, so a write by another process in between makes it match nothing.
        An unset key is claimed by insert, which the unique ``key`` index lets only
        one writer win."""
        existing = self.get_by_key(key)
        if existing is None:
            if expected is not None:
                return False
            try:
                self.add(StoredKeyValuePair(key=key, value=value))
            except (IntegrityError, ValueError):
                return False
            return True
        if existing.value != expected:
            return False
        model = self._sql_alchemy_model
        session = self._get_session_without_retry()
        try:
            uuid_col = getattr(model, UUID_COLUMN_NAME)
            json_col = getattr(model, JSON_COLUMN_NAME)
            stored = session.query(json_col).filter(uuid_col == existing.uuid).scalar()
            if stored is None:
                return False
            if self._convert_json_column(stored) != expected:
                return False
            row = self._convert_item_to_row_dict(
                StoredKeyValuePair(
                    uuid=existing.uuid,
                    key=key,
                    value=value,
                    created_time=existing.created_time,
                )
            )
            matched = (
                session.query(model)
                .filter(uuid_col == existing.uuid, json_col == stored)
                .update(row, synchronize_session=False)
            )
            session.commit()
            return matched == 1
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _convert_json_column(stored: str) -> Optional[Dict[str, Any]]:
        return StoredKeyValuePair.model_validate_json(stored).value
