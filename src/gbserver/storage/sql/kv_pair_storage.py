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
        self._ensure_table()
        return self._compare_and_set_without_retry(key, expected, value)

    def _stored_value_without_retry(self, key: str) -> Optional[Dict[str, Any]]:
        """The value under ``key`` in one plain query: no retry, no lock of its own."""
        model = self._sql_alchemy_model
        session = self._get_session_without_retry()
        try:
            stored = (
                session.query(getattr(model, JSON_COLUMN_NAME))
                .filter(getattr(model, "key") == key)
                .scalar()
            )
            return None if stored is None else self._convert_json_column(stored)
        finally:
            session.close()

    def _compare_and_set_without_retry(
        self, key: str, expected: Optional[Dict[str, Any]], value: Dict[str, Any]
    ) -> bool:
        """:meth:`compare_and_set` as plain statements in one session.

        Nothing in it retries on its own, so a caller that retries the whole (and,
        on SQLite, holds the file lock per attempt) never nests a backoff inside
        it. The table must exist (see ``_ensure_table``).
        """
        model = self._sql_alchemy_model
        uuid_col = getattr(model, UUID_COLUMN_NAME)
        json_col = getattr(model, JSON_COLUMN_NAME)
        key_col = getattr(model, "key")
        session = self._get_session_without_retry()
        try:
            found = session.query(uuid_col, json_col).filter(key_col == key).first()
            if found is None:
                if expected is not None:
                    return False
                item = StoredKeyValuePair(key=key, value=value)
                session.add(model(**self._convert_item_to_row_dict(item)))
                try:
                    session.commit()
                except IntegrityError:
                    session.rollback()
                    return False
                return True
            row_uuid, stored = found
            current = StoredKeyValuePair.model_validate_json(stored)
            if current.value != expected:
                return False
            row = self._convert_item_to_row_dict(
                StoredKeyValuePair(
                    uuid=row_uuid,
                    key=key,
                    value=value,
                    created_time=current.created_time,
                )
            )
            matched = (
                session.query(model)
                .filter(uuid_col == row_uuid, json_col == stored)
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
