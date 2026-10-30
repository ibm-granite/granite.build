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

from typing import Any, Callable, Dict, Optional

from sqlalchemy.exc import IntegrityError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from gbserver.storage.kv_pair_storage import (
    BaseKeyValuePairStorage,
    IKeyValuePairStorage,
)
from gbserver.storage.sql.sql_storage import (
    BaseSQLItemStorage,
    _is_retryable_read_error,
)
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
        # Whether compare_and_set has seen the table exist, so the checkpoint
        # write does not inspect the schema on every call. Reset with the table
        # (delete_table) and on a missing-table error, so it is recreated.
        self._table_ready = False

    def compare_and_set(
        self, key: str, expected: Optional[Dict[str, Any]], value: Dict[str, Any]
    ) -> bool:
        """Optimistic: the update matches the row only while its stored JSON is the
        one read, so a write by another process in between makes it match nothing.
        An unset key is claimed by insert, which the unique ``key`` index lets only
        one writer win."""
        tried = False

        def attempt() -> bool:
            nonlocal tried
            if not self._table_ready:
                # What the add() this replaced did on first use and after the
                # table was dropped: create it, or adjust it to the item.
                sample = self._convert_item_to_row_dict(self._get_sample_item())
                self._create_or_adjust_schema_item_dict(sample)
                if not self._ensure_table_without_retry():
                    return False
                self._table_ready = True
            # A retried attempt may follow one whose write committed before it
            # raised; finding ``value`` already stored then counts as this call's
            # success.
            if tried and expected != value:
                if self._stored_value_without_retry(key) == value:
                    return True
            tried = True
            try:
                return self._compare_and_set_without_retry(key, expected, value)
            except Exception as exc:
                if _is_missing_table(exc):
                    # Dropped under a live process: the next attempt recreates it.
                    self._table_ready = False
                raise

        return self._run_attempts(attempt)

    def _delete_table(self):
        self._table_ready = False
        super()._delete_table()

    def _run_attempts(self, attempt: Callable[[], bool]) -> bool:
        """Retry ``attempt`` like a read (``retry_read``): a dropped connection or a
        busy database is retried, as the ``get_by_key``/``add`` it replaced were.
        Each attempt runs only plain statements, so no retry nests inside another.
        """
        return _CAS_RETRY(attempt)()

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
            if stored is None:
                return None
            return self._convert_row_dict_to_item({JSON_COLUMN_NAME: stored}).value
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
            # Through the storage's own deserialization, so a legacy row with no
            # created_time reads it as get_by_key does (BEGINNING_OF_TIME) and the
            # update keeps that rather than stamping it now.
            current = self._convert_row_dict_to_item({JSON_COLUMN_NAME: stored})
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


def _is_missing_table(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "no such table" in message or (
        "relation" in message and "does not exist" in message
    )


def _is_retryable_cas_error(exc: BaseException) -> bool:
    """Like a read, plus a missing table: the next attempt recreates it."""
    return _is_missing_table(exc) or _is_retryable_read_error(exc)


# The compare-and-set's retry: a read's policy, and a table dropped under a live
# process is recreated on the next attempt rather than failing every scan.
_CAS_RETRY = retry(
    wait=wait_random_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(10),
    retry=retry_if_exception(_is_retryable_cas_error),
    reraise=True,
)
