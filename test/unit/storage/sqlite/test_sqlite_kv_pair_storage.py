import os
import uuid as uuid_module

import integration.ibm.storage.sql.test_sql_kv_pair_storage as HIDE_FROM_PYTEST
import pytest

from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSqliteKeyValuePairStorage(HIDE_FROM_PYTEST.TestSQLKeyValuePairStorage):

    @classmethod
    def _get_storage_factory(cls):
        return SqliteStorageFactory()


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSqliteKeyValuePairValueMethods(
    HIDE_FROM_PYTEST.TestSQLKeyValuePairValueMethods
):

    @classmethod
    def _get_storage_factory(cls):
        return SqliteStorageFactory()


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
def test_compare_and_set_holds_the_file_lock_and_retries_a_busy_file(monkeypatch):
    """The read and the conditional update are two statements: the file lock
    keeps another process from writing between them, and a locked database is
    retried instead of surfacing to the indexer."""
    from unittest.mock import patch

    from sqlalchemy.exc import OperationalError

    from gbserver.storage.sql.kv_pair_storage import SQLKeyValuePairStorage

    storage = SqliteStorageFactory().create_kv_pair_storage(
        table_name=f"t_kv_{uuid_module.uuid4().hex[:8]}"
    )
    storage.set_value("k", {"v": 1})
    held = []
    attempts = {"n": 0}
    original = SQLKeyValuePairStorage._compare_and_set_without_retry

    def flaky(self, key, expected, value):
        held.append(self._db_file_lock.is_locked)
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OperationalError("UPDATE", {}, Exception("database is locked"))
        return original(self, key, expected, value)

    with (
        patch.object(SQLKeyValuePairStorage, "_compare_and_set_without_retry", flaky),
        patch("tenacity.nap.time.sleep"),
    ):
        assert storage.compare_and_set("k", {"v": 1}, {"v": 2}) is True
    assert held == [True, True]
    assert storage.get_value("k") == {"v": 2}


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
def test_compare_and_set_nests_no_retry_under_the_file_lock():
    """Each attempt runs plain statements only: a retried read inside it would
    sleep its backoff while every other SQLite call in the process waits."""
    from unittest.mock import patch

    storage = SqliteStorageFactory().create_kv_pair_storage(
        table_name=f"t_kv_{uuid_module.uuid4().hex[:8]}"
    )
    storage.set_value("k", {"v": 1})
    with (
        patch.object(
            type(storage), "get_by_key", side_effect=AssertionError("retried read")
        ),
        patch.object(type(storage), "add", side_effect=AssertionError("retried add")),
    ):
        assert storage.compare_and_set("k", {"v": 1}, {"v": 2}) is True
        assert storage.compare_and_set("k", {"v": 1}, {"v": 3}) is False
        assert storage.compare_and_set("new", None, {"v": 1}) is True
        assert storage.compare_and_set("new", None, {"v": 2}) is False
    assert storage.get_value("k") == {"v": 2}
    assert storage.get_value("new") == {"v": 1}


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
def test_compare_and_set_keeps_a_legacy_rows_missing_created_time():
    """A row written before created_time existed reads it as BEGINNING_OF_TIME
    (as get_by_key does); a CAS update must keep that, not stamp it now."""
    import json

    from sqlalchemy import text

    from gbserver.storage.storage import BEGINNING_OF_TIME, JSON_COLUMN_NAME

    storage = SqliteStorageFactory().create_kv_pair_storage(
        table_name=f"t_kv_{uuid_module.uuid4().hex[:8]}"
    )
    storage.set_value("k", {"v": 1})
    blob = json.loads(storage.get_by_key("k").model_dump_json())
    del blob["created_time"]
    with storage._engine.begin() as connection:
        connection.execute(
            text(f"UPDATE {storage.table_name} SET {JSON_COLUMN_NAME} = :j"),
            {"j": json.dumps(blob)},
        )
    assert storage.get_by_key("k").created_time == BEGINNING_OF_TIME
    assert storage.compare_and_set("k", {"v": 1}, {"v": 2}) is True
    assert storage.get_by_key("k").created_time == BEGINNING_OF_TIME


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
def test_compare_and_set_recreates_a_table_dropped_under_it():
    """The table-exists flag is per instance: a table dropped under a live
    process is recreated, not failed on every checkpoint write until restart."""
    from unittest.mock import patch

    from sqlalchemy import text

    storage = SqliteStorageFactory().create_kv_pair_storage(
        table_name=f"t_kv_{uuid_module.uuid4().hex[:8]}"
    )
    assert storage.compare_and_set("k", None, {"v": 1}) is True
    storage.delete_table()
    assert storage.compare_and_set("k", None, {"v": 1}) is True
    # Dropped behind the storage's back, not through delete_table.
    with storage._engine.begin() as connection:
        connection.execute(text(f"DROP TABLE {storage.table_name}"))
    with patch("tenacity.nap.time.sleep"):
        assert storage.compare_and_set("k", None, {"v": 2}) is True
    assert storage.get_value("k") == {"v": 2}


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
def test_compare_and_set_prepares_the_table_outside_the_file_lock():
    """Creating the table retries on its own: it must not run under the lock an
    attempt holds, nor inside the attempts' retry."""
    from unittest.mock import patch

    from gbserver.storage.sql.kv_pair_storage import SQLKeyValuePairStorage

    storage = SqliteStorageFactory().create_kv_pair_storage(
        table_name=f"t_kv_{uuid_module.uuid4().hex[:8]}"
    )
    held = []
    original = SQLKeyValuePairStorage._prepare_table

    def prepare(self):
        held.append(self._db_file_lock.is_locked)
        return original(self)

    with patch.object(SQLKeyValuePairStorage, "_prepare_table", prepare):
        assert storage.compare_and_set("k", None, {"v": 1}) is True
        assert storage.compare_and_set("k", {"v": 1}, {"v": 2}) is True
    assert held == [False]
