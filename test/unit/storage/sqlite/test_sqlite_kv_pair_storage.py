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
    original = SQLKeyValuePairStorage.compare_and_set

    def flaky(self, key, expected, value):
        held.append(self._db_file_lock.is_locked)
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OperationalError("UPDATE", {}, Exception("database is locked"))
        return original(self, key, expected, value)

    with (
        patch.object(SQLKeyValuePairStorage, "compare_and_set", flaky),
        patch("tenacity.nap.time.sleep"),
    ):
        assert storage.compare_and_set("k", {"v": 1}, {"v": 2}) is True
    assert held == [True, True]
    assert storage.get_value("k") == {"v": 2}
