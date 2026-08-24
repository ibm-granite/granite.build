import os

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
