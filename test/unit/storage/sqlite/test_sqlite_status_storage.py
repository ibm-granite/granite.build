import os

import integration.ibm.storage.sql.test_sql_status_storage as HIDE_FROM_PYTEST
import pytest

from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSqliteStatusStorage(HIDE_FROM_PYTEST.TestSQLStatusStorage):

    @classmethod
    def _get_storage_factory(cls):
        return SqliteStorageFactory()


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSqliteStatusValueMethods(HIDE_FROM_PYTEST.TestSQLStatusValueMethods):

    @classmethod
    def _get_storage_factory(cls):
        return SqliteStorageFactory()
