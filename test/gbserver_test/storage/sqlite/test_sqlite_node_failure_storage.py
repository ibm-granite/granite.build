import os

import gbserver_test.storage.sql.test_sql_node_failure_storage as HIDE_FROM_PYTEST
import pytest

from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSqliteNodeFailureStorage(HIDE_FROM_PYTEST.TestSQLNodeFailureStorage):

    @classmethod
    def _get_storage_factory(cls):
        return SqliteStorageFactory()
