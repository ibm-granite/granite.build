from libgbtest.local_postgres import LOCAL_POSTGRES_TEST_MARKS
from libgbtest.storage.node_failure_storage import (
    BaseNodeFailureStorageTest,
)

from gbserver.storage.sql.storage_factory import SQLStorageFactory

# The test bodies are mixins (SQL*Tests) shared with the IBM copy in
# test/integration/ibm/storage/sql/test_sql_node_failure_storage.py,
# which runs them against the hosted PostgreSQL; here they run against the
# local test PostgreSQL.
pytestmark = LOCAL_POSTGRES_TEST_MARKS


class SQLNodeFailureStorageTests(BaseNodeFailureStorageTest):
    """SQLNodeFailureStorage tests, shared with the IBM copy (see TestSQLNodeFailureStorage below)."""

    @classmethod
    def _get_storage_factory(cls):
        return SQLStorageFactory()


class TestSQLNodeFailureStorage(SQLNodeFailureStorageTests):
    """SQLNodeFailureStorage tests against the local test PostgreSQL."""
