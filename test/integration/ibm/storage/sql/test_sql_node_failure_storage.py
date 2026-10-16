"""SQL storage tests against the hosted PostgreSQL (GBSERVER_SQL_* from the environment).

The non-legacy test bodies live in
test/unit/storage/postgres/test_postgres_node_failure_storage.py,
which runs them against the local test PostgreSQL. The legacy tests read existing rows in the hosted
database's real tables, so they exist only here.
"""

from libgbtest.local_postgres import HOSTED_POSTGRES_TEST_MARKS
from libgbtest.storage.node_failure_storage import BaseLegacyNodeFailureTest
from libgbtest.storage.postgres_storage import PostgresStorageTestMixin
from unit.storage.postgres.test_postgres_node_failure_storage import (
    SQLNodeFailureStorageTests,
)

pytestmark = HOSTED_POSTGRES_TEST_MARKS


class TestSQLNodeFailureStorage(SQLNodeFailureStorageTests):
    """SQLNodeFailureStorage tests against the hosted PostgreSQL."""


class TestSQLLegacyNodeFailure(PostgresStorageTestMixin, BaseLegacyNodeFailureTest):
    """Reads the existing rows in the hosted database's real tables (no prefix)."""
