"""SQL storage tests against the hosted PostgreSQL (GBSERVER_SQL_* from the environment).

The non-legacy test bodies live in
test/unit/storage/postgres/test_postgres_space_storage.py,
which runs them against the local test PostgreSQL. The legacy tests read existing rows in the hosted
database's real tables, so they exist only here.
"""

from libgbtest.local_postgres import HOSTED_POSTGRES_TEST_MARKS
from libgbtest.storage.postgres_storage import PostgresStorageTestMixin
from libgbtest.storage.space_storage import BaseLegacyStoredSpaceTest
from unit.storage.postgres.test_postgres_space_storage import (
    SQLSpaceStorageTests,
)

pytestmark = HOSTED_POSTGRES_TEST_MARKS


class TestSQLSpaceStorage(SQLSpaceStorageTests):
    """SQLSpaceStorage tests against the hosted PostgreSQL."""


class TestSQLLegacyStoredSpace(PostgresStorageTestMixin, BaseLegacyStoredSpaceTest):
    """Reads the existing rows in the hosted database's real tables (no prefix)."""
