"""SQL storage tests against the hosted PostgreSQL (GBSERVER_SQL_* from the environment).

The non-legacy test bodies live in
test/unit/storage/postgres/test_postgres_build_storage.py,
which runs them against the local test PostgreSQL. The legacy tests read existing rows in the hosted
database's real tables, so they exist only here.
"""

from libgbtest.local_postgres import HOSTED_POSTGRES_TEST_MARKS
from libgbtest.storage.build_storage import BaseLegacyStoredBuildTest
from libgbtest.storage.postgres_storage import PostgresStorageTestMixin
from unit.storage.postgres.test_postgres_build_storage import (
    SQLBuildStorageTests,
)

pytestmark = HOSTED_POSTGRES_TEST_MARKS


class TestSQLBuildStorage(SQLBuildStorageTests):
    """SQLBuildStorage tests against the hosted PostgreSQL."""


class TestSQLLegacyStoredBuild(PostgresStorageTestMixin, BaseLegacyStoredBuildTest):
    """Reads the existing rows in the hosted database's real tables (no prefix)."""
