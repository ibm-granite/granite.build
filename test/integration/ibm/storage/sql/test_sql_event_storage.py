"""SQL storage tests against the hosted PostgreSQL (GBSERVER_SQL_* from the environment).

The non-legacy test bodies live in
test/unit/storage/postgres/test_postgres_event_storage.py,
which runs them against the local test PostgreSQL. The legacy tests read existing rows in the hosted
database's real tables, so they exist only here.
"""

from libgbtest.local_postgres import HOSTED_POSTGRES_TEST_MARKS
from libgbtest.storage.event_storage import BaseLegacyEventStorageTest
from libgbtest.storage.postgres_storage import PostgresStorageTestMixin
from unit.storage.postgres.test_postgres_event_storage import (
    SQLEventStorageTests,
)

pytestmark = HOSTED_POSTGRES_TEST_MARKS


class TestSQLEventStorage(SQLEventStorageTests):
    """SQLEventStorage tests against the hosted PostgreSQL."""


class TestSQLLegacyStoredEvent(PostgresStorageTestMixin, BaseLegacyEventStorageTest):
    """Reads the existing rows in the hosted database's real tables (no prefix)."""
