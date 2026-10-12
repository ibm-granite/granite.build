from libgbtest.local_postgres import LOCAL_POSTGRES_TEST_MARKS
from libgbtest.storage.event_storage import (
    BaseEventStorageTest,
)
from libgbtest.storage.postgres_storage import PostgresStorageTestMixin

# The test bodies are mixins (SQL*Tests) shared with the IBM copy in
# test/integration/ibm/storage/sql/test_sql_event_storage.py,
# which runs them against the hosted PostgreSQL; here they run against the
# local test PostgreSQL.
pytestmark = LOCAL_POSTGRES_TEST_MARKS


class SQLEventStorageTests(PostgresStorageTestMixin, BaseEventStorageTest):
    """SQLEventStorage tests, shared with the IBM copy (see TestSQLEventStorage below)."""


class TestSQLEventStorage(SQLEventStorageTests):
    """SQLEventStorage tests against the local test PostgreSQL."""
