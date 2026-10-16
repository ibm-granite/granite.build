from libgbtest.local_postgres import LOCAL_POSTGRES_TEST_MARKS
from libgbtest.storage.postgres_storage import PostgresStorageTestMixin
from libgbtest.storage.space_storage import (
    BaseSpaceStorageTest,
)

# The test bodies are mixins (SQL*Tests) shared with the IBM copy in
# test/integration/ibm/storage/sql/test_sql_space_storage.py,
# which runs them against the hosted PostgreSQL; here they run against the
# local test PostgreSQL.
pytestmark = LOCAL_POSTGRES_TEST_MARKS


class SQLSpaceStorageTests(PostgresStorageTestMixin, BaseSpaceStorageTest):
    """SQLSpaceStorage tests, shared with the IBM copy (see TestSQLSpaceStorage below)."""


class TestSQLSpaceStorage(SQLSpaceStorageTests):
    """SQLSpaceStorage tests against the local test PostgreSQL."""
