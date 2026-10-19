from libgbtest.local_postgres import LOCAL_POSTGRES_TEST_MARKS
from libgbtest.storage.postgres_storage import PostgresStorageTestMixin
from libgbtest.storage.target_storage import (
    BaseTargetStorageTest,
)

# The test bodies are mixins (SQL*Tests) shared with the IBM copy in
# test/integration/ibm/storage/sql/test_sql_target_storage.py,
# which runs them against the hosted PostgreSQL; here they run against the
# local test PostgreSQL.
pytestmark = LOCAL_POSTGRES_TEST_MARKS


class SQLTargetStorageTests(PostgresStorageTestMixin, BaseTargetStorageTest):
    """SQLTargetStorage tests, shared with the IBM copy (see TestSQLTargetStorage below)."""


class TestSQLTargetStorage(SQLTargetStorageTests):
    """SQLTargetStorage tests against the local test PostgreSQL."""
