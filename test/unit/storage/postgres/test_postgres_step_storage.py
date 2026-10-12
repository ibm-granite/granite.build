from libgbtest.local_postgres import LOCAL_POSTGRES_TEST_MARKS
from libgbtest.storage.postgres_storage import PostgresStorageTestMixin
from libgbtest.storage.step_storage import (
    BaseStepStorageTest,
)

# The test bodies are mixins (SQL*Tests) shared with the IBM copy in
# test/integration/ibm/storage/sql/test_sql_step_storage.py,
# which runs them against the hosted PostgreSQL; here they run against the
# local test PostgreSQL.
pytestmark = LOCAL_POSTGRES_TEST_MARKS


class SQLStepStorageTests(PostgresStorageTestMixin, BaseStepStorageTest):
    """SQLStepStorage tests, shared with the IBM copy (see TestSQLStepStorage below)."""


class TestSQLStepStorage(SQLStepStorageTests):
    """SQLStepStorage tests against the local test PostgreSQL."""
