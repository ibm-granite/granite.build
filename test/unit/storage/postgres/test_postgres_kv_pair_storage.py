from libgbtest.local_postgres import LOCAL_POSTGRES_TEST_MARKS
from libgbtest.storage.kv_pair_storage import (
    BaseKeyValuePairStorageTest,
)
from libgbtest.storage.kv_pair_storage import (
    TestKeyValuePairValueMethods as _TestKeyValuePairValueMethods,
)
from libgbtest.storage.postgres_storage import PostgresStorageTestMixin

# The test bodies are mixins (SQL*Tests) shared with the IBM copy in
# test/integration/ibm/storage/sql/test_sql_kv_pair_storage.py,
# which runs them against the hosted PostgreSQL; here they run against the
# local test PostgreSQL.
pytestmark = LOCAL_POSTGRES_TEST_MARKS


class SQLKeyValuePairStorageTests(
    PostgresStorageTestMixin, BaseKeyValuePairStorageTest
):
    """SQLKeyValuePairStorage tests, shared with the IBM copy (see TestSQLKeyValuePairStorage below)."""


class SQLKeyValuePairValueMethodsTests(
    PostgresStorageTestMixin, _TestKeyValuePairValueMethods
):
    """SQLKeyValuePairValueMethods tests, shared with the IBM copy (see TestSQLKeyValuePairValueMethods below)."""


class TestSQLKeyValuePairStorage(SQLKeyValuePairStorageTests):
    """SQLKeyValuePairStorage tests against the local test PostgreSQL."""


class TestSQLKeyValuePairValueMethods(SQLKeyValuePairValueMethodsTests):
    """SQLKeyValuePairValueMethods tests against the local test PostgreSQL."""
