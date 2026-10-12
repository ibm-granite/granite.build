import pytest
from libgbtest.constants import extended_testing_only
from libgbtest.local_postgres import requires_local_postgres
from libgbtest.storage.kv_pair_storage import (
    BaseKeyValuePairStorageTest,
)
from libgbtest.storage.kv_pair_storage import (
    TestKeyValuePairValueMethods as _TestKeyValuePairValueMethods,
)

from gbserver.storage.sql.storage_factory import SQLStorageFactory

# The test bodies are mixins (SQL*Tests) shared with the IBM copy in
# test/integration/ibm/storage/sql/test_sql_kv_pair_storage.py,
# which runs them against the hosted PostgreSQL; here they run against the
# local test PostgreSQL.
# thirdparty: needs the psycopg driver from the `thirdparty` extra.
pytestmark = pytest.mark.thirdparty


class SQLKeyValuePairStorageTests(BaseKeyValuePairStorageTest):
    """SQLKeyValuePairStorage tests, shared with the IBM copy (see TestSQLKeyValuePairStorage below)."""

    @classmethod
    def _get_storage_factory(cls):
        return SQLStorageFactory()


class SQLKeyValuePairValueMethodsTests(_TestKeyValuePairValueMethods):
    """SQLKeyValuePairValueMethods tests, shared with the IBM copy (see TestSQLKeyValuePairValueMethods below)."""

    @classmethod
    def _get_storage_factory(cls):
        return SQLStorageFactory()


# live("storage"): keep the real SQL factory even in mock mode, where conftest
# would otherwise swap in SQLite.
@extended_testing_only
@requires_local_postgres
@pytest.mark.live("storage")
class TestSQLKeyValuePairStorage(SQLKeyValuePairStorageTests):
    """SQLKeyValuePairStorage tests against the local test PostgreSQL."""


# live("storage"): keep the real SQL factory even in mock mode, where conftest
# would otherwise swap in SQLite.
@extended_testing_only
@requires_local_postgres
@pytest.mark.live("storage")
class TestSQLKeyValuePairValueMethods(SQLKeyValuePairValueMethodsTests):
    """SQLKeyValuePairValueMethods tests against the local test PostgreSQL."""
