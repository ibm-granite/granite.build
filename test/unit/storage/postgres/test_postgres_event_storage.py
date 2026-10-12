import pytest
from libgbtest.constants import extended_testing_only
from libgbtest.local_postgres import requires_local_postgres
from libgbtest.storage.event_storage import (
    BaseEventStorageTest,
)

from gbserver.storage.sql.storage_factory import SQLStorageFactory

# The test bodies are mixins (SQL*Tests) shared with the IBM copy in
# test/integration/ibm/storage/sql/test_sql_event_storage.py,
# which runs them against the hosted PostgreSQL; here they run against the
# local test PostgreSQL.
# thirdparty: needs the psycopg driver from the `thirdparty` extra.
pytestmark = pytest.mark.thirdparty


class SQLEventStorageTests(BaseEventStorageTest):
    """SQLEventStorage tests, shared with the IBM copy (see TestSQLEventStorage below)."""

    @classmethod
    def _get_storage_factory(cls):
        # This makes it a SQL test.
        return SQLStorageFactory()


# live("storage"): keep the real SQL factory even in mock mode, where conftest
# would otherwise swap in SQLite.
@extended_testing_only
@requires_local_postgres
@pytest.mark.live("storage")
class TestSQLEventStorage(SQLEventStorageTests):
    """SQLEventStorage tests against the local test PostgreSQL."""
