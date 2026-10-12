from libgbtest.local_postgres import LOCAL_POSTGRES_TEST_MARKS
from libgbtest.storage.artifact_storage import (
    BaseArtifactStorageTest,
)

from gbserver.storage.sql.storage_factory import SQLStorageFactory

# The test bodies are mixins (SQL*Tests) shared with the IBM copy in
# test/integration/ibm/storage/sql/test_sql_artifact_storage.py,
# which runs them against the hosted PostgreSQL; here they run against the
# local test PostgreSQL.
pytestmark = LOCAL_POSTGRES_TEST_MARKS


class SQLArtifactStorageTests(BaseArtifactStorageTest):
    """SQLArtifactStorage tests, shared with the IBM copy (see TestSQLArtifactStorage below)."""

    @classmethod
    def _get_storage_factory(cls):
        # This makes it a SQL test.
        return SQLStorageFactory()


class TestSQLArtifactStorage(SQLArtifactStorageTests):
    """SQLArtifactStorage tests against the local test PostgreSQL."""
