from libgbtest.local_postgres import LOCAL_POSTGRES_TEST_MARKS
from libgbtest.storage.artifact_storage import (
    BaseArtifactStorageTest,
)
from libgbtest.storage.postgres_storage import PostgresStorageTestMixin

# The test bodies are mixins (SQL*Tests) shared with the IBM copy in
# test/integration/ibm/storage/sql/test_sql_artifact_storage.py,
# which runs them against the hosted PostgreSQL; here they run against the
# local test PostgreSQL.
pytestmark = LOCAL_POSTGRES_TEST_MARKS


class SQLArtifactStorageTests(PostgresStorageTestMixin, BaseArtifactStorageTest):
    """SQLArtifactStorage tests, shared with the IBM copy (see TestSQLArtifactStorage below)."""


class TestSQLArtifactStorage(SQLArtifactStorageTests):
    """SQLArtifactStorage tests against the local test PostgreSQL."""
