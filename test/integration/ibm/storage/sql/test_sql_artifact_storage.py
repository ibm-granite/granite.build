"""SQL storage tests against the hosted PostgreSQL (GBSERVER_SQL_* from the environment).

The non-legacy test bodies live in
test/unit/storage/postgres/test_postgres_artifact_storage.py,
which runs them against the local test PostgreSQL. The legacy tests read existing rows in the hosted
database's real tables, so they exist only here.
"""

from libgbtest.local_postgres import HOSTED_POSTGRES_TEST_MARKS
from libgbtest.storage.artifact_storage import BaseLegacyArtifactStorageTest
from libgbtest.storage.postgres_storage import PostgresStorageTestMixin
from unit.storage.postgres.test_postgres_artifact_storage import (
    SQLArtifactStorageTests,
)

pytestmark = HOSTED_POSTGRES_TEST_MARKS


class TestSQLArtifactStorage(SQLArtifactStorageTests):
    """SQLArtifactStorage tests against the hosted PostgreSQL."""


class TestSQLLegacyArtifactStorage(
    PostgresStorageTestMixin, BaseLegacyArtifactStorageTest
):
    """Reads the existing rows in the hosted database's real tables (no prefix)."""
