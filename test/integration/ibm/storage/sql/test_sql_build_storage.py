"""SQL storage tests against the hosted PostgreSQL (GBSERVER_SQL_* from the environment).

The non-legacy test bodies live in
test/unit/storage/postgres/test_postgres_build_storage.py,
which runs them against the local test PostgreSQL. The legacy tests read existing rows in the hosted
database's real tables, so they exist only here.
"""

import os

import pytest
from libgbtest.local_postgres import requires_hosted_postgres
from libgbtest.storage.build_storage import BaseLegacyStoredBuildTest
from unit.storage.postgres.test_postgres_build_storage import (
    SQLBuildStorageTests,
)

from gbserver.storage.sql.storage_factory import SQLStorageFactory

# requires_hosted_postgres: these target the hosted database, so skip in
# GB_ENVIRONMENT=STANDALONE runs (no such schema there, or GBSERVER_SQL_* point at
# the local test PostgreSQL).
pytestmark = [pytest.mark.ibm, requires_hosted_postgres]


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSQLBuildStorage(SQLBuildStorageTests):
    """SQLBuildStorage tests against the hosted PostgreSQL."""


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSQLLegacyStoredBuild(BaseLegacyStoredBuildTest):

    @classmethod
    def _get_storage_factory(cls):
        # This makes it a SQL test.
        return SQLStorageFactory()
