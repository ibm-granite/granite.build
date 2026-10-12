"""SQL storage tests against the hosted PostgreSQL (GBSERVER_SQL_* from the environment).

The non-legacy test bodies live in
test/unit/storage/postgres/test_postgres_build_storage.py,
which runs them against the local test PostgreSQL. The legacy tests read existing rows in the hosted
database's real tables, so they exist only here.
"""

import os

import pytest
from libgbtest.local_postgres import HOSTED_POSTGRES_TEST_MARKS
from libgbtest.storage.build_storage import BaseLegacyStoredBuildTest
from unit.storage.postgres.test_postgres_build_storage import (
    SQLBuildStorageTests,
)

from gbserver.storage.sql.storage_factory import SQLStorageFactory

pytestmark = HOSTED_POSTGRES_TEST_MARKS


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
