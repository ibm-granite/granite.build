"""SQL storage tests against the hosted PostgreSQL (GBSERVER_SQL_* from the environment).

The non-legacy test bodies live in
test/unit/storage/postgres/test_postgres_node_failure_storage.py,
which runs them against the local test PostgreSQL. The legacy tests read existing rows in the hosted
database's real tables, so they exist only here.
"""

import os

import pytest
from libgbtest.storage.node_failure_storage import BaseLegacyNodeFailureTest
from unit.storage.postgres.test_postgres_node_failure_storage import (
    SQLNodeFailureStorageTests,
)

from gbserver.storage.sql.storage_factory import SQLStorageFactory

pytestmark = pytest.mark.ibm


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSQLNodeFailureStorage(SQLNodeFailureStorageTests):
    """SQLNodeFailureStorage tests against the hosted PostgreSQL."""


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSQLLegacyNodeFailure(BaseLegacyNodeFailureTest):

    @classmethod
    def _get_storage_factory(cls):
        return SQLStorageFactory()
