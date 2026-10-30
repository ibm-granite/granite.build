"""Lineage SQL storage tests against the hosted PostgreSQL (GBSERVER_SQL_* from the environment).

The test bodies live in test/unit/storage/postgres/test_postgres_lineage_index_storage.py,
which runs them against the local test PostgreSQL.
"""

from libgbtest.local_postgres import HOSTED_POSTGRES_TEST_MARKS
from unit.storage.postgres.test_postgres_lineage_index_storage import (
    SQLLineageIndexStorageTests,
)

pytestmark = HOSTED_POSTGRES_TEST_MARKS


class TestSQLLineageIndexStorage(SQLLineageIndexStorageTests):
    """``gb_lineage_index`` tests against the hosted PostgreSQL."""
