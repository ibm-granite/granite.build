"""Lineage SQL storage tests against the hosted PostgreSQL (GBSERVER_SQL_* from the environment).

The test bodies live in test/unit/storage/postgres/test_postgres_lineage_job_storage.py,
which runs them against the local test PostgreSQL.
"""

from libgbtest.local_postgres import HOSTED_POSTGRES_TEST_MARKS
from unit.storage.postgres.test_postgres_lineage_job_storage import (
    SQLLineageJobStorageTests,
)

pytestmark = HOSTED_POSTGRES_TEST_MARKS


class TestSQLLineageJobStorage(SQLLineageJobStorageTests):
    """``gb_lineage_job`` tests against the hosted PostgreSQL."""
