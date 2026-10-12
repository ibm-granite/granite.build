"""SQL storage tests against the hosted PostgreSQL (GBSERVER_SQL_* from the environment).

The non-legacy test bodies live in
test/unit/storage/postgres/test_postgres_kv_pair_storage.py,
which runs them against the local test PostgreSQL. The legacy tests read existing rows in the hosted
database's real tables, so they exist only here.
"""

import os

import pytest
from libgbtest.local_postgres import HOSTED_POSTGRES_TEST_MARKS
from unit.storage.postgres.test_postgres_kv_pair_storage import (
    SQLKeyValuePairStorageTests,
    SQLKeyValuePairValueMethodsTests,
)

pytestmark = HOSTED_POSTGRES_TEST_MARKS


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSQLKeyValuePairStorage(SQLKeyValuePairStorageTests):
    """SQLKeyValuePairStorage tests against the hosted PostgreSQL."""


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSQLKeyValuePairValueMethods(SQLKeyValuePairValueMethodsTests):
    """SQLKeyValuePairValueMethods tests against the hosted PostgreSQL."""
