"""SQL storage tests against the hosted PostgreSQL (GBSERVER_SQL_* from the environment).

The non-legacy test bodies live in
test/unit/storage/postgres/test_postgres_space_user_storage.py,
which runs them against the local test PostgreSQL. The legacy tests read existing rows in the hosted
database's real tables, so they exist only here.
"""

from libgbtest.local_postgres import HOSTED_POSTGRES_TEST_MARKS
from unit.storage.postgres.test_postgres_space_user_storage import (
    SQLSpaceUserStorageTests,
)

pytestmark = HOSTED_POSTGRES_TEST_MARKS


class TestSQLSpaceUserStorage(SQLSpaceUserStorageTests):
    """SQLSpaceUserStorage tests against the hosted PostgreSQL."""
