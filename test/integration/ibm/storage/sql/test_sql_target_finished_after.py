"""Lineage SQL storage tests against the hosted PostgreSQL (GBSERVER_SQL_* from the environment).

The test bodies live in test/unit/storage/postgres/test_postgres_target_finished_after.py,
which runs them against the local test PostgreSQL.
"""

from libgbtest.local_postgres import HOSTED_POSTGRES_TEST_MARKS
from unit.storage.postgres.test_postgres_target_finished_after import *  # noqa: F401,F403

pytestmark = HOSTED_POSTGRES_TEST_MARKS
