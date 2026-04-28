import os

import gbserver_test.storage.sql.test_sql_build_storage as HIDE_FROM_PYTEST  # Doing it this way keeps pytest from also adding TestSQLBuildStorage tests as part of this file.
import pytest
from gbserver_test.storage.test_build_storage import BuildStorageTestSupport

from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory

pytestmark = pytest.mark.ibm


class SqliteBuildStorageTestSupport(BuildStorageTestSupport):
    """Standalone-safe test support that doesn't require GitHub API."""

    def _get_build_user(self) -> str:
        return "test-user"


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSqliteBuildStorage(HIDE_FROM_PYTEST.TestSQLBuildStorage):

    @classmethod
    def _is_cloud_config_required(cls) -> bool:
        return False

    @classmethod
    def _get_storage_factory(cls):
        return SqliteStorageFactory()

    @classmethod
    def _get_test_config(cls) -> BuildStorageTestSupport:
        return SqliteBuildStorageTestSupport()


# @pytest.mark.skipif( os.environ.get("SKIP_SQL_ADMIN_TESTS","False").lower() == 'true', reason="Don't want to run this in CICD.")
# class TestSqliteLegacyStoredBuild(BaseLegacyStoredBuildTest):

#     @classmethod
#     def _get_storage_factory(cls):
#         # This makes it a SQL test.
#         return SqliteStorageFactory()
