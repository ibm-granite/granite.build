import os

import pytest
from gbserver_test.storage.test_step_storage import BaseStepStorageTest

from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSqliteStepStorage(BaseStepStorageTest):

    @classmethod
    def _is_cloud_config_required(cls) -> bool:
        return False

    @classmethod
    def _get_storage_factory(cls):
        return SqliteStorageFactory()


# @pytest.mark.skipif( os.environ.get("SKIP_SQL_ADMIN_TESTS","False").lower() == 'true', reason="Don't want to run this in CICD.")
# class TestSQLLegacyStoredStep(BaseLegacyStoredStepTest):

#     @classmethod
#     def _get_storage_factory(cls):
#         # This makes it a SQL test.
#         return SqliteStorageFactory()
