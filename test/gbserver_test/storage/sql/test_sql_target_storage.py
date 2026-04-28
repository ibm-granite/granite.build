import os

import pytest
from gbserver_test.storage.test_target_storage import (
    BaseLegacyStoredTargetTest,
    BaseTargetStorageTest,
)

from gbserver.storage.sql.storage_factory import SQLStorageFactory

pytestmark = pytest.mark.ibm


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSQLTargetStorage(BaseTargetStorageTest):

    @classmethod
    def _get_storage_factory(cls):
        # This makes it a SQL test.
        return SQLStorageFactory()


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSQLLegacyStoredTarget(BaseLegacyStoredTargetTest):

    @classmethod
    def _get_storage_factory(cls):
        # This makes it a SQL test.
        return SQLStorageFactory()
