import os

import pytest
from gbserver_test.storage.test_target_storage import (
    BaseLegacyStoredTargetTest,
    BaseTargetStorageTest,
)

from gbserver.storage.shadowed.storage_factory import (
    LhSQLStorageFactory,
    SQLLhStorageFactory,
)
from gbserver.storage.sql.storage_factory import SQLStorageFactory

pytestmark = pytest.mark.ibm


@pytest.mark.skipif(
    os.environ.get("SKIP_SHADOW_ADMIN_TESTS", "False").lower() == "true"
    or os.environ.get("GBTEST_SKIP_SHADOW_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSQLLhTargetStorage(BaseTargetStorageTest):

    @classmethod
    def _get_storage_factory(cls):
        return SQLLhStorageFactory()


@pytest.mark.skipif(
    os.environ.get("SKIP_SHADOW_ADMIN_TESTS", "False").lower() == "true"
    or os.environ.get("GBTEST_SKIP_SHADOW_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestLhSQLTargetStorage(BaseTargetStorageTest):

    @classmethod
    def _get_storage_factory(cls):
        return LhSQLStorageFactory()
