import os

import pytest
from gbserver_test.storage.test_space_storage import (
    BaseLegacyStoredSpaceTest,
    BaseSpaceStorageTest,
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
class TestSQLLhSpaceStorage(BaseSpaceStorageTest):

    @classmethod
    def _get_storage_factory(cls):
        # This makes it a SQL as primary.
        return SQLLhStorageFactory()


@pytest.mark.skipif(
    os.environ.get("SKIP_SHADOW_ADMIN_TESTS", "False").lower() == "true"
    or os.environ.get("GBTEST_SKIP_SHADOW_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestLhSQLSpaceStorage(BaseSpaceStorageTest):

    @classmethod
    def _get_storage_factory(cls):
        # This makes it a SQL as secondary.
        return LhSQLStorageFactory()
