import os

import pytest
from gbserver_test.storage.test_space_storage import (
    BaseLegacyStoredSpaceTest,
    BaseSpaceStorageTest,
)

from gbserver.storage.lh.storage_factory import LhStorageFactory

pytestmark = pytest.mark.ibm


@pytest.mark.skipif(
    os.environ.get("SKIP_LAKEHOUSE_ADMIN_TESTS", "False").lower() == "true"
    or os.environ.get("GBTEST_SKIP_LAKEHOUSE_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD anymore.",
)
class TestLhSpaceStorage(BaseSpaceStorageTest):

    @classmethod
    def _get_storage_factory(cls):
        # This makes it a Lakehouse test.
        return LhStorageFactory()


@pytest.mark.skipif(
    os.environ.get("SKIP_LAKEHOUSE_ADMIN_TESTS", "False").lower() == "true"
    or os.environ.get("GBTEST_SKIP_LAKEHOUSE_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD anymore.",
)
class TestLhLegacyStoredSpace(BaseLegacyStoredSpaceTest):

    @classmethod
    def _get_storage_factory(cls):
        # This makes it a Lakehouse test.
        return LhStorageFactory()
