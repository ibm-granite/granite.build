import os

import pytest
from gbserver_test.storage.test_target_storage import (
    BaseLegacyStoredTargetTest,
    BaseTargetStorageTest,
)

from gbserver.storage.lh.storage_factory import LhStorageFactory

pytestmark = pytest.mark.ibm


@pytest.mark.skipif(
    os.environ.get("SKIP_LAKEHOUSE_ADMIN_TESTS", "False").lower() == "true"
    or os.environ.get("GBTEST_SKIP_LAKEHOUSE_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD anymore.",
)
class TestLhTargetStorage(BaseTargetStorageTest):

    @classmethod
    def _get_storage_factory(cls):
        # This makes it a Lakehouse test.
        return LhStorageFactory()


@pytest.mark.skipif(
    os.environ.get("SKIP_LAKEHOUSE_ADMIN_TESTS", "False").lower() == "true"
    or os.environ.get("GBTEST_SKIP_LAKEHOUSE_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD anymore.",
)
class TestLhLegacyStoredTarget(BaseLegacyStoredTargetTest):

    @classmethod
    def _get_storage_factory(cls):
        # This makes it a Lakehouse test.
        return LhStorageFactory()
