import os

import pytest
from gbserver_test.storage.test_artifact_storage import BaseArtifactStorageTest

from gbserver.storage.shadowed.storage_factory import (
    LhSQLStorageFactory,
    SQLLhStorageFactory,
)

pytestmark = pytest.mark.ibm


@pytest.mark.skipif(
    os.environ.get("SKIP_SHADOW_ADMIN_TESTS", "False").lower() == "true"
    or os.environ.get("GBTEST_SKIP_SHADOW_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSQLLhArtifactStorage(BaseArtifactStorageTest):

    @classmethod
    def _get_storage_factory(cls):
        return SQLLhStorageFactory()


@pytest.mark.skipif(
    os.environ.get("SKIP_SHADOW_ADMIN_TESTS", "False").lower() == "true"
    or os.environ.get("GBTEST_SKIP_SHADOW_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestLhSQLArtifactStorage(BaseArtifactStorageTest):

    @classmethod
    def _get_storage_factory(cls):
        return LhSQLStorageFactory()
