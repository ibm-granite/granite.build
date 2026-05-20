import os

import pytest
from libgbtest.storage.node_failure_storage import (
    BaseLegacyNodeFailureTest,
    BaseNodeFailureStorageTest,
)

from gbserver.storage.sql.storage_factory import SQLStorageFactory

pytestmark = pytest.mark.ibm


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSQLNodeFailureStorage(BaseNodeFailureStorageTest):

    @classmethod
    def _get_storage_factory(cls):
        return SQLStorageFactory()


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSQLLegacyNodeFailure(BaseLegacyNodeFailureTest):

    @classmethod
    def _get_storage_factory(cls):
        return SQLStorageFactory()
