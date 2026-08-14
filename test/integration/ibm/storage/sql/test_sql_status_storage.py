import os

import pytest
from libgbtest.storage.status_storage import (
    BaseStatusStorageTest,
)
from libgbtest.storage.status_storage import (
    TestStatusValueMethods as _TestStatusValueMethods,
)

from gbserver.storage.sql.storage_factory import SQLStorageFactory

pytestmark = pytest.mark.ibm


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSQLStatusStorage(BaseStatusStorageTest):

    @classmethod
    def _get_storage_factory(cls):
        return SQLStorageFactory()


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSQLStatusValueMethods(_TestStatusValueMethods):

    @classmethod
    def _get_storage_factory(cls):
        return SQLStorageFactory()
