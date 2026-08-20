import os

import pytest
from libgbtest.storage.kv_pair_storage import (
    BaseKeyValuePairStorageTest,
)
from libgbtest.storage.kv_pair_storage import (
    TestKeyValuePairValueMethods as _TestKeyValuePairValueMethods,
)

from gbserver.storage.sql.storage_factory import SQLStorageFactory

pytestmark = pytest.mark.ibm


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSQLKeyValuePairStorage(BaseKeyValuePairStorageTest):

    @classmethod
    def _get_storage_factory(cls):
        return SQLStorageFactory()


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSQLKeyValuePairValueMethods(_TestKeyValuePairValueMethods):

    @classmethod
    def _get_storage_factory(cls):
        return SQLStorageFactory()
