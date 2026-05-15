import pytest
from lib.storage.target_storage import BaseTargetStorageTest

from gbserver.storage.shadowed.storage_factory import DualSQLSqliteStorageFactory

pytestmark = pytest.mark.ibm


class TestDualSQLSqliteTargetStorage(BaseTargetStorageTest):

    @classmethod
    def _get_storage_factory(cls):
        return DualSQLSqliteStorageFactory()
