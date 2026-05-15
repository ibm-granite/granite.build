import pytest
from lib.storage.build_storage import BaseBuildStorageTest

from gbserver.storage.shadowed.storage_factory import DualSQLSqliteStorageFactory

pytestmark = pytest.mark.ibm


class TestDualSQLSqliteBuildStorage(BaseBuildStorageTest):

    @classmethod
    def _get_storage_factory(cls):
        return DualSQLSqliteStorageFactory()
