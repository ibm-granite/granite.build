import pytest
from lib.storage.space_storage import BaseSpaceStorageTest

from gbserver.storage.shadowed.storage_factory import DualSQLSqliteStorageFactory

pytestmark = pytest.mark.ibm


class TestDualSQLSqliteSpaceStorage(BaseSpaceStorageTest):

    @classmethod
    def _get_storage_factory(cls):
        return DualSQLSqliteStorageFactory()
