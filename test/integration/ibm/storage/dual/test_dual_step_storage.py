import pytest
from lib.storage.step_storage import BaseStepStorageTest

from gbserver.storage.shadowed.storage_factory import DualSQLSqliteStorageFactory

pytestmark = pytest.mark.ibm


class TestDualSQLSqliteStepStorage(BaseStepStorageTest):

    @classmethod
    def _get_storage_factory(cls):
        return DualSQLSqliteStorageFactory()
