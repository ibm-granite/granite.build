import pytest
from lib.storage.artifact_storage import BaseArtifactStorageTest

from gbserver.storage.shadowed.storage_factory import DualSQLSqliteStorageFactory

pytestmark = pytest.mark.ibm


class TestDualSQLSqliteArtifactStorage(BaseArtifactStorageTest):

    @classmethod
    def _get_storage_factory(cls):
        return DualSQLSqliteStorageFactory()
