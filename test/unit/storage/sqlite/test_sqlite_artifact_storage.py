import os

import pytest
from libgbtest.storage.artifact_storage import BaseArtifactStorageTest

from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSqliteArtifactStorage(BaseArtifactStorageTest):

    @classmethod
    def _get_storage_factory(cls):
        return SqliteStorageFactory()


# @pytest.mark.skipif( os.environ.get("SKIP_SQL_ADMIN_TESTS","False").lower() == 'true', reason="Don't want to run this in CICD.")
# class TestSQLLegacyArtifactStorage(BaseLegacyArtifactStorageTest):

#     @classmethod
#     def _get_storage_factory(cls):
#         # This makes it a SQL test.
#         return SqliteStorageFactory()
