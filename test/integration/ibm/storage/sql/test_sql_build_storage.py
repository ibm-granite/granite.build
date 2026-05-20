import os

import pytest
from libgbtest.storage.build_storage import (
    BaseBuildStorageTest,
    BaseLegacyStoredBuildTest,
)

from gbserver.storage.build_storage import (
    _BUILD_SCHEMA_VERSION1,
    _BUILD_SCHEMA_VERSION2,
    BaseStoredBuildStorage,
)
from gbserver.storage.sql.build_storage import SQLBuildStorage
from gbserver.storage.sql.storage_factory import SQLStorageFactory

pytestmark = pytest.mark.ibm
# from gbserver.storage import build_storage as xyz_build_storage


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSQLBuildStorage(BaseBuildStorageTest):

    @classmethod
    def _get_storage_factory(cls):
        # This makes it a SQL test.
        return SQLStorageFactory()

    def test_tags_column_addition(self):

        # Add an item using old schema
        storage1 = self.storage.build_storage
        assert isinstance(storage1, BaseStoredBuildStorage)
        storage1._schema_version = _BUILD_SCHEMA_VERSION1
        build1 = self._get_test_item(1)
        storage1.add(build1)
        old_columns = storage1.get_column_names()

        # Add an item using new schema
        # xyz_build_storage._BUILD_SCHEMA_VERSION = 2
        # storage2 = SQLBuildStorage(table_name=storage1.get_table_name())
        storage2 = self._get_storage_factory().create_build_storage(
            table_name=storage1.get_table_name()
        )
        assert isinstance(storage2, BaseStoredBuildStorage)
        storage2._schema_version = _BUILD_SCHEMA_VERSION2
        build2 = self._get_test_item(2)
        storage2.add(build2)

        # Confirm that the version upgrade added the tags column
        new_columns = storage2.get_column_names()
        added_columns = [item for item in new_columns if item not in old_columns]
        assert len(added_columns) == 1
        assert "tags" in added_columns

        # Now read both items with storage2
        items = storage2.get_by_uuid(None)
        self._verify_get_results(items, [build1, build2], ordered=False)

        # Now read both items with storage1
        # xyz_build_storage._BUILD_SCHEMA_VERSION = 1 # Currently, this is not necessary since it is only used on add/update.
        items = storage1.get_by_uuid(None)
        # storage1._schema_version = _BUILD_SCHEMA_VERSION1
        self._verify_get_results(items, [build1, build2], ordered=False)


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSQLLegacyStoredBuild(BaseLegacyStoredBuildTest):

    @classmethod
    def _get_storage_factory(cls):
        # This makes it a SQL test.
        return SQLStorageFactory()
