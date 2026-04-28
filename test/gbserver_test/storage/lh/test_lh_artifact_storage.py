import os

import pytest
from gbserver_test.storage.test_artifact_storage import (
    BaseArtifactStorageTest,
    BaseLegacyArtifactStorageTest,
)

from gbserver.storage.lh import artifact_registry as lh_artifact_registry
from gbserver.storage.lh.storage_factory import LhStorageFactory

pytestmark = pytest.mark.ibm


@pytest.mark.skipif(
    os.environ.get("SKIP_LAKEHOUSE_ADMIN_TESTS", "False").lower() == "true"
    or os.environ.get("GBTEST_SKIP_LAKEHOUSE_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD anymore.",
)
class TestLhArtifactStorage(BaseArtifactStorageTest):

    @classmethod
    def _get_storage_factory(cls):
        # This makes it a Lakehouse test.
        return LhStorageFactory()

    def test_v2_schema_change(self):
        storage = self._get_tested_storage()

        # Add an item using the v1 schema
        item = self._get_test_item(0)
        lh_artifact_registry._IS_V2_SCHEMA = False
        storage.add(item)  # This first add(), creates the table with the pre-V2 schema.
        v1_columns = storage.get_column_names()

        # Start using v2 schema
        lh_artifact_registry._IS_V2_SCHEMA = True

        # Add an item with v2 schema set should trigger the automatic addition of the "is_archived" column
        item = self._get_test_item(1)
        storage.add(item)
        item = self._get_test_item(2)
        item.is_archived = True
        storage.add(item)
        v2_columns = storage.get_column_names()

        # Make sure the column added in v2 is "is_archived"
        assert len(v2_columns) == len(v1_columns) + 1
        for x in v1_columns:
            v2_columns.remove(x)
        assert v2_columns[0] == "is_archived"

        items = storage.get_by_where()
        assert len(items) == 3

        items = storage.get_by_where({"is_archived": True})
        assert len(items) == 1

        items = storage.get_by_where({"is_archived": False})
        assert len(items) == 2


@pytest.mark.skipif(
    os.environ.get("SKIP_LAKEHOUSE_ADMIN_TESTS", "False").lower() == "true"
    or os.environ.get("GBTEST_SKIP_LAKEHOUSE_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD anymore.",
)
class TestLhLegacyArtifactStorage(BaseLegacyArtifactStorageTest):

    @classmethod
    def _get_storage_factory(cls):
        # This makes it a Lakehouse test.
        return LhStorageFactory()
