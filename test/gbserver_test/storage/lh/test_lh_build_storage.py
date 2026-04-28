import os

import pytest
from gbserver_test.storage.test_build_storage import (
    BaseBuildStorageTest,
    BaseLegacyStoredBuildTest,
)

import gbserver.storage.lh.build_storage as BUILD
from gbserver.storage.lh.storage_factory import LhStorageFactory
from gbserver.storage.storage import CREATED_TIME_FIELD_NAME, UPDATED_TIME_FIELD_NAME

pytestmark = pytest.mark.ibm


@pytest.mark.skipif(
    os.environ.get("SKIP_LAKEHOUSE_ADMIN_TESTS", "False").lower() == "true"
    or os.environ.get("GBTEST_SKIP_LAKEHOUSE_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD anymore.",
)
class TestLhBuildStorage(BaseBuildStorageTest):

    @classmethod
    def _get_storage_factory(cls):
        # This makes it a Lakehouse test.
        return LhStorageFactory()

    def test_v2_schema_change(self):
        storage = self._get_tested_storage()

        # Add an item using the v1 schema
        item = self._get_test_item(0)
        BUILD._IS_V2_SCHEMA = False
        storage.add(item)  # This first add(), creates the table with the pre-V2 schema.
        v1_columns = storage.get_column_names()

        # Start using v2 schema
        BUILD._IS_V2_SCHEMA = True

        # Add an item with v2 schema set should trigger the automatic addition of the "created_time" and "updated_time" column
        item = self._get_test_item(1)
        storage.add(item)
        item = self._get_test_item(2)
        storage.add(item)
        v2_columns = storage.get_column_names()

        # Make sure the column added in v2 is "created_time" and "updated_time"
        assert len(v2_columns) == len(v1_columns) + 2
        for x in v1_columns:
            v2_columns.remove(x)
        assert len(v2_columns) == 2
        assert CREATED_TIME_FIELD_NAME in v2_columns
        assert UPDATED_TIME_FIELD_NAME in v2_columns

        items = storage.get_by_where()
        assert len(items) == 3

        items = storage.get_by_where(
            {CREATED_TIME_FIELD_NAME: getattr(item, CREATED_TIME_FIELD_NAME)}
        )
        assert len(items) == 1

        items = storage.get_by_where(
            {UPDATED_TIME_FIELD_NAME: getattr(item, UPDATED_TIME_FIELD_NAME)}
        )
        assert len(items) == 1


@pytest.mark.skipif(
    os.environ.get("SKIP_LAKEHOUSE_ADMIN_TESTS", "False").lower() == "true"
    or os.environ.get("GBTEST_SKIP_LAKEHOUSE_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD anymore.",
)
class TestLhLegacyStoredBuild(BaseLegacyStoredBuildTest):

    @classmethod
    def _get_storage_factory(cls):
        # This makes it a Lakehouse test.
        return LhStorageFactory()
