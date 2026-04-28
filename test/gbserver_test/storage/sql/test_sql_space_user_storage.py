
import os
import uuid
import pytest
from gbserver.storage.sql.storage_factory import SQLStorageFactory
from gbserver.storage.storage import UUID_COLUMN_NAME, BaseItemStorage, BaseStoredItem
from gbserver.storage.stored_space_user import StoredSpaceUser
from gbserver_test.storage.test_space_user_storage import BaseSpaceUserStorageTest


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSQLSpaceUserStorage(BaseSpaceUserStorageTest):

    @classmethod
    def _get_storage_factory(cls):
        return SQLStorageFactory()

    def _get_where_test_item(self, index: int) -> StoredSpaceUser:
        # Give each call a unique (space_name, username) pair to avoid the unique
        # constraint violation, but assign role based on index parity so the
        # multi-match WHERE (on role only) distinguishes index=0 from index=1.
        item = super()._get_where_test_item(index)
        unique_suffix = str(uuid.uuid4())[:8]
        item.space_name = f"space{index}_{unique_suffix}"
        item.username = f"user{index}_{unique_suffix}"
        item.role = "member" if index % 2 == 0 else "admin"
        return item

    def _get_where_search_columns(self, storage: BaseItemStorage, item: BaseStoredItem) -> dict:
        columns = super()._get_where_search_columns(storage, item)
        # (space_name, username) is a unique pair — exclude both so the multi-match
        # where test can insert two rows that differ only by those columns.
        columns.pop("space_name", None)
        columns.pop("username", None)
        return columns

    def test_count_with_where(self):
        # The base test uses _get_where_search_columns which strips space_name and username,
        # leaving only role. All test items share role="member", so filtering by role returns
        # count=3 instead of 1. Use uuid directly since it is always unique.
        storage = self._get_tested_storage()

        item0 = self._get_test_item(0)
        item1 = self._get_test_item(1)
        item2 = self._get_test_item(2)
        storage.add([item0, item1, item2])

        assert storage.count() == 3, "Expected count of 3 for all items"

        count_filtered = storage.count(where={UUID_COLUMN_NAME: item0.uuid})
        assert count_filtered == 1, "Expected count of 1 when filtering by uuid"

        item3 = self._get_test_item(3)
        count_none = storage.count(where={UUID_COLUMN_NAME: item3.uuid})
        assert count_none == 0, "Expected count of 0 when no items match"
