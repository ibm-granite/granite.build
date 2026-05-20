import time
from abc import abstractmethod
from typing import Any, Self

import pytest
from libgbtest.utils import (
    AbstractReadonlySingletonStorageUsingTest,
    AbstractSingletonStorageUsingTest,
)

from gbserver.storage import singleton_storage
from gbserver.storage.storage import (
    CREATED_TIME_FIELD_NAME,
    JSON_COLUMN_NAME,
    UPDATED_TIME_FIELD_NAME,
    UUID_COLUMN_NAME,
    BaseItemStorage,
    BaseStoredItem,
    Pagination,
    QueryControl,
    SortOrder,
    TaggedItem,
)
from gbserver.utils.logger import get_logger
from gbserver.utils.utils import get_differing_attributes

logger = get_logger(__name__)


class AbstractStorageTestSupport:

    def __init__(self, sort_column: str):
        """

        Args:
            sort_column (str): the name of the column that sorts the items returned by _get_ascending_sorted_test_items().
        """
        self.sort_column = sort_column

    @abstractmethod
    def _get_test_item(self, index: int) -> BaseStoredItem:
        raise NotImplemented(
            "Sub-classes must implement this to create instances to be stored in the test storage"
        )

    def _get_ascending_sorted_test_items(
        self, count: int
    ) -> tuple[str, list[BaseStoredItem]]:
        """Get a list of items to store sorted by the returned column name.
        This uses the _get_test_item() method to generate the items and the self.sort_column which is expected to match
        the sort order of the indexed items 0,1,2,3,...

        Args:
            count[int]:  number of items to return in the list.

        Returns:
            tuple[str,list[BaseStoredItem]]: 1st member is the column to use to sort the query results by to match the 2nd item
            2nd is a list of items sorted in ascending order per the returned column name.
        """
        items = []
        for index in range(count):
            item = self._get_test_item(index)
            items.append(item)

        return self.sort_column, items


class AbstractStorageTest(AbstractSingletonStorageUsingTest):
    """
    This class defines the tests on concrete extension of the BaseItemStorage class.
    The tests are generally independent of the item stored in the storage.
    To use this on a new BaseItemStorage implementation, extend this class and implement...
        - _get_tested_storage() to return an instead of the BaseItemStorage extension being tested.
        - _get_test_item() to create instances of the items stored in the storage being tested.
        - any other test methods (test_*()) that might specific to the tested BaseItemStorage.

    Raises:
        NotImplemented: when the expected _get_tested_storage() method is not implemented by the sub-class
        NotImplemented: when the expected _get_test_item() method is not implemented by the sub-class

    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.config = cls._get_test_config()

    def _get_storage_to_clear(self) -> list[BaseItemStorage]:
        """Override so that we only clear the storage instance being tested, thereby speeding up the tests, hopefully. Seems to reduce times by ~15%"""
        return [self._get_tested_storage()]

    @classmethod
    @abstractmethod
    def _get_test_config(cls) -> AbstractStorageTestSupport:
        raise NotImplemented(
            "Sub-classes must implement this to create instances to be stored in the test storage"
        )

    def _get_tested_storage(self) -> BaseItemStorage:
        raise NotImplemented(
            "Sub-classes must implement this to get the storage under test from self.storage"
        )
        # return self.config._get_tested_storage()

    def _get_test_item(self, index: int) -> BaseStoredItem:
        return self.config._get_test_item(index)

    def _get_ascending_sorted_test_items(
        self, count: int
    ) -> tuple[str, list[BaseStoredItem]]:
        return self.config._get_ascending_sorted_test_items(count)

    def _get_where_test_item(self, index: int) -> BaseStoredItem:
        return self.config._get_test_item(index)

    def test_add(self):

        # Get empty test storage
        storage = self._get_tested_storage()

        # add 1st item.
        item0 = self._get_test_item(0)
        uuid = storage.add(item0)
        assert uuid is not None, "Did not get back a uuid from added item"
        item = storage.get_by_uuid(item0.uuid)
        assert item is not None, "Stored item was not retrieved from storage"
        # print(f"diffs={get_differing_attributes(item0.build_event,item.build_event)}")
        assert item0 == item, "Stored item not equal"

        # add 2nd item.
        item1 = self._get_test_item(1)
        uuid = storage.add(item1)
        assert uuid is not None, "Did not get back a uuid from added item"

        # Make sure we can't use duplicate uuids
        try:
            item2 = self._get_test_item(2)
            item2.uuid = item1.uuid
            storage.add(item2)  # diff uri, same uuid.
            assert False, "Was allowed to store the same item twice"
        except Exception as exc:
            logger.info(
                "Storage may have issued a WARNING or ERROR above for adding a duplicate uuid and can safely be ignored."
            )
            pass

        # Get 2 items at a time
        item2 = self._get_test_item(2)
        item3 = self._get_test_item(3)
        uuids = storage.add([item2, item3])
        assert len(uuids) == 2, f"Got back {len(uuids)} uuids for 2 stored items"
        assert uuids == [item2.uuid, item3.uuid]
        items: list = storage.get_by_uuid(uuids)
        assert len(items) == 2
        assert items[0] == item2, "Did not get back expected item 0"
        assert items[1] == item3, "Did not get back expected item 1"

    def test_get_by_uuid(self):

        # Get empty storage
        storage = self._get_tested_storage()

        # Try reading from storage before adding an item to make sure it can handle initilization w/o add()
        items = storage.get_by_uuid(None)
        self._verify_get_results(items, [])

        # Add one item
        item0 = self._get_test_item(0)
        uuid0 = storage.add(item0)

        # Test getting item item
        item = storage.get_by_uuid(uuid0)
        self._verify_get_results(item, item0)

        # Add some more items
        item1 = self._get_test_item(1)
        uuid1 = storage.add(item1)
        item2 = self._get_test_item(2)  # add an extra item that we don't expect back
        uuid2 = storage.add(item2)

        # Test getting all items.
        items = storage.get_by_uuid(None)
        self._verify_get_results([item0, item1, item2], items, ordered=False)

        # Now test retrieving more than 1 at a time.
        items = storage.get_by_uuid([uuid0, uuid1])
        self._verify_get_results([item0, item1], items)

        # Now test retrieving non-existent uuids
        item = storage.get_by_uuid("non-uuid")
        assert item == None, "Should have gotten None for non-existent uuid"

        # Test 1st uid bad in a list
        items = storage.get_by_uuid(["non-uuid", uuid0])
        self._verify_get_results([None, item0], items)

        # Test 2nd uid bad in a list
        items = storage.get_by_uuid([uuid0, "non-uuid"])
        self._verify_get_results([item0, None], items)

    def disable_test_get_by_where_clause(self):
        """Disabled for now since we don't expect WHERE clauses, only row_filter since those happen in the server and not the client"""
        self.__test_get_by_where_multimatch(False)

    def test_get_by_where_dict_multimatch(self):
        self.__test_get_by_where_multimatch(True)

    def __test_get_by_where_multimatch(self, use_dict: bool):
        """Test where search with more than 1 item matches the where"""

        # Get empty storage
        storage = self._get_tested_storage()

        item0_0 = self._get_where_test_item(0)
        item0_1 = self._get_where_test_item(0)
        item1_0 = self._get_where_test_item(1)
        uuid0_0 = storage.add(item0_0)
        uuid0_1 = storage.add(item0_1)
        uuid1_0 = storage.add(item1_0)

        if use_dict:  # Only need to test these once.
            # Search for all items
            items = storage.get_by_where()
            self._verify_get_results([item0_0, item0_1, item1_0], items, ordered=False)

            # Search for all items using empty dictionary
            items = storage.get_by_where({})
            self._verify_get_results([item0_0, item0_1, item1_0], items, ordered=False)

        if use_dict:
            where = self.__get_where_dict(storage, item0_0)
        else:
            where = self.__get_where_clause(storage, item0_0)
        if where is None:
            print("No columns defined for item. Skipping where test")
            return

        # Search for items matching item0, should be _0 and _1 which have the same column values
        items = storage.get_by_where(where)
        self._verify_get_results([item0_0, item0_1], items, ordered=False)

        # Search for item matching item1, should be _0 only
        if use_dict:
            where = self.__get_where_dict(storage, item1_0)
        else:
            where = self.__get_where_clause(storage, item1_0)
        items = storage.get_by_where(where)
        self._verify_get_results([item1_0], items)

    def test_get_by_where_dict_singlematch(self):
        self.__test_get_by_where_singlematch(True)

    def __test_get_by_where_singlematch(self, use_dict: bool):
        """Test simple where search with only a single item matching the where"""
        # Get empty storage
        storage = self._get_tested_storage()

        item0_0 = self._get_where_test_item(0)
        item1_0 = self._get_where_test_item(1)
        uuid0_0 = storage.add(item0_0)
        uuid1_0 = storage.add(item1_0)

        if use_dict:
            where = self.__get_where_dict(storage, item0_0)
        else:
            where = self.__get_where_clause(storage, item0_0)
        if where is None:
            print("No columns defined for item. Skipping where test")
            return

        # Search for items matching item0,
        items = storage.get_by_where(where)
        self._verify_get_results([item0_0], items, ordered=False)

        # Search for item matching item1, should be _0 only
        if use_dict:
            where = self.__get_where_dict(storage, item1_0)
        else:
            where = self.__get_where_clause(storage, item1_0)
        items = storage.get_by_where(where)
        self._verify_get_results([item1_0], items)

    def test_search_by_time(self):
        """Test where search on created/updated_times here since they are otherwise not included in search where criteria"""

        item0 = self._get_where_test_item(0)
        item1 = self._get_where_test_item(1)
        item0_created_time = getattr(item0, CREATED_TIME_FIELD_NAME, None)
        item0_updated_time = getattr(item0, UPDATED_TIME_FIELD_NAME, None)
        if item0_created_time is None and item0_updated_time is None:
            pytest.skip(
                f"Stored items contain neither {CREATED_TIME_FIELD_NAME} or {UPDATED_TIME_FIELD_NAME} fields."
            )
            return

        storage = self._get_tested_storage()
        storage.add([item0, item1])

        # The add updated the two timestamp fields of the items passed in, so get the new matching values
        item0_created_time = getattr(item0, CREATED_TIME_FIELD_NAME, None)
        item0_updated_time = getattr(item0, UPDATED_TIME_FIELD_NAME, None)

        if item0_created_time is not None:
            where = {CREATED_TIME_FIELD_NAME: item0_created_time}
            items = storage.get_by_where(where)
            self._verify_get_results([item0], items)
        if item0_updated_time is not None:
            where = {UPDATED_TIME_FIELD_NAME: item0_updated_time}
            items = storage.get_by_where(where)
            self._verify_get_results([item0], items)

    def test_delete(self):

        # Get empty test storage
        storage = self._get_tested_storage()

        items = []
        uuids = []
        # add 4 items
        for index in range(4):
            item = self._get_test_item(index)
            items.append(item)
            xuuid = storage.add(item)
            assert xuuid is not None, "Did not get back a uuid from added item"
            uuids.append(xuuid)

        # Delete item 0
        storage.delete(uuids[0])
        got_items = storage.get_by_uuid(uuids[0])
        self._verify_get_results(None, got_items)

        # Delete items 1 and 2 using a list, leaving 3
        storage.delete([uuids[1], uuids[2]])
        got_items = storage.get_by_uuid(uuids[1])
        self._verify_get_results(None, got_items)
        got_items = storage.get_by_uuid(uuids[2])
        self._verify_get_results(None, got_items)

        # Make sure  item 3 is still present when requesting it
        got_items = storage.get_by_uuid(uuids[3])
        self._verify_get_results(items[3], got_items)

        # Get all remaining items and expect 3 to still be there
        got_items = storage.get_by_uuid(None)
        self._verify_get_results([items[3]], got_items)

    def test_update(self):

        # Get empty test storage
        storage = self._get_tested_storage()

        # Test create on update when the table does not exists (seems to be a problem)
        item3 = self._get_test_item(3)
        storage.update(item3, update_updated_time=True, create_if_not_exist=True)
        items = storage.get_by_uuid(item3.uuid)
        self._verify_get_results(item3, items)

        storage.delete_table()

        # First try and update a non-existent item
        item0 = self._get_test_item(0)
        try:
            storage.update(item0, update_updated_time=True, create_if_not_exist=False)
            assert False, "Was allowed to update non-existing item."
        except:
            logger.info(
                "Storage may have issued a WARNING or ERROR above for adding a duplicate uuid and can safely be ignored."
            )
            pass

        # Now add 1st item.
        uuid0 = storage.add(item0)
        assert uuid0 is not None, "Did not get back a uuid from added item"

        # add 2nd item.
        item1 = self._get_test_item(1)
        uuid1 = storage.add(item1)
        assert uuid1 is not None, "Did not get back a uuid from added item"

        # Verify uuid1 still has its item1 value
        items = storage.get_by_uuid(uuid1)
        self._verify_get_results(item1, items)

        # Now update item1's uuid with item 10's value
        updated_item1 = self._get_test_item(10)
        updated_item1.uuid = item1.uuid
        if hasattr(item1, CREATED_TIME_FIELD_NAME):
            setattr(
                updated_item1,
                CREATED_TIME_FIELD_NAME,
                getattr(item1, CREATED_TIME_FIELD_NAME),
            )
        if hasattr(item1, UPDATED_TIME_FIELD_NAME):
            setattr(
                updated_item1,
                UPDATED_TIME_FIELD_NAME,
                getattr(item1, UPDATED_TIME_FIELD_NAME),
            )
        assert (
            updated_item1 != item1
        ), "the items should have different values for this test to be valid"
        storage.update(
            updated_item1, update_updated_time=False
        )  # Update item0 with item1 (which now has item0's uuid)
        items = storage.get_by_uuid(item1.uuid)
        self._verify_get_results(
            items, updated_item1
        )  # Verify that the item stored under item0's uuid == item1

        # Make sure we can create a new record with update.
        item3 = self._get_test_item(3)
        storage.update(item3, update_updated_time=True, create_if_not_exist=True)
        items = storage.get_by_uuid(item3.uuid)
        self._verify_get_results(item3, items)

        # Make sure we get an exception if we don't want to create the record.
        item4 = self._get_test_item(4)
        try:
            storage.update(item4, update_updated_time=True, create_if_not_exist=False)
            assert False, "Should have gotten exception since item did not pre-exist"
        except Exception:
            pass
        items = storage.get_by_uuid(item4.uuid)
        self._verify_get_results(None, items)

        # Test create on update when the table already existed
        storage.delete_table()
        item3 = self._get_test_item(3)
        storage.update(item3, update_updated_time=True, create_if_not_exist=True)
        items = storage.get_by_uuid(item3.uuid)
        self._verify_get_results(item3, items)

    def test_updated_time(self):
        item = self._get_test_item(0)

        if not hasattr(item, UPDATED_TIME_FIELD_NAME):
            pytest.skip(
                f"Items do not include an {UPDATED_TIME_FIELD_NAME} field so skipping this test"
            )
            return

        # Store an item
        storage = self._get_tested_storage()
        storage.add(item)

        # Retrieve the stored item and its updated time.
        uuid = item.uuid
        stored_item = storage.get_by_uuid(uuid)
        t1 = getattr(stored_item, UPDATED_TIME_FIELD_NAME)

        # make sure we get different update times.
        time.sleep(0.100)

        # Update the item and expect different times.
        storage.update(item, update_updated_time=True)
        stored_item = storage.get_by_uuid(uuid)
        t2 = getattr(stored_item, UPDATED_TIME_FIELD_NAME)
        assert t1 != t2, "Updated times are supposed to be different"
        t1 = t2

        # Update the item and expect the same times.
        storage.update(item, update_updated_time=False)
        stored_item = storage.get_by_uuid(uuid)
        t2 = getattr(stored_item, UPDATED_TIME_FIELD_NAME)
        assert t1 == t2, "Updated times are supposed to be the same"

    def test_update_fields_invalid_field_name(self):
        """Test that update_fields raises ValueError for invalid field names."""
        storage = self._get_tested_storage()

        # Add an item to storage
        item = self._get_test_item(0)
        storage.add(item)

        # Try to update a field that doesn't exist
        with pytest.raises(ValueError) as exc_info:
            storage.update_fields(item.uuid, {"nonexistent_field": "some_value"})
        assert "nonexistent_field" in str(exc_info.value)

        # Try to update with a non-existent uuid
        with pytest.raises(ValueError) as exc_info:
            storage.update_fields("nonexistent-uuid", {"uuid": "new_value"})
        assert "nonexistent-uuid" in str(exc_info.value)

    def _get_where_search_columns(
        self, storage: BaseItemStorage, item: BaseStoredItem
    ) -> dict[str, Any]:
        """Get the list of columns which should be searched on from the item to return more than 1 value.
        Sub-classes (i.e. artifactregistry) can override to remove some columns, especially those that require unique values
        like the uri field of artifacts.
        Be default, we use the columns names/values as defined by the storage instance get_column_values(item).
        """
        columns = storage._get_column_values(item)
        # Ignore timestamps since some of the tests expect to match multiple items for a single query
        if getattr(item, CREATED_TIME_FIELD_NAME, None):
            del columns[CREATED_TIME_FIELD_NAME]
        if getattr(item, UPDATED_TIME_FIELD_NAME, None):
            del columns[UPDATED_TIME_FIELD_NAME]
        # For TaggedItems, the where query on tags is a list of strings and not a single csv string of the tag values
        # as is produced by _get_column_values() above.
        if isinstance(item, TaggedItem):
            value = item.tags
            if value is not None and len(value) > 0:
                columns["tags"] = value

        return columns

    def __get_where_clause(self, storage: BaseItemStorage, item: BaseStoredItem) -> str:
        columns = self._get_where_search_columns(storage, item)
        if len(columns) == 0:
            return None
        clause = None
        for key, value in columns.items():
            if not clause is None:
                clause = clause + " AND "
            else:
                clause = ""
            if isinstance(value, str):
                expr = f"{key} = '{value}'"
            else:
                expr = f"{key} = {value}"
            clause = clause + expr
        return clause

    def __get_where_dict(self, storage: BaseItemStorage, item: BaseStoredItem) -> dict:
        columns = self._get_where_search_columns(storage, item)
        if len(columns) == 0:
            return None
        return columns

    def _verify_equal(self, result, expected):
        """Enables refined object comparison, for example, to allow ignoring updated_time field, when it is present."""
        if hasattr(result, UPDATED_TIME_FIELD_NAME):
            # set the updated_time fields to be the same, effectively ignoring them.
            result_updated_time = getattr(result, UPDATED_TIME_FIELD_NAME)
            expected_updated_time = getattr(expected, UPDATED_TIME_FIELD_NAME)
            setattr(expected, UPDATED_TIME_FIELD_NAME, result_updated_time)
            is_equal = result == expected
            setattr(expected, UPDATED_TIME_FIELD_NAME, expected_updated_time)
        else:
            is_equal = result == expected

        if not is_equal:
            diffs = get_differing_attributes(result, expected)
            logger.error(f"Differences found: {diffs}")
        return is_equal

    def _duplication_test_helper(self, field_names: list[str]):
        """Provided for subclasses to enable testing for field value duplication in items to be stored against items in storage.
        Args:
            field_name (str): name of item fields, which together, must be unique to check for existence already stored in the db.
        """
        assert (
            len(field_names) > 0
        ), "This helper test method must be called with 1 or 2 field names"
        assert (
            len(field_names) <= 2
        ), "This helper test method only supports 1 or 2 field names"
        item1 = self._get_test_item(1)
        item2a = self._get_test_item(2)
        item2b = None if len(field_names) == 1 else self._get_test_item(2)
        first = True
        for field_name in field_names:
            item1_value = getattr(item1, field_name, None)
            assert (
                item1_value is not None
            ), f"Item does not appear to have the expected field {field_name}"
            setattr(item2a, field_name, item1_value)
            if item2b and not first:
                setattr(
                    item2b, field_name, item1_value
                )  # Make all but the first field the same.  We should be able to add this one.
            first = False
        storage = self._get_tested_storage()
        storage.add(item1)  # diff uuids, same uri.
        # Make sure add is actually working
        assert storage.get_by_uuid(item1.uuid) is not None, "Basic item add failed"

        # Test add item with all duplcating fields the same
        try:
            storage.add(item2a)
            assert False, f"Was allowed to add the same field {field_names} twice"
        except Exception as exc:
            logger.info(
                f"The test may have logged an expected exception above for the addition of a duplicate item having common values for {field_names} field(s) "
            )
            assert (
                storage.get_by_uuid(item2a.uuid) is None
            ), "Found item with duplicate value in storage"

        # Test update/upsert (new item)
        try:
            storage.update(item2a)
            assert False, f"Was allowed to upsert the same field {field_names} twice"
        except Exception as exc:
            logger.info(
                f"The test may have logged an expected exception above for the upsert of a duplicate item having common values for {field_names} field(s) "
            )
            assert (
                storage.get_by_uuid(item2a.uuid) is None
            ), "Found item with duplicate value in storage"

        # Test adding an item that only as N-1 of the fields the same (when there are more than 1 field provided)
        if item2b:
            try:
                storage.add(item2b)
            except Exception as exc:
                assert (
                    False
                ), f"Was NOT allowed to add the item with N-1 fields the same"

            # Test update/upsert (new item)
            try:
                storage.update(item2b)
            except Exception as exc:
                assert (
                    False
                ), f"Was NOT allowed to upsert the item with N-1 fields the same"
            # Clean up storage
            storage.delete(item2b.uuid)
            assert storage.get_by_uuid(item2b.uuid) is None

        # Leave the storage the same as it was when we entered,
        # so we can call this method more than once from the same test.
        storage.delete(item1.uuid)
        assert storage.get_by_uuid(item1.uuid) is None

    def test_pagination(self):
        storage = self._get_tested_storage()

        # Add pages of items
        size = 2
        pages = 3
        total = size * pages
        items = []
        for i in range(total):
            item = self._get_test_item(i)
            storage.add(item)
            items.append(item)

        for page in range(pages):
            paginate = Pagination(index=page, size=size)
            results = storage.get_by_where(
                where=None, query_control=QueryControl(pagination=paginate)
            )
            begin_index = page * size
            end_index = begin_index + size  # Exclusive
            expected = items[begin_index:end_index]
            self._verify_get_results(expected, results)

        # Query beyond the last page
        paginate = Pagination(index=pages, size=size)
        results = storage.get_by_where(
            where=None, query_control=QueryControl(pagination=paginate)
        )
        self._verify_get_results([], results)

    def test_sorting(self):
        # Get the items to use from the sub-class
        sort_column, items = self._get_ascending_sorted_test_items(3)
        items_to_add = []

        # Insert the items in an unsorted order.
        for index in [0, 2, 1]:
            items_to_add.append(items[index])
        storage = self._get_tested_storage()
        storage.add(items_to_add)

        # search via ascending, with trivial pagination
        pagination = Pagination(index=0, size=len(items))
        so = SortOrder(column=sort_column, ascending=True)
        query_control = QueryControl(sort_orders=[so], pagination=pagination)
        results = storage.get_by_where(where=None, query_control=query_control)
        self._verify_get_results(items, results)

        # search via descending, with no pagination
        items.reverse()
        so = SortOrder(column=sort_column, ascending=False)
        query_control = QueryControl(sort_orders=[so])
        results = storage.get_by_where(where=None, query_control=query_control)
        self._verify_get_results(items, results)

    def test_count(self):
        """Test the count() method returns the correct number of items in storage."""
        storage = self._get_tested_storage()

        # Count on empty/non-existent table should return 0
        assert storage.count() == 0, "Expected count of 0 on empty storage"

        # Add items and verify count increases
        item0 = self._get_test_item(0)
        storage.add(item0)
        assert storage.count() == 1, "Expected count of 1 after adding first item"

        item1 = self._get_test_item(1)
        storage.add(item1)
        assert storage.count() == 2, "Expected count of 2 after adding second item"

        # Add multiple items at once
        item2 = self._get_test_item(2)
        item3 = self._get_test_item(3)
        storage.add([item2, item3])
        assert storage.count() == 4, "Expected count of 4 after adding two more items"

        # Delete an item and verify count decreases
        storage.delete(item0.uuid)
        assert storage.count() == 3, "Expected count of 3 after deleting one item"

        # Delete multiple items
        storage.delete([item1.uuid, item2.uuid])
        assert storage.count() == 1, "Expected count of 1 after deleting two items"

    def test_count_with_where(self):
        """Test the count() method with a where parameter."""
        storage = self._get_tested_storage()

        # Add items
        item0 = self._get_test_item(0)
        item1 = self._get_test_item(1)
        item2 = self._get_test_item(2)
        storage.add([item0, item1, item2])

        # Count all should return 3
        assert storage.count() == 3, "Expected count of 3 for all items"

        # Count with where clause matching one item by uuid
        where = self.__get_where_dict(storage, item0)
        count_filtered = storage.count(where=where)
        assert count_filtered == 1, "Expected count of 1 when filtering by uuid"

        # Count with where clause matching no items
        item3 = self._get_test_item(3)
        where = self.__get_where_dict(storage, item3)
        count_none = storage.count(where=where)
        assert count_none == 0, "Expected count of 0 when no items match"

    def test_update_fields(self):
        storage = self._get_tested_storage()

        # Add 1 item
        item0 = self._get_test_item(0)
        storage.add(item0)
        item0_attributes = vars(item0)
        matches = {}
        for name, value in item0_attributes.items():
            if name not in [
                UPDATED_TIME_FIELD_NAME,
                CREATED_TIME_FIELD_NAME,
                UUID_COLUMN_NAME,
            ] and not name.startswith("_"):
                matches[name] = value

        # Create a 2nd item and use it to update the stored item's attributes to match
        item1 = self._get_test_item(1)
        item1_attributes = vars(item1)
        updates = {}
        for name, value in item1_attributes.items():
            if name not in [
                UPDATED_TIME_FIELD_NAME,
                CREATED_TIME_FIELD_NAME,
                UUID_COLUMN_NAME,
            ] and not name.startswith("_"):
                updates[name] = value

        # Update the stored item's attributes to match item1, but with should_update returning False
        updated_item0 = storage.update_fields(
            item0.uuid, updates, should_update=lambda item: False
        )
        assert updated_item0 == None, "The update did not fail"

        # Update the stored item's attributes to match item1, with should_update checking attributes match
        def check_matches(item):
            for name, value in matches.items():
                if getattr(item, name) != value:
                    return False
            return True

        updated_item0 = storage.update_fields(
            item0.uuid, updates, should_update=check_matches
        )
        assert item0 != updated_item0, "Updates do not appear to have taken effect"

        # Make sure the stored item got updated with item1 public attributes.
        updated_item0_attributes = vars(updated_item0)
        for name, value in updates.items():
            assert (
                updated_item0_attributes[name] == value
            ), "Updated attribute {name} did not get assigned value {value}"

        item = storage.get_by_uuid(item0.uuid)
        assert (
            item == updated_item0
        ), "Reading item from db did not give the updated item"

        # Make sure we can update protected UUID column
        try:
            item = self._get_test_item(2)
            storage.update_fields(item0.uuid, {UUID_COLUMN_NAME: item.uuid})
            assert False, "Was allowed to update the UUID"
        except Exception:
            pass

        # Make sure we can update protected JSON column
        try:
            item = self._get_test_item(3)
            storage.update_fields(item0.uuid, {JSON_COLUMN_NAME: '{ "a": 1 }'})
            assert False, "Was allowed to update the JSON column"
        except Exception:
            pass


class AbstractExistingDataReadTest(AbstractReadonlySingletonStorageUsingTest):

    def test_reading_existing_content(self):
        singleton_storage.set_storage_prefix(None)
        storage = singleton_storage.get_admin_storage()
        storage: BaseItemStorage = self._get_tested_readonly_storage(storage)
        try:
            items = storage.get_by_uuid(None)
            for item in items:
                self._validate_item(item)
        except Exception as exc:
            assert False, f"Got exception reading legacy data {exc}"

    def _get_tested_readonly_storage(
        self, storage: singleton_storage.SingletonAdminStorage
    ):
        raise ValueError("Subclass must implement this method")

    def _validate_item(self, item: BaseStoredItem):
        """Allow the sub-class to further validate the item.

        Args:
            item (BaseStoredItem): item read from storage returned by  _get_tested_readonly_storage()
        """
        pass
