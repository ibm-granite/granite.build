"""Storage module."""

import datetime
import json
import re
import threading
from abc import abstractmethod
from typing import (
    Annotated,
    Any,
    Callable,
    Generic,
    Iterator,
    List,
    Optional,
    Self,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from gbserver.types.constants import getenv_boolean
from gbserver.utils.logger import LoggingUtility, get_logger
from gbserver.utils.utils import get_utc_time, get_uuid

UUID_FIELD_NAME = "uuid"
UUID_COLUMN_NAME = UUID_FIELD_NAME
JSON_COLUMN_NAME = "json"
# NCOLUMNS_ADDED_INTERNALLY = 2   # UUID and JSON columns


CREATED_TIME_FIELD_NAME = "created_time"
"""If the item has this field it will be managed in the update() method to always keep the initial value"""
UPDATED_TIME_FIELD_NAME = "updated_time"
"""If the item has this field it will be managed in the update() method to always update the stored value"""
BEGINNING_OF_TIME = datetime.datetime(2025, 1, 1, 12)
"""This is a somewhat arbitrary date, but before items started containing created_time and updated_time fields"""

logger = get_logger(__name__)


class SortOrder(BaseModel):
    """Specifies a single column ordering"""

    column: str
    ascending: bool = True

    @staticmethod
    def parse(spec: str) -> "SortOrder":
        """Parse a string of the form column[:(asc|desc)] to create the implied SortOrder object.
        If order is not specified, ascending is the default.

        Args:
            spec (str): _description_

        Returns:
            SortOrder: _description_

        Raises:
            ValueError: if the spec is malformed.
        """
        if not ":" in spec:
            return SortOrder(column=spec, ascending=True)
        split_spec = spec.split(":")
        if len(split_spec) != 2:
            raise ValueError(f"Saw too many ':' in sort order specification {spec}")
        column = split_spec[0]
        order = split_spec[1]
        if "asc" in order:
            so = SortOrder(column=column, ascending=True)
        elif "desc" in order:
            so = SortOrder(column=column, ascending=False)
        else:
            raise ValueError(
                f"Order did not contain 'asc' or 'desc' in sort order specification {spec}"
            )
        return so


class Pagination(BaseModel):
    """Specifies the pagination of a where query"""

    index: Annotated[int, Field(ge=0)]
    size: Annotated[int, Field(gt=0)]


class QueryControl(BaseModel):
    """Specifies the pagination and ordeing of query results."""

    pagination: Optional[Pagination] = None
    sort_orders: Optional[list[SortOrder]] = None


class TaggedItem(BaseModel):
    """Tagged Item implementation."""

    model_config = ConfigDict(validate_assignment=True)

    tags: Optional[list[str]] = Field(default_factory=list)

    @field_validator("tags")
    def validate_tags(cls, v: list):
        """Verify tags."""
        if v is None:
            return []
        for item in v:
            if re.search(r"\s", item):
                raise ValueError(f"Tag value '{item}' contains whitespace.")
            elif (
                "," in item
            ):  # Because we use a comma as a value separator in the tags column of SQLArtifactRegistry
                raise ValueError(f"Tag value '{item}' contains a comma.")
        return v


class BaseStoredItem(BaseModel):
    """
    Base class for all/most objects used in the llm.build, but especially objects placed in storage.

    Args:
        BaseModel (_type_): pydantic infrastructure class
    """

    uuid: str = Field(default_factory=get_uuid)


BASE_ITEM_TYPE = TypeVar("BASE_ITEM_TYPE", bound=BaseStoredItem)


class IItemStorage(BaseModel, Generic[BASE_ITEM_TYPE]):
    """
    This serves as an "interface" class for all storage implementations operating on BASE_ITEM_TYPE.
    """

    @abstractmethod
    def get_table_name(self) -> str:
        """Get the table name."""
        raise ValueError("Must be implemented by sub-class")

    @abstractmethod
    def _get_column_values(self, item: BASE_ITEM_TYPE) -> dict[str, BASE_ITEM_TYPE]:
        """
        WARNING: This (for now) defined as part of the interface since one of the tests on get_by_where() needs it.

        Used to retrieve the values that should be used to define the schema and values stored in the associated table.
        UUID_COLUMN_NAME and JSON_COLUMN_NAME is always expected and handled independently and so should not be included here.
        Sub-classes should override to control which fields of the object are exposed in the table views and
        on which searches can be made.

        Returns:
            dict[str,Any]: key is column name, value is value stored in the row for that column.
            Values are generallly expected to be primitives, but that is not a requirement.
            The set of returned keys should always be the same for a given sub-class.
        """
        raise NotImplementedError(
            f"Sub-class {self.__class__.__name__} did not implement method throwing this exception"
        )

    @abstractmethod
    def add(self, items: Union[BASE_ITEM_TYPE, list[BASE_ITEM_TYPE]]) -> Union[str, list[str]]:
        """
        Add the given object(s) to the underlying storage instance.
        Checks for item field uniquess, if  configured in the initializer.
        If any of the items have a uuid which is already in the table, then an exception is raise.


        Args:
            items (Union[BASE_ITEM_TYPE,list[BASE_ITEM_TYPE]]): BaseSBASE_ITEM_TYPEtoredItem object or list of objects to store.

        Returns:
            Union[str,list[str]]: a uuid or list of uuids under which the object(s) was(were) stored.

        Raises:
            ValueError: if any of the items' uuids already exist in the database

        """
        raise NotImplementedError(
            f"Sub-class {self.__class__.__name__} did not implement method throwing this exception"
        )

    @abstractmethod
    def get_by_uuid(
        self, uuids: Optional[Union[str, list[str]]]
    ) -> Union[Optional[BASE_ITEM_TYPE], list[BASE_ITEM_TYPE]]:
        """Get zero or more items corresponding to one or more uuids.

        Args:
            uuids (Union[str,list[str]]): list of uuids to search for. If None, then get all.

        Returns:
            Union[BASE_ITEM_TYPE, list[BASE_ITEM_TYPE]]: If a single uuid is provided, then the matching item is returned or None if not found.
            if a list of uuids is provided, then a list of items of the same length is returned with None in place in the list where the item
            for the corresponding uuid in the input was not found.
            If uuids is None, then a list is always returned, even if empty.
        """
        raise NotImplementedError(
            f"Sub-class {self.__class__.__name__} did not implement method throwing this exception"
        )

    @abstractmethod
    def get_by_where(
        self,
        where: Optional[Union[str, dict]] = None,
        query_control: Optional[QueryControl] = None,
    ) -> list[BASE_ITEM_TYPE]:
        """
        Search for items via column values.
        The column values that are stored, and are therefore queryable,
        are defined by the sub-class implementation of get_column_values(item)

        Args:
            where: if None, then get all.
            where(str): SQL WHERE clause (w/o the WHERE).
            where(dict): a dicitonary of columns names mapped to column values that will be used to build the WHERE clause by
            ANDing all the column=value expressions. If the key dictionary value is a list/set/tuple then match any one of the items
            in the list.
            query_control[QueryControl]: If provide specifies keys 'index' and 'size' controlling the zer-based page index and the number of rows in a page.
            if specified, the ordering of the list should be such that the most recently added rows are in the first page.

        Returns:
            list[BASE_ITEM_TYPE]: list of matching items if any found, otherwise and empty list.  Ordering of this list is undefined.
        """
        raise NotImplementedError(
            f"Sub-class {self.__class__.__name__} did not implement method throwing this exception"
        )

    @abstractmethod
    def delete_table(self) -> None:
        """
        Remove the table storing the items.
        A noop if the table does not exist.
        """
        raise NotImplementedError(
            f"Sub-class {self.__class__.__name__} did not implement method throwing this exception"
        )

    @abstractmethod
    def delete(self, uuids: Union[str, list[str]]) -> None:
        """
        Delete the stored object/record having the given uuid(s).

        Args:
            uuid (str): identifier under which an object is stored.  Returned by add().
        """
        raise NotImplementedError(
            f"Sub-class {self.__class__.__name__} did not implement method throwing this exception"
        )

    @abstractmethod
    def update(
        self,
        item: BASE_ITEM_TYPE,
        update_updated_time: bool = True,
        create_if_not_exist: bool = True,
    ) -> None:
        """Update/insert the item with the given UUID, with special handling for updated_time and created_time fields.
        If created_time field exists:
            1) If an item is already stored under the uuid, then preserve that time in the stored record, and
            2) Set the previously stored created_time into the given item (its modified on return).
        If updated_time field exists:
            1) update the field with the current time, and
            2) Set this new update time into the given item (its modified on return).

        Args:
            item (BASE_ITEM_TYPE): Item to replace under the given uuid.
            update_updated_time: if true and the item has an updated_time field, then update the field.
            created_if_not_exist:  if true then upsert, otherwise verify the item is not present and add()

        Raises:
            ValueError: if create_if_not_exist=False and the uuid used is NOT in the database already.

        """
        raise NotImplementedError(
            f"Sub-class {self.__class__.__name__} did not implement method throwing this exception"
        )

    @abstractmethod
    def update_fields(
        self,
        uuid: str,
        fields: dict[str, Any],
        should_update: Optional[Callable[[BASE_ITEM_TYPE], bool]] = None,
        update_updated_time: bool = True,
    ) -> Optional[BASE_ITEM_TYPE]:
        """Update the given fields of the item stored under the given item uuid.
        The implementation SHOULD do this atomically.

        Args:
            uuid (str): id of item
            fields (dict[str,Any]): dictionary of field names and values to be replaced in the referenced item.
            should_update (Optional[Callable[[BASE_ITEM_TYPE], bool]]): If provided, a function that takes the
                current stored item and returns True if the update should proceed. If the function returns False,
                the update is NOT performed and None is returned. This check SHOULD be atomic with the update
                operation to prevent race conditions.
            update_updated_time (bool, optional): whether to update the updated_time field. Defaults to True.

        Raises:
            ValueError: if fields are not present on the items stored in this instance.

        Returns:
            Optional[BASE_ITEM_TYPE]: The updated item if the update was performed, or None if
            should_update was provided and returned False.
        """
        raise NotImplementedError(
            f"Sub-class {self.__class__.__name__} did not implement method throwing this exception"
        )

    @abstractmethod
    def get_column_names(self) -> list[str]:
        """Get the names of the searchable columns in the table. If the table has not been created yet, return an empty list"""
        raise NotImplementedError(
            f"Sub-class {self.__class__.__name__} did not implement method throwing this exception"
        )

    @abstractmethod
    def count(self, where: Optional[Union[str, dict]] = None) -> int:
        """Return the number of items in storage matching the where clause.

        Args:
            where: if None, count all items.
            where(str): SQL WHERE clause (w/o the WHERE).
            where(dict): dictionary of column names mapped to values for WHERE clause.

        Returns:
            int: the count of matching items.
        """
        raise NotImplementedError(
            f"Sub-class {self.__class__.__name__} did not implement method throwing this exception"
        )


class BaseItemStorage(IItemStorage[BASE_ITEM_TYPE], Generic[BASE_ITEM_TYPE]):
    """
    Provides CRUD capabilities over pydantic BASE_ITEM_TYPE objects in underlying Lakehouse storage.
    A given instance of this class is intended to be used with only one class of BASE_ITEM_TYPE
    (e.g., Space, Artifact, etc).
    """

    table_name: str
    item_class: Type[BASE_ITEM_TYPE]
    # Item fields that should be unique within the items placed in storage.
    # These are enforced via get_by_where() calls so these names should match the searchable column names.
    # This was originally provided for LH which did not support uniquess checks in the columns.
    # For performance reasons, SQL or other storage should probably use another mechanism in the DB to enforce uniqueness.
    unique_fields: list[str] = []
    logger: Any = None  # set in the initializer
    # If the storage schema adjustment/initialization has been done
    __is_storage_initialized: bool = False
    __storage_modification_lock: Any
    """ 
    optional path to a yaml file holding lakehouse configuration.  If not specified, both the
    environment and lh-conf.yaml will be searched, in that order, for configuration.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = LoggingUtility(logger, msg_prefix=f"{self.table_name}")
        self.__storage_modification_lock = (
            threading.RLock()
        )  # for multiple lock calls from the same thread.

    @classmethod
    @abstractmethod
    def _get_sample_item(cls) -> BASE_ITEM_TYPE:
        """Generates a representative item with all fields set.
        This is used by the BaseItemStorage class to allow determination of the storage schema, as needed.
        The item will never be inserted to the storage instance.

        Raises:
            NotImplemented: _description_
        """
        raise NotImplementedError(
            f"Sub-class {cls.__name__} did not implement method throwing this exception"
        )

    def get_table_name(self) -> str:
        """Get the table name."""
        return self.table_name

    def _get_column_values(self, item: BASE_ITEM_TYPE) -> dict[str, Any]:
        """
        Used to retrieve the values that should be used to define the schema and values stored in the associated table.
        UUID_COLUMN_NAME and JSON_COLUMN_NAME is always expected and handled independently and so should not be included here.
        Sub-classes should override to control which fields of the object are exposed in the table views and
        on which searches can be made.

        Returns:
            dict[str,Any]: key is column name, value is value stored in the row for that column.
            Values are generallly expected to be primitives, but that is not a requirement.
            The set of returned keys should always be the same for a given sub-class.
        """
        raise ValueError("Must be defined by the sub-class")

    def add(self, items: Union[BASE_ITEM_TYPE, list[BASE_ITEM_TYPE]]) -> Union[str, list[str]]:
        """
        Add the given object(s) to the underlying storage instance.
        Checks for item field uniquess, if  configured in the initializer.
        If any of the items have a uuid which is already in the table, then an exception is raise.


        Args:
            items (Union[BASE_ITEM_TYPE,list[BASE_ITEM_TYPE]]): BASE_ITEM_TYPE object or list of objects to store.

        Returns:
            Union[str,list[str]]: a uuid or list of uuids under which the object(s) was(were) stored.

        Raises:
            ValueError: if any of the items' uuids already exist in the database

        """
        assert items != None
        if not isinstance(items, list):
            items = [items]
        self.__validate_items(items)

        self.logger.info(f"Begin adding {len(items)} item(s)")
        self.__initialize_storage()
        self.__enforce_uniqueness(items)
        self.__add_items(items)

        uuids = [x.uuid for x in items]
        if len(uuids) == 1:
            uuids = uuids[0]  # When only adding a single item, don't return a list.
        self.logger.info(f"Done adding {len(items)} item(s)")
        return uuids

    def __initialize_storage(self):
        """Use the sample item provided by self.item_class, convert it  to dict and then call the sub-class self._create_or_adjust_schema_item_dict()
        We try and only call self._create_or_adjust_schema_item_dict() once per instance on the add/get/update/delete methods.

        Args:
            item (BASE_ITEM_TYPE): _description_
        """
        if self.__is_storage_initialized:
            return
        with self.__storage_modification_lock:
            if not self.__is_storage_initialized:
                self.logger.info("Begin initializing storage")
                item = self._get_sample_item()
                item_dict = self._convert_item_to_row_dict(item)
                self._create_or_adjust_schema_item_dict(item_dict)
                self.__is_storage_initialized = True
                self.logger.info("Done initializing storage")
        return

    def _create_or_adjust_schema_item_dict(self, item: dict[str, Any]):
        """
        Create the table to match the given item and schema as defined by the columns/values contained in the given dictionary.
        """
        raise NotImplementedError(
            f"Sub-class {self.__class__.__name__} did not implement method throwing this exception"
        )

    def __add_items(self, items: list[BASE_ITEM_TYPE]):
        """Called from add() after item validation and schema alignment to add the given list of 1 or more items to the database.

        Args:
            items (list[BASE_ITEM_TYPE]): _description_
        """
        item_dicts = []
        for item in items:
            has_created_field = hasattr(item, CREATED_TIME_FIELD_NAME)
            has_updated_field = hasattr(item, UPDATED_TIME_FIELD_NAME)
            if has_created_field or has_updated_field:
                # Note: This is a side-effect in that it updates the times of the items passed in.
                t = get_utc_time()
                if has_created_field:
                    setattr(item, CREATED_TIME_FIELD_NAME, t)
                if has_updated_field:
                    setattr(item, UPDATED_TIME_FIELD_NAME, t)
            d = self._convert_item_to_row_dict(item)
            item_dicts.append(d)
        self._add_item_dicts(item_dicts)

    def _add_item_dicts(self, items: list[dict[str, Any]]):
        """Called from add() after item validation and schema alignment to add the given list of 1 or more items as dictionaries to the database.

        Args:
            items (list[dict[str,Any]]): a list of BASE_ITEM_TYPE converted to dictionaries.  Each dictionary includes
            the UUID_COLUMN_NAME, JSON_COLUMN_NAME and any other keys/values as defined by the sub-classes' _get_column_values(item) method.

        Raises:
            ValueError:
        """
        raise NotImplementedError(
            f"Sub-class {self.__class__.__name__} did not implement method throwing this exception"
        )

    def __validate_items(self, items: Union[BASE_ITEM_TYPE, list[BASE_ITEM_TYPE]]) -> None:
        if not isinstance(items, list):
            items = [items]
        else:
            assert len(items) > 0, "List of items is empty"
        for item in items:
            if not isinstance(item, self.item_class):
                raise ValueError(f"Item is not the expected type {self.item_class.__name__}")
            if item.uuid is None or item.uuid == "":
                raise ValueError(f"Item does not have a uuid: {item}")

        pass

    def __do_item_uuids_exist(
        self, items: Union[BASE_ITEM_TYPE, list[BASE_ITEM_TYPE]]
    ) -> list[str]:
        """See if any of the items' uuids already exist in the table.

        Args:
            items (Union[BASE_ITEM_TYPE,list[BASE_ITEM_TYPE]]): _description_

        Returns:
            list[str]: list of uuids that are already in the table.  Empty list if none exist.
        """
        uuids = []
        if not isinstance(items, list):
            items = [items]
        for item in items:
            uuids.append(item.uuid)
        return self.__do_uuids_exist(uuids)

    def __do_uuids_exist(self, uuids: Union[str, list[str]]) -> list[str]:
        """Check if any of the given uuids are already in the table and return a list of the already-existing ones.

        Args:
            uuids (Union[str,list[str]]): list of UUIDs to check.

        Returns:
            list[str]: list of uuids that are already in the table.  Empty list if none exist.
        """
        existing = []
        if isinstance(uuids, str):
            uuids = [uuids]
        for uuid in uuids:
            items = self.get_by_uuid(uuid)
            if items is not None:  # or len(items) > 0:
                existing.append(uuid)
        return existing

    def get_by_where(
        self,
        where: Optional[Union[str, dict]] = None,
        query_control: Optional[QueryControl] = None,
    ) -> list[BASE_ITEM_TYPE]:
        """
        Search for items via column values.
        The column values that are stored, and are therefore queryable,
        are defined by the sub-class implementation of get_column_values(item)

        Args:
            where: if None, then get all.
            where(str): SQL WHERE clause (w/o the WHERE).
            where(dict): a dicitonary of columns names mapped to column values that will be used to build the WHERE clause by
            ANDing all the column=value expressions.
            query_control[QueryControl]: If provide specifies keys 'index' and 'size' controlling the zer-based page index and the number of rows in a page.
            if specified, the ordering of the list should be such that the most recently added rows are in the first page.

        Returns:
            list[BASE_ITEM_TYPE]: list of matching items if any found, otherwise an empty list.
        """
        if not self.__is_storage_initialized and not self._does_table_exist():
            # This avoid creating/initializing the table, which is useful for buildrunner tests that time out.
            return []

        if where is not None and not isinstance(where, str) and not isinstance(where, dict):
            raise ValueError("where must be a string or dict")
        self.__initialize_storage()
        item_dict = self.__get_by_where(where=where, query_control=query_control)
        return list(item_dict.values())

    def get_paged(
        self, where: Optional[Union[str, dict]] = None, page_size: int = 200
    ) -> Iterator[list[BASE_ITEM_TYPE]]:
        """Yield pages of items matching the where clause.

        Each iteration yields one page (a list of up to page_size items).
        Callers can filter or aggregate incrementally without loading all
        matching items into memory at once.

        Args:
            where: Column filter (same semantics as get_by_where).
            page_size: Maximum items per page.

        Yields:
            list[BASE_ITEM_TYPE]: One page of matching items.
        """
        page_index = 0
        while True:
            qc = QueryControl(pagination=Pagination(index=page_index, size=page_size))
            page = self.get_by_where(where, query_control=qc)
            if not page:
                break
            yield page
            if len(page) < page_size:
                break
            page_index += 1

    def __get_by_where(
        self,
        where: Optional[Union[str, dict]] = None,
        query_control: Optional[QueryControl] = None,
    ) -> dict[str, BASE_ITEM_TYPE]:
        items = {}
        self.logger.debug(f"Begin searching by where={where}")
        row_dicts = self._get_by_where_row_dicts(where=where, query_control=query_control)
        for row_dict in row_dicts:
            item = self._convert_row_dict_to_item(row_dict)
            items[item.uuid] = item
        self.logger.debug(f"Done searching by where={where}. found {len(items)} items.")
        return items

    @abstractmethod
    def _get_by_where_row_dicts(
        self,
        where: Optional[Union[str, dict]] = None,
        query_control: Optional[QueryControl] = None,
    ) -> list[dict[str, Any]]:
        """Called from get_by_were()
        Search for items via column values.
        The column values that are stored, and are therefore queryable,
        are defined by the sub-class implementation of _get_column_values(item)

        Args:
            where: if None, then get all.
            where(str): SQL WHERE clause (w/o the WHERE).
            where(dict): a dicitonary of columns names mapped to column values that will be used to build the WHERE clause by
            ANDing all the column=value expressions.
            query_control[Pagination]: If provide specifies keys 'index' and 'size' controlling the zer-based page index and the number of rows in a page.
            if specified, the ordering of the list should be such that the most recently added rows are in the first page.

        Returns:
            list[dict[str,Any]]: list of matching dictionary row representations,  if any found, otherwise an empty list.
            Ordering of this list is undefined. dictionaries should be the same as the dictionaries received by
            _add_item_dict().
        """
        raise NotImplementedError(
            f"Sub-class {self.__class__.__name__} did not implement method throwing this exception"
        )

    def get_by_uuid(
        self, uuids: Optional[Union[str, list[str]]]
    ) -> Union[Optional[BASE_ITEM_TYPE], list[BASE_ITEM_TYPE]]:
        """Get zero or more items corresponding to one or more uuids.

        Args:
            uuids (Union[str,list[str]]): list of uuids to search for. If None, then get all.

        Returns:
            Union[BASE_ITEM_TYPE, list[BASE_ITEM_TYPE]]: If a single uuid is provided, then the matching item is returned or None if not found.
            if a list of uuids is provided, then a list of items of the same length is returned with None in place in the list where the item
            for the corresponding uuid in the input was not found.
            If uuids is None, then a list is always returned, even if empty.
        """
        self.logger.debug(f"Begin searching for items by uuid with uuids={uuids}")
        uuids_was_list = isinstance(uuids, list)

        if not self.__is_storage_initialized and not self._does_table_exist():
            # This avoids creating/initializing the table, which is useful for buildrunner tests that time out.
            return [] if uuids_was_list else [] if uuids is None else None

        if uuids is None or isinstance(uuids, str) or isinstance(uuids, list):
            if isinstance(uuids, str):
                row_filter = {UUID_COLUMN_NAME: uuids}
                uuids = [uuids]  # Below expects a list if not None
            elif isinstance(uuids, list):
                row_filter = {UUID_COLUMN_NAME: uuids}
            else:
                row_filter = {}
            items = self.get_by_where(row_filter)
        else:
            where = f"{UUID_COLUMN_NAME} in {uuids}"
            where = where.replace("[", "(")
            where = where.replace("]", ")")
            items = self.get_by_where(where)  # where is a list of uuids

        if uuids is None:
            new_items = items
        else:
            items_by_uuid = {}
            for item in items:
                items_by_uuid[item.uuid] = item
            new_items = []
            for uuid in uuids:
                item = items_by_uuid.get(uuid)
                new_items.append(item)  # Puts None in place where a uuid/item was not found.
            if len(uuids) == 1 and not uuids_was_list:
                new_items = new_items[0]

        new_items_list = new_items if isinstance(new_items, list) else [new_items]
        num_found = len(new_items_list) - new_items_list.count(None)
        self.logger.debug(f"Done searching for items by uuid, found {num_found} matching items")

        return new_items

    def _does_table_exist(self: Self) -> bool:
        """Determine if the table exists already. May be called before _create_or_adjust_schema_item_dict()."""
        raise NotImplementedError(
            f"Sub-class {self.__class__.__name__} did not implement method throwing this exception"
        )

    def delete_table(self) -> None:
        """
        Remove the table storing the items.
        A noop if the table does not exist.
        """
        self.logger.debug(f"Begin deleting table")
        if (
            not self._does_table_exist()
        ):  # Try and avoid needlessly creating the table only to delete it, seen in setup/teardown of tests.
            self.logger.debug(f"Done deleting table: table does not exist.")
            return

        with self.__storage_modification_lock:
            # self.__initialize_storage()  # Make sure the table exists to easy condition checking in the sub-class
            self._delete_table()
            self.__is_storage_initialized = False
        self.logger.debug(f"Done deleting table")

    def _delete_table(self):
        """Called by delete_table()"""
        raise NotImplementedError(
            f"Sub-class {self.__class__.__name__} did not implement method throwing this exception"
        )

    def update(
        self,
        item: BASE_ITEM_TYPE,
        update_updated_time: bool = True,
        create_if_not_exist: bool = True,
    ) -> None:
        """Update/insert the item with the given UUID, with special handling for updated_time and created_time fields.
        If created_time field exists:
            1) If an item is already stored under the uuid, then preserve that time in the stored record, and
            2) Set the previously stored created_time into the given item (its modified on return).
        If updated_time field exists:
            1) update the field with the current time, and
            2) Set this new update time into the given item (its modified on return).

        Args:
            item (BASE_ITEM_TYPE): Item to replace under the given uuid.
            update_updated_time: if true and the item has an updated_time field, then update the field.
            created_if_not_exist:  if true then upsert, otherwise verify the item is not present and add()

        Raises:
            ValueError: if create_if_not_exist=False and the uuid used is NOT in the database already.

        """
        uuid = item.uuid
        self.logger.info(f"Begin updating item with uuid {uuid}")
        self.__initialize_storage()

        # Get existing item if any
        pre_existing = self.get_by_uuid(uuid)

        if pre_existing is None:
            # Item doesn't exist
            if not create_if_not_exist:
                raise ValueError(f"Item with uuid {uuid} not found in table {self.table_name}")
            self.add(item)
        else:
            # Get all fields from the item and call update_fields()
            fields = vars(item).copy()  # Copy, otherwise the pops modify the input item.
            fields.pop(UUID_FIELD_NAME, None)  # Exclude since we never update that
            fields.pop(CREATED_TIME_FIELD_NAME, None)  # Exclude since we never update that
            fields.pop(
                UPDATED_TIME_FIELD_NAME, None
            )  # Exclude since we handle that in update_fields
            self.update_fields(
                uuid,
                fields,
                should_update=None,
                update_updated_time=update_updated_time,
            )

        self.logger.info(f"Done updating item with uuid {uuid}")

    def _convert_item_to_json_str(self, item: BASE_ITEM_TYPE) -> str:
        """Create a json string from the item being stored in the table.  This is the json column value.
        By default, this us a pydantic model_dump_json() call, but sub-classes may override as necessary.
        If overriding, you may also need to override the method _convert_json_str_to_item().

        Args:
            item (BASE_ITEM_TYPE): item from which to produce json.

        Returns:
            str: json formatted string that will be later passed to _convert_json_str_to_item() to recreate the item in a given row.
        """
        json_str = item.model_dump_json(round_trip=True)
        return json_str

    def _convert_json_str_to_item(self, json_str: str) -> BASE_ITEM_TYPE:
        """Convert the JSON column value to the underlying BaseStoreItem object.
        By default this is a simple pydantic validate_json() call (after also calling _prep_json_before_serialization()), but
        sub-classes may add additional), but sub-classes may override if they have  the requirement.
        This was originally motivated by the need to support gb_events table which has non-pydantic/non-primitive attributes (i.e. BuildEvent and
        sub-classes).
        If overriding, you may also need to override the method _convert_item_json_str().

        Args:
            json_item (str): The json formatted string from which to generate the items stored in this implementation.

        Returns:
            BASE_ITEM_TYPE: _description_
        """
        json_str = self._prep_json_before_deserialization(json_str)
        item = TypeAdapter(self.item_class).validate_json(json_str)
        return item

    def _convert_item_to_row_dict(self, item: BASE_ITEM_TYPE) -> dict[str, Any]:
        """Convert the item to a dictionary to be stored in storage.
        The dictionary includes keys
            1. UUID_COLUMN_NAME
            2. JSON_COLUMN_NAME
            3. Additional key/values as defined by the sub-class _get_column_values(item) method.

        Args:
            item (BASE_ITEM_TYPE): item to convert to dictionary to be stored.

        Returns:
            dict[str,Any]: converted dictionary
        """
        json = self._convert_item_to_json_str(item=item)
        item_dict = {}
        item_dict[UUID_COLUMN_NAME] = item.uuid
        item_dict[JSON_COLUMN_NAME] = json
        item_dict = item_dict | self._get_column_values(item)
        return item_dict

    def _prep_json_before_deserialization(self, json_item: str) -> str:
        """Provided to allow sub-classes to update/normalize the json after it is read as json and before it is deserialized back to BASE_ITEM_TYPE"""
        return json_item

    def __fill_missing_times(self, item: BASE_ITEM_TYPE, json_str: str):
        """If the named field is not present in the item dictionary, then set the item's field to the BEGINNING_OF_TIME.
        This is provided to override the new default values for created/updated_time fields that were given defaults
        AFTER having stored items without these fields.  These old items should have BEGINNING_OF_TIME as their time field values.

        Args:
            item (BASE_ITEM_TYPE): _description_
            json(str): _description_
        """
        json_dict = None
        for field_name in [CREATED_TIME_FIELD_NAME, UPDATED_TIME_FIELD_NAME]:
            has_field = hasattr(item, field_name)
            if has_field:
                json_dict = json_dict if json_dict is not None else json.loads(json_str)
                has_field = (
                    json_dict.get(field_name, None) is not None
                )  # Field was not in the original item
                if not has_field:
                    setattr(item, field_name, BEGINNING_OF_TIME)

    def _convert_row_dict_to_item(self, item_dict: dict[str, Any]) -> BASE_ITEM_TYPE:
        """Convert the given dictionary created by __to_item_dict() or retrived from storage back into the item
        of class self.item_class.  This deserializes the value in the JSON_COLUMN_NAME key.

        Args:
            item_dict (dict[str,Any]): _description_

        Returns:
            BASE_ITEM_TYPE: _description_
        """
        json_item = item_dict[JSON_COLUMN_NAME]  # json content of full item

        item = self._convert_json_str_to_item(json_item)
        self.__fill_missing_times(item, json_item)
        return item

    def delete(self, uuids: Union[str, list[str]]) -> None:
        """
        Delete the stored object/record having the given uuid(s).

        Args:
            uuid (str): identifier under which an object is stored.  Returned by add().
        """
        self.logger.info(f"Begin deleting item with uuid(s) {uuids}")
        self.__initialize_storage()
        if not isinstance(uuids, list):
            uuids = [uuids]
        self._delete(uuids)
        self.logger.info(f"Done deleting items with uuids {uuids}")

    def _delete(self, uuids: list[str]):
        """Delete the given ids.  Must handle if table has not been created (and ignore the request).
        Args:
            uuids (list[str]): _description_
        """
        raise NotImplementedError(
            f"Sub-class {self.__class__.__name__} did not implement method throwing this exception"
        )

    def get_column_names(self) -> list[str]:
        """Initialize the storage then call the sub-class's _get_column_names().

        Returns:
            list[str]: _description_
        """
        self.__initialize_storage()
        return self._get_column_names()

    def _get_column_names(self) -> list[str]:
        """Get the names of the searchable columns in the existing table"""
        raise NotImplementedError(
            f"Sub-class {self.__class__.__name__} did not implement method throwing this exception"
        )

    def __enforce_uniqueness(self, items: Union[BASE_ITEM_TYPE, list[BASE_ITEM_TYPE]]) -> None:
        """Enforce uniqueness of fields as defined by the sub-class, including uuid.
        LH had no in-db enforcement of uniqueness, so list of unique fields are set of LH storage sub-classes.
        SQL can enforce uniqueness in the DB, so generally these sub-classes have self.unique_fields=[] and thus disable this check.

        Args:
            items (Union[BASE_ITEM_TYPE,list[BASE_ITEM_TYPE]]): _description_

        Raises:
            ValueError: _description_
        """
        for field_name in self.unique_fields:
            if field_name == UUID_COLUMN_NAME:
                existing_uuids = self.__do_item_uuids_exist(items)
                if len(existing_uuids) > 0:
                    len_items = len(items) if isinstance(items, list) else 1
                    raise ValueError(
                        f"The following uuids are already present {existing_uuids}.  None of the {len_items} items added."
                    )
            else:
                self.__check_for_duplications(field_name, items)

    def __check_for_duplications(
        self, field_name: str, items: Union[BASE_ITEM_TYPE, list[BASE_ITEM_TYPE]]
    ) -> None:
        """Helper function to help sub-classes enforce uniqueness of field values stored in the db.

        Args:
            field_name (str): _description_
            items (Union[BASE_ITEM_TYPE,list[BASE_ITEM_TYPE]]): _description_

        Raises:
            ValueError: _description_
            ValueError: _description_
        """
        if isinstance(items, list):
            item_list = items
        else:
            item_list = [items]
        field_values = []
        for item in item_list:
            # Check that URI is unique
            field_value = getattr(item, field_name, None)
            if field_value is not None:
                where = {field_name: field_value}
                where_items = self.get_by_where(where)
                if field_value in field_values:
                    raise ValueError(
                        f"More than 1 item with field {field_name}={field_value} found in items to be stored."
                    )
                if len(where_items) > 0:
                    raise ValueError(
                        f"Item with field {field_name}={field_value} already in storage."
                    )
                field_values.append(field_value)

    @staticmethod
    def _get_only_one(items: List[BASE_ITEM_TYPE]) -> Optional[BASE_ITEM_TYPE]:
        """Helper function to get a single item or None from a list or throw exception if more than 1 items are present."""
        if len(items) == 0:
            return None
        if len(items) == 1:
            return items[0]
        raise ValueError(f"Expected either 0 or 1 items, actual: {len(items)}")

    def _get_by_single_field(
        self: Self, column_name: str, column_value: Any, allow_multiple: bool
    ) -> Union[List[BASE_ITEM_TYPE], Optional[BASE_ITEM_TYPE]]:
        """Helper function for sub-classes to expose simple field searches.

        Args:
            self (Self): _description_
            field_name (str): name of column to search on
            field_value (Any): value of column to search on.
            allow_multiple (bool): if true, then allow a list of items to be returned, otherwise consider a list to be an error.
        Raises:
            ValueError: if more than 1 item was found and all_multiple=False

        Returns:
            Union[list[BASE_ITEM_TYPE], Optional[BASE_ITEM_TYPE]]: A list (empty or otherwise) if allow_multiple is True. . Otherwise, None if no items were found or
            the single item if found.
        """
        row_filter = {column_name: column_value}
        items = self.get_by_where(row_filter)
        return items if allow_multiple else self._get_only_one(items=items)

    def count(self, where: Optional[Union[str, dict]] = None) -> int:
        """Return the number of items in storage matching the where clause.

        Args:
            where: if None, count all items.
            where(str): SQL WHERE clause (w/o the WHERE).
            where(dict): dictionary of column names mapped to values for WHERE clause.

        Returns:
            int: the count of matching items.
        """
        self.__initialize_storage()
        if not self._does_table_exist():
            return 0
        return self._count(where=where)

    def _count(self, where: Optional[Union[str, dict]] = None) -> int:
        """Subclass hook to count items. Called after verifying table exists."""
        raise NotImplementedError(
            f"Sub-class {self.__class__.__name__} did not implement method throwing this exception"
        )
