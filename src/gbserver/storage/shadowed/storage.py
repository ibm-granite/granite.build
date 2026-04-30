#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import traceback
from typing import Any, Optional, Type, Union

from gbserver.storage.storage import (
    BASE_ITEM_TYPE,
    BaseStoredItem,
    IItemStorage,
    QueryControl,
)
from gbserver.utils.logger import LoggingUtility, get_logger
from gbserver.utils.unwrap_errors import get_readable_error_message

logger = get_logger(__name__)


class BaseDualItemStorage(IItemStorage):
    """
    Provides support for using 2 storage instances and keeping them in sync and performing similar operations on both.
    1 is considered a primary and is used for the read results.
    """

    table_name: str
    primary_class: Optional[Type[IItemStorage]] = None
    secondary_class: Optional[Type[IItemStorage]] = None
    primary: Optional[IItemStorage] = None
    secondary: Optional[IItemStorage] = None
    logger: Any = None  # Set in constructor

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = LoggingUtility(logger, msg_prefix=f"{self.table_name}")

        if self.primary_class is not None:
            assert (
                self.primary == None
            ), "Can not specify both the primary storage instance and its class"
            self.primary = self.primary_class(**kwargs)

        if self.secondary_class is not None:
            assert (
                self.secondary == None
            ), "Can not specify both the secondary storage instance and its class"
            try:
                self.secondary = self.secondary_class(**kwargs)
            except Exception as e:
                # Gracefully ignore errors on the secondary
                stack_trace = traceback.format_exc()
                self.logger.error(
                    f"Secondary storage creation gave an exception on creation\n{stack_trace}"
                )  # Gracefully ignore errors on the secondary
                self.secondary = None

        assert self.primary is not None, "Primary storage or its class was not provided"

        if self.secondary is None:
            self.logger.warning(
                f"Secondary storage not provided/available.  Operating with primary only."
            )
        else:
            assert self.primary.get_table_name() == self.secondary.get_table_name()

    def add(self, items: Union[BaseStoredItem, list[BaseStoredItem]]) -> Union[str, list[str]]:
        """
        Add the given object(s) to the underlying storage instance.
        Checks for item field uniquess, if  configured in the initializer.
        If any of the items have a uuid which is already in the table, then an exception is raise.


        Args:
            items (Union[BaseStoredItem,list[BaseStoredItem]]): BaseStoredItem object or list of objects to store.

        Returns:
            Union[str,list[str]]: a uuid or list of uuids under which the object(s) was(were) stored.

        Raises:
            ValueError: if any of the items' uuids already exist in the database

        """
        r = self.primary.add(items)
        if self.secondary is not None:
            try:
                self.secondary.add(items)
            except Exception as e:
                err_stack = traceback.format_exc()
                body = get_readable_error_message(e=e, err_stack=err_stack)
                self.logger.error(
                    f"Secondary storage got exception after success on the primary. {body}"
                )
        return r

    def get_table_name(self) -> str:
        return self.primary.get_table_name()

    def _get_column_values(self, item: BaseStoredItem) -> dict[str, Any]:
        """Needed for test on get_by_where()"""
        p = self.primary._get_column_values(item)
        if self.secondary is not None:
            s = self.secondary._get_column_values(item)
            if p != s:
                self.logger.error(
                    f"Dictionaries of item values not the same for primary and secondary?"
                )
        return p

    def get_by_where(
        self,
        where: Optional[Union[str, dict]] = None,
        paginate: Optional[QueryControl] = None,
    ) -> list[BaseStoredItem]:
        """
        Search for items via column values.
        The column values that are stored, and are therefore queryable,
        are defined by the sub-class implementation of get_column_values(item)

        Args:
            where: if None, then get all.
            where(str): SQL WHERE clause (w/o the WHERE).
            where(dict): a dicitonary of columns names mapped to column values that will be used to build the WHERE clause by
            ANDing all the column=value expressions.

        Returns:
            list[BaseStoredItem]: list of matching items if any found, otherwise and empty list.  Ordering of this list is undefined.
        """
        r = self.primary.get_by_where(where=where, query_control=paginate)
        if self.secondary is not None:
            try:
                self.secondary.get_by_where(where=where, query_control=paginate)
            except Exception as e:
                err_stack = traceback.format_exc()
                body = get_readable_error_message(e=e, err_stack=err_stack)
                self.logger.error(
                    f"Secondary storage got exception after success on the primary. {body}"
                )
        return r

    def get_by_uuid(
        self, uuids: Optional[Union[str, list[str]]]
    ) -> Union[Optional[BASE_ITEM_TYPE], list[BASE_ITEM_TYPE]]:
        """Get zero or more items corresponding to one or more uuids.

        Args:
            uuids (Union[str,list[str]]): list of uuids to search for. If None, then get all.

        Returns:
            Union[Optional[BASE_ITEM_TYPE], list[BASE_ITEM_TYPE]]:: If a single uuid is provided, then the matching item is returned or None if not found.
            if a list of uuids is provided, then a list of items of the same length is returned with None in place in the list where the item
            for the corresponding uuid in the input was not found.
            If uuids is None, then a list is always returned, even if empty.
        """
        r = self.primary.get_by_uuid(uuids)
        if self.secondary is not None:
            try:
                self.secondary.get_by_uuid(uuids)
            except Exception as e:
                err_stack = traceback.format_exc()
                body = get_readable_error_message(e=e, err_stack=err_stack)
                self.logger.error(
                    f"Secondary storage got exception after success on the primary. {body}"
                )
        return r

    def delete_table(self) -> None:
        """
        Remove the table storing the items.
        A noop if the table does not exist.
        """
        self.primary.delete_table()
        if self.secondary is not None:
            try:
                self.secondary.delete_table()
            except Exception as e:
                err_stack = traceback.format_exc()
                body = get_readable_error_message(e=e, err_stack=err_stack)
                self.logger.error(
                    f"Secondary storage got exception after success on the primary. {body}"
                )

    def update(
        self,
        item: BaseStoredItem,
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
            item (BaseStoredItem): Item to replace under the given uuid.
            created_if_not_exist:  if true then upsert, otherwise verify the item is not present and add()

        Raises:
            ValueError: if create_if_not_exist=False and the uuid used is NOT in the database already.

        """
        self.primary.update(
            item=item,
            update_updated_time=update_updated_time,
            create_if_not_exist=create_if_not_exist,
        )
        if self.secondary is not None:
            try:
                self.secondary.update(
                    item=item,
                    update_updated_time=update_updated_time,
                    create_if_not_exist=create_if_not_exist,
                )
            except Exception as e:
                err_stack = traceback.format_exc()
                body = get_readable_error_message(e=e, err_stack=err_stack)
                self.logger.error(
                    f"Secondary storage got exception after success on the primary. {body}"
                )

    def delete(self, uuids: Union[str, list[str]]) -> None:
        """
        Delete the stored object/record having the given uuid(s).

        Args:
            uuid (str): identifier under which an object is stored.  Returned by add().
        """
        self.primary.delete(uuids)
        if self.secondary is not None:
            try:
                self.secondary.delete(uuids)
            except Exception as e:
                err_stack = traceback.format_exc()
                body = get_readable_error_message(e=e, err_stack=err_stack)
                self.logger.error(
                    f"Secondary storage got exception after success on the primary. {body}"
                )

    def get_column_names(self) -> list[str]:
        """Get the names of the searchable columns in the table"""
        r = self.primary.get_column_names()
        if self.secondary is not None:
            try:
                self.secondary.get_column_names()
            except Exception as e:
                err_stack = traceback.format_exc()
                body = get_readable_error_message(e=e, err_stack=err_stack)
                self.logger.error(
                    f"Secondary storage got exception after success on the primary. {body}"
                )
        return r

    def _does_table_exist(self) -> bool:
        """As required by the super class"""
        r = self.primary._does_table_exist()
        if self.secondary is not None:
            try:
                self.secondary._does_table_exist()
            except Exception as e:
                err_stack = traceback.format_exc()
                body = get_readable_error_message(e=e, err_stack=err_stack)
                self.logger.error(
                    f"Secondary storage got exception after success on the primary. {body}"
                )
        return r
