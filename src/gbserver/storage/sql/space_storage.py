#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Self

from gbserver.storage.space_storage import BaseStoredSpaceStorage, IStoredSpaceStorage
from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.storage.stored_space import StoredSpace


class SQLSpaceStorage(
    BaseSQLItemStorage[StoredSpace], BaseStoredSpaceStorage, IStoredSpaceStorage
):

    def __init__(self: Self, **kwargs):
        # kwargs['item_class'] = StoredSpace
        # if kwargs.get('table_name') is None:    # Allow for testing using alternate table names.
        #     kwargs['table_name'] = GB_SPACES_TABLE_NAME
        # Only `name` is unique.  `git_repo_uri` is intentionally NOT unique
        # so that multiple aliases (e.g. the legacy `standalone` name and the
        # current `public` name) can point at the exact same URI without
        # contortion at insert time.
        kwargs["unique_columns"] = {"name": None}
        super().__init__(**kwargs)

    # def _get_column_values(self:Self, item: StoredSpace) -> dict:
    #     fields_to_include = {"name", "git_repo_uri", "lakehouse_spacename" }
    #     json = item.model_dump(include=fields_to_include)
    #     return json

    # def get_by_name(self, name:str) -> Optional[StoredSpace]:
    #     """Look up the unique space by name.

    #     Args:
    #         name (str): _description_

    #     Raises:
    #         ValueError: if more than 1 space is found with the given name.

    #     Returns:
    #         StoredSpace: named space or None if not found.
    #     """
    #     return self._get_by_single_field(column_name="name", column_value=name, allow_multiple=False)
