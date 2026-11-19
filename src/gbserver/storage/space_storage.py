#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Self

from gbserver.storage.storage import BaseItemStorage, IItemStorage
from gbserver.storage.stored_space import StoredSpace
from gbserver.types.constants import GB_SPACES_TABLE_NAME


class IStoredSpaceStorage(IItemStorage[StoredSpace]):

    def get_by_name(self, name: str) -> Optional[StoredSpace]:
        """Look up the unique space by name.

        Args:
            name (str): _description_

        Raises:
            ValueError: if more than 1 space is found with the given name.

        Returns:
            StoredSpace: named space or None if not found.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} did not implement this method that throwing the exception"
        )


class BaseStoredSpaceStorage(BaseItemStorage[StoredSpace], IStoredSpaceStorage):

    def __init__(self: Self, **kwargs):
        kwargs["item_class"] = StoredSpace
        if (
            kwargs.get("table_name") is None
        ):  # Allow for testing using alternate table names.
            kwargs["table_name"] = GB_SPACES_TABLE_NAME
        super().__init__(**kwargs)

    def _get_column_values(self: Self, item: StoredSpace) -> dict:
        fields_to_include = {"name", "git_repo_uri", "lakehouse_spacename"}
        json = item.model_dump(include=fields_to_include)
        return json

    def get_by_name(self, name: str) -> Optional[StoredSpace]:
        """Look up the unique space by name.

        Args:
            name (str): _description_

        Raises:
            ValueError: if more than 1 space is found with the given name.

        Returns:
            StoredSpace: named space or None if not found.
        """
        return self._get_by_single_field(  # type: ignore[return-value]
            column_name="name", column_value=name, allow_multiple=False
        )

    @classmethod
    def _get_sample_item(cls) -> StoredSpace:
        """Implemented per superclass requirements to return an item for use by BaseItemStorage"""
        item = StoredSpace(
            name="space_name",
            git_repo_uri="https:///some-repo",
            lakehouse_namespace="lh_namespace",
        )
        return item
