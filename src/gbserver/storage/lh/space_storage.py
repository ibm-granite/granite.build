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

from typing import Optional, Self, Union

from gbserver.storage.lh.lh_storage import BaseLakehouseItemStorage
from gbserver.storage.space_storage import IStoredSpaceStorage
from gbserver.storage.stored_space import StoredSpace
from gbserver.types.constants import GB_SPACES_TABLE_NAME


class LhSpaceStorage(BaseLakehouseItemStorage, IStoredSpaceStorage):
    # TODO: this class should also inherit from BaseStoredSpaceStorage to pickup a lot of the function below.

    def __init__(self: Self, **kwargs):
        kwargs["item_class"] = StoredSpace
        if (
            kwargs.get("table_name") is None
        ):  # Allow for testing using alternate table names.
            kwargs["table_name"] = GB_SPACES_TABLE_NAME
        kwargs["unique_fields"] = ["uuid", "name", "git_repo_uri"]
        super().__init__(**kwargs)

    def _get_column_values(self: Self, item: StoredSpace) -> dict:
        fields_to_include = {"name", "git_repo_uri", "lakehouse_spacename"}
        json = item.model_dump(include=fields_to_include)
        # json["type"] = str(artifact.type)
        # json["tags"] = ",".join(sorted(artifact.tags))
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
        return self._get_by_single_field(
            column_name="name", column_value=name, allow_multiple=False
        )


if __name__ == "__main__":
    obj = StoredSpace(
        name="foo", git_repo_uri="http://foo.bar", lakehouse_spacename="myspace"
    )
    print(f"Space: {obj}")
    storage = LhSpaceStorage()
    print(f"Storage: {storage}")
