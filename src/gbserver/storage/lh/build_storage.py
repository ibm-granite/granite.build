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

from gbserver.storage.build_storage import IStoredBuildStorage
from gbserver.storage.lh.lh_storage import BaseLakehouseItemStorage
from gbserver.storage.storage import CREATED_TIME_FIELD_NAME, UPDATED_TIME_FIELD_NAME
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.constants import GB_BUILDS_TABLE_NAME

_IS_V2_SCHEMA = True  # Used ONLY for testing.
"""This is provided to allow control over what schema is used by the table. 
In general, this is NOT intended for any other purpose than writing tests that test the
automatic migration from v1 to v2 schemas.
It is NOT intended for general use.
"""


class LhBuildStorage(BaseLakehouseItemStorage, IStoredBuildStorage):
    # TODO: this class should also inherit from BaseStoredBuldStorage to pickup a lot of the function below.

    def __init__(self, **kwargs) -> None:
        kwargs["item_class"] = StoredBuild
        if kwargs.get("table_name") is None:  # Allow for testing using alternate table names.
            kwargs["table_name"] = GB_BUILDS_TABLE_NAME
        kwargs["unique_fields"] = ["uuid"]
        super().__init__(**kwargs)

    def _get_column_values(self, item: StoredBuild) -> dict:
        fields_to_include = {"space_name", "name", "username", "status", "source_uri"}

        json = item.model_dump(include=fields_to_include)

        json["status"] = item.status.name

        if _IS_V2_SCHEMA:
            # It is not strictly necessary to have the column name equal the field name, but why not.
            json[CREATED_TIME_FIELD_NAME] = item.created_time
            json[UPDATED_TIME_FIELD_NAME] = item.updated_time

        return json


if __name__ == "__main__":
    obj = StoredBuild(
        name="bname",
        space_name="myspace",
        source_uri="https://git.ibm.com",
        username="dawood",
        build_archive="",
    )
    print(f"Build: {obj}")
