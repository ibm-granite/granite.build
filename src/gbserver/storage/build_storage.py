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

"""Build storage module."""

from gbserver.storage.storage import (
    CREATED_TIME_FIELD_NAME,
    UPDATED_TIME_FIELD_NAME,
    BaseItemStorage,
    IItemStorage,
)
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.constants import GB_BUILDS_TABLE_NAME


class IStoredBuildStorage(IItemStorage[StoredBuild]):
    """I Stored Build Storage implementation."""

    pass


# These are generally not used externally except by tests.
_BUILD_SCHEMA_VERSION1 = 1  # base schema with space_name, name, username, status, source_uri
_BUILD_SCHEMA_VERSION2 = 2  # Add tags column, 10/2025
_BUILD_SCHEMA_VERSION_LATEST = _BUILD_SCHEMA_VERSION2


class BaseStoredBuildStorage(BaseItemStorage[StoredBuild], IStoredBuildStorage):
    """Base Stored Build Storage implementation."""

    def __init__(self, **kwargs) -> None:
        kwargs["item_class"] = StoredBuild
        if kwargs.get("table_name") is None:  # Allow for testing using alternate table names.
            kwargs["table_name"] = GB_BUILDS_TABLE_NAME
        super().__init__(**kwargs)
        self._schema_version = _BUILD_SCHEMA_VERSION_LATEST  # Not private so that tests can modify as needed for testing

    def _get_column_values(self, item: StoredBuild) -> dict:
        fields_to_include = {"space_name", "name", "username", "status", "source_uri"}

        json = item.model_dump(include=fields_to_include)

        json["status"] = item.status.name

        json[CREATED_TIME_FIELD_NAME] = item.created_time
        json[UPDATED_TIME_FIELD_NAME] = item.updated_time

        if self._schema_version > _BUILD_SCHEMA_VERSION1:
            json["tags"] = ",".join(sorted(item.tags))

        return json

    @classmethod
    def _get_sample_item(cls) -> StoredBuild:
        """Implemented per superclass requirements to return an item for use by BaseItemStorage"""
        item = StoredBuild(
            name="build-name",
            space_name="space-name",
            source_uri=f"https://some.url",
            username="some-user",
            tags=["tag1"],
        )
        return item
