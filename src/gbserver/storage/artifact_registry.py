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

from abc import abstractmethod
from typing import List, Optional, Self, Union

from sqlalchemy.exc import IntegrityError

from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.storage import BaseItemStorage, IItemStorage
from gbserver.types.artifact import ArtifactType
from gbserver.types.constants import GB_ARTIFACT_REGISTRY_TABLE_NAME
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# This one is not used here, but is in testing of migration from v1 to v2
_BUILD_SCHEMA_VERSION1 = 1  # base schema with uri, space_name, username, name, created_by_build_id, is_archived, type, tags
_BUILD_SCHEMA_VERSION2 = 2  # Add checksum 1/20/2026

_BUILD_SCHEMA_VERSION_LATEST = _BUILD_SCHEMA_VERSION2


class IArtifactRegistry(IItemStorage[ArtifactRegistration]):
    # Some notes:

    @abstractmethod
    def get_by_uri(
        self: Self, uri: str, space_name: str = ""
    ) -> Union[List[ArtifactRegistration], Optional[ArtifactRegistration]]:
        """
        Get the artifact registration(s) associated with the given URI in the given space.

        Args:
            uri (str): URI to search for
            space_name (str): Name of the Space in which to search. If this is empty string then we will search all spaces.

        Raises:
            ValueError: when space_name is given and more than one artifact is found with the given URI and space_name

        Returns:
            List[ArtifactRegistration]: If space name is not given, a List of artifact registrations - zero length if none found.
            Optional[ArtifactRegistration]: If space name is given, None if not found, otherwise the single ArtifactRegistration with the given URI.
        """
        raise NotImplementedError(
            f"Sub-class {self.__class__.__name__} did not implement method throwing this exception"
        )


class ChecksumConflictException(Exception):

    def __init__(self, existing: ArtifactRegistration):
        self.existing_artifact = existing


class BaseArtifactRegistry(BaseItemStorage[ArtifactRegistration], IArtifactRegistry):

    def __init__(self: Self, **kwargs) -> None:
        kwargs["item_class"] = ArtifactRegistration
        if (
            kwargs.get("table_name") is None
        ):  # Allow for testing using alternate table names.
            kwargs["table_name"] = GB_ARTIFACT_REGISTRY_TABLE_NAME
        super().__init__(**kwargs)
        self._schema_version = _BUILD_SCHEMA_VERSION_LATEST

    def add(
        self: Self, items: Union[ArtifactRegistration, list[ArtifactRegistration]]
    ) -> Union[str, list[str]]:
        """Add artifact(s) to the registry.

        In addition to the base class uniqueness checks, this method also checks for
        duplicate checksums when the artifact's checksum is non-empty.

        Args:
            items: ArtifactRegistration or list of ArtifactRegistrations to add.

        Returns:
            UUID or list of UUIDs of the added artifacts.

        Raises:
            ChecksumConflictException: If an artifact with the same non-empty checksum already exists.
        """
        items_list = items if isinstance(items, list) else [items]

        try:
            r = super().add(items)
            return r
        except IntegrityError as e: # TODO should avoid using exception from underlying (sql alchemy) implementation.
            # Check if this is a checksum uniqueness violation
            for item in items_list:
                if item.checksum != "":
                    existing = self.get_by_where({"checksum": item.checksum})
                    if existing:
                        raise ChecksumConflictException(existing=existing[0]) from e
            # Not a checksum conflict, re-raise the original error
            raise

    def update(
        self: Self,
        item: ArtifactRegistration,
        update_updated_time: bool = True,
        create_if_not_exist: bool = True,
    ) -> None:
        """Update an artifact in the registry.

        Args:
            item: ArtifactRegistration to update.

        Raises:
            ChecksumConflictException: If the updated checksum conflicts with an existing artifact.
        """
        try:
            super().update(item, update_updated_time, create_if_not_exist)
        except IntegrityError as e:
            # Check if this is a checksum uniqueness violation
            if item.checksum != "":
                existing = self.get_by_where({"checksum": item.checksum})
                if existing and existing[0].uuid != item.uuid:
                    raise ChecksumConflictException(existing=existing[0]) from e
            # Not a checksum conflict, re-raise the original error
            raise

    def _get_column_values(self: Self, artifact: ArtifactRegistration) -> dict:
        fields_to_include = [
            "name",
            "uri",
            "space_name",
            "username",
            "created_by_build_id",
        ]
        if self._schema_version >= _BUILD_SCHEMA_VERSION2:
            fields_to_include.append("checksum")

        json = artifact.model_dump(include=fields_to_include)
        json["is_archived"] = artifact.is_archived

        json["type"] = str(artifact.type)
        json["tags"] = ",".join(sorted(artifact.tags))

        return json

    def get_by_uri(
        self: Self, uri: str, space_name: str = ""
    ) -> Union[List[ArtifactRegistration], Optional[ArtifactRegistration]]:
        """
        Get the artifact registration(s) associated with the given URI in the given space.

        Args:
            uri (str): URI to search for
            space_name (str): Name of the Space in which to search. If this is empty string then we will search all spaces.

        Raises:
            ValueError: when space_name is given and more than one artifact is found with the given URI and space_name

        Returns:
            List[ArtifactRegistration]: If space name is not given, a List of artifact registrations - zero length if none found.
            Optional[ArtifactRegistration]: If space name is given, None if not found, otherwise the single ArtifactRegistration with the given URI.
        """
        if space_name == "":
            self.logger.warning(
                f"space name is empty, fetching artifact uri {uri} from all spaces"
            )
            return self._get_by_single_field(
                column_name="uri", column_value=uri, allow_multiple=True
            )
        row_filter = {"uri": uri, "space_name": space_name}
        items = self.get_by_where(row_filter)
        item = self._get_only_one(items=items)
        assert isinstance(item, (type(None), ArtifactRegistration))
        return item

    @classmethod
    def _get_sample_item(cls) -> ArtifactRegistration:
        """Implemented per superclass requirements to return an item for use by BaseItemStorage"""
        item = ArtifactRegistration(
            type=ArtifactType.TABLE,
            uri="https://some-uri",
            space_name="space-name",
            username="some-user",
            tags=["tag1"],
        )
        return item
