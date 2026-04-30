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

from typing import List, Optional, Self, Union

from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.artifact_registry import IArtifactRegistry
from gbserver.storage.lh.lh_storage import BaseLakehouseItemStorage
from gbserver.types.artifact import ArtifactType
from gbserver.types.constants import GB_ARTIFACT_REGISTRY_TABLE_NAME
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# Schemas (columns exposed in addition to json and uuid )
# v1 (original) - "name", "uri", "space_name","username", "created_by_build_id", "type" "tags"
# v2 = v1 and adds "is_archived"
_IS_V2_SCHEMA = True  # Used ONLY for testing.
"""This is provided to allow control over what schema is used by the artifacts table. 
In general, this is NOT intended for any other purpose than writing tests that test the
automatic migration from v1 to v2 schemas.
It is NOT intended for general use.
"""


class LhArtifactRegistry(BaseLakehouseItemStorage, IArtifactRegistry):
    # TODO: this class should also inherit from BaseArtifactRegistry to pickup a lot of the function below.

    def __init__(self: Self, **kwargs) -> None:
        kwargs["item_class"] = ArtifactRegistration
        if kwargs.get("table_name") is None:  # Allow for testing using alternate table names.
            kwargs["table_name"] = GB_ARTIFACT_REGISTRY_TABLE_NAME
        kwargs["unique_fields"] = ["uuid", "uri"]
        super().__init__(**kwargs)

    def _get_column_values(self: Self, artifact: ArtifactRegistration) -> dict:
        fields_to_include = {
            "name",
            "uri",
            "space_name",
            "username",
            "created_by_build_id",
        }

        json = artifact.model_dump(include=fields_to_include)
        if _IS_V2_SCHEMA:
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
            ValueError: if more than one artifact is found with the given URI

        Returns:
            Optional[ArtifactRegistration]: None if not found, otherwise the single ArtifactRegistration with the given URI.
        """
        if space_name == "":
            logger.warning("space name is empty, fetching artifact uri '%s' from all spaces", uri)
            return self._get_by_single_field(
                column_name="uri", column_value=uri, allow_multiple=True
            )
        row_filter = {"uri": uri, "space_name": space_name}
        items = self.get_by_where(row_filter)
        item = self._get_only_one(items=items)
        assert isinstance(item, (type(None), ArtifactRegistration))
        return item


if __name__ == "__main__":
    obj = ArtifactRegistration(
        name="foo",
        uri="http://foo.bar",
        type=ArtifactType.FILESET,
        produced_by_build_id="someid",
        space_name="myspace",
        username="dawood",
        created_by_build_id="some uuid",
        created_by_target_id="some other uuid",
        lineage_hash="dasdfafd",
    )
    print(f"Artifact: {obj}")

    storage = LhArtifactRegistry()
    print(f"Storage: {storage}")
    storage.get_by_uuid(None)
