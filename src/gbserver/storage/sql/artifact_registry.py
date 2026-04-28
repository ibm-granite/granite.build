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

from typing import Self

from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.artifact_registry import BaseArtifactRegistry, IArtifactRegistry
from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class SQLArtifactRegistry(
    BaseSQLItemStorage[ArtifactRegistration], BaseArtifactRegistry, IArtifactRegistry
):
    # Some notes:

    def __init__(self: Self, **kwargs) -> None:
        kwargs["unique_columns"] = {
            ("uri", "space_name"): None,  # uri+space_name combo must be unique
            "checksum": "",  # checksum must be unique, but empty strings can be duplicated
        }
        # Tags are stored as a string, such as "tag1,tag2,tag3...", so we enable special handling for this column
        kwargs["exact_liked_list_columns"] = {"tags": "tags"}
        kwargs["indexed_columns"] = ["checksum"]

        super().__init__(**kwargs)
