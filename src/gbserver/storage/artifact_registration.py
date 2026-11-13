# Copyright Granite.Build Authors
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

from datetime import datetime
from enum import StrEnum, auto
from typing import Optional, Self

from pydantic import Field

from gbserver.storage.storage import BaseStoredItem, TaggedItem
from gbserver.types.artifact import ArtifactType
from gbserver.utils.utils import get_utc_time


class ArtifactRegistrationStatus(StrEnum):
    """Status of the artifact"""

    PENDING = auto()  # still being pushed or lhpush failed
    SUCCESS = auto()  # artifact push completed successfully
    FAILED = auto()  # build failed before artifact could be pushed
    CANCELLED = auto()  # build was cancelled before artifact could be pushed


class ArtifactRegistration(BaseStoredItem, TaggedItem):
    # Required initializations
    type: ArtifactType
    uri: str  # s3://cosbucket/path/to/filename data URI
    space_name: str
    username: str  # GitHub username
    # set default status to success for backwards compatibility
    status: ArtifactRegistrationStatus = ArtifactRegistrationStatus.SUCCESS

    # Defaulting initializations
    name: str = ""
    is_archived: bool = False
    created_by_build_id: str = ""  # empty means not produced by a build
    created_by_target_id: str = ""  # empty means not produced by a build
    created_by_step_id: str = (
        ""  # empty means not produced by a builda  TODO: left for temp backwards compat. needs to be removed.
    )
    created_at: datetime = Field(
        default_factory=get_utc_time,
        json_schema_extra={"help": "Time at which it was created"},
    )

    # Added after schema change to include is_archived   - 4/15/2025 or so
    origin_uris: Optional[list[str]] = None
    certified_no_restrictions: bool = False
    description: str = ""

    checksum: str = ""

    def __init__(self: Self, **kwargs) -> None:
        super().__init__(**kwargs)
        # self.created_by_step_id = self.created_by_target_id
        self.created_by_step_id = "unspecified"
