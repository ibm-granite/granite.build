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

"""
Storage model for space user membership.
"""

import datetime
from typing import Literal

from pydantic import Field

from gbserver.storage.storage import BaseStoredItem
from gbserver.utils.utils import get_utc_time


class StoredSpaceUser(BaseStoredItem):
    """
    Represents a user's membership in a space with an associated role.

    Each instance becomes a row in the gb_space_users table.
    The combination of space_name + username is unique.
    """

    space_name: str = Field(..., description="Name of the space")
    username: str = Field(..., description="Username of the member")
    role: Literal["admin", "member"] = Field(
        ..., description="Role of the user in the space: 'admin' or 'member'"
    )
    # Field names must match CREATED_TIME_FIELD_NAME / UPDATED_TIME_FIELD_NAME constants
    created_time: datetime.datetime = Field(
        default_factory=get_utc_time,
        description="When the membership was created",
    )
    updated_time: datetime.datetime = Field(
        default_factory=get_utc_time,
        description="When the membership was last updated",
    )
