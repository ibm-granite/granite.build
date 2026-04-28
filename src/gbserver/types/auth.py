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

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from gbserver.utils.utils import get_time


class User(BaseModel):
    """Info about the user."""

    login: str
    id: int
    url: str
    html_url: str
    name: str
    email: str = ""
    auth_provider: str = "github"
    gbserver_created_at: datetime = Field(default_factory=get_time)

    @field_validator("email", mode="before")
    @classmethod
    def _coerce_null_email(cls, v):
        """GitHub API may return null for email; coerce to empty string."""
        return v or ""
