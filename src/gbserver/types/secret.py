#!/usr/bin/env python3

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

"""
A client for the IBM Cloud Secrets Manager API.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class MySecret(BaseModel):
    """A secret from the secrets manager."""

    created_by: str = ""
    created_at: datetime
    crn: str = ""
    id: str
    name: str = ""
    secret_group_id: str = ""
    secret_type: str = ""
    state: int = -1
    state_description: str = ""
    updated_at: Optional[datetime] = None
    description: str = ""
    payload: str
    # custom_metadata: dict
    # labels: list[str]
    # downloaded: bool
    # locks_total: int
    # versions_total: int


class TokenResponse(BaseModel):
    """Response from a IAM token request."""

    access_token: str
    refresh_token: str
    ims_user_id: int
    token_type: str  # "Bearer"
    expires_in: int
    expiration: int
    scope: str  # "ibm openid"
