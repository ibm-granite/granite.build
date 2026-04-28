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
Storage models for node failure events.

Provides persistent storage of node failure history for long-term analysis
and reporting.
"""

import datetime
from typing import Any, Dict, Optional

from pydantic import Field

from gbserver.storage.stored_build import BaseStoredItem
from gbserver.utils.utils import get_utc_time


class StoredNodeFailure(BaseStoredItem):
    """
    Persistent storage model for node failure events.

    This is stored in the admin storage alongside builds, targets, and steps,
    allowing for cross-build analysis of node health patterns.

    Attributes:
        node_name: Name of the Kubernetes node
        build_id: ID of the build that experienced the failure
        launch_id: ID of the specific launch/step within the build
        failure_type: Type of failure (e.g., "FailedMount", "FailedAttachVolume")
        created_time: When the failure was recorded
        retry_count: Which retry attempt this was (0 = first attempt)
        metadata: Additional context about the failure
        resolved: Whether the issue was resolved (by successful retry or manual fix)
        resolved_timestamp: When the issue was resolved
    """

    node_name: str = Field(..., description="Name of the Kubernetes node")
    build_id: str = Field(..., description="ID of the build")
    launch_id: str = Field(..., description="ID of the launch/step")
    failure_type: str = Field(
        ...,
        description="Type of failure (e.g., FailedMount, FailedAttachVolume)",
    )
    # The name of this field must match that defined in storage.CREATED_TIME_FIELD_NAME
    created_time: datetime.datetime = Field(
        default_factory=get_utc_time,
        description="When the failure was recorded",
    )
    retry_count: int = Field(
        default=0,
        description="Which retry attempt this was",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context about the failure",
    )
    resolved: bool = Field(
        default=False,
        description="Whether the issue was resolved",
    )
    resolved_timestamp: Optional[datetime.datetime] = Field(
        default=None,
        description="When the issue was resolved",
    )
