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
Send metrics to a backend for analytics and display in a  dashboard.
"""

from datetime import datetime
from enum import StrEnum, auto
from typing import Any, List, Optional, Self, Union

from pydantic import BaseModel, Field


class MetricUnits(StrEnum):
    """The units to use to interpret the metric's value."""

    TIMESTAMP = auto()  # a ISO 8601 format timestamp (e.g. time when the status changed)
    SECONDS = auto()  # a time duration in seconds (e.g. processing delay)
    COUNT = auto()  # a count of something (e.g. retries)


class MetricName(StrEnum):
    """The name of the metric."""

    PROCESSING_DELAY = auto()
    VALIDATION_TIME = auto()
    EXP_MOV_AVG_PROCESSING_DELAY = auto()
    APPWRAPPER_STATUS_CHANGE_TIMESTAMP = auto()
    BUILD_STATUS_RACE_DETECTED = auto()


class MetricMetadata(BaseModel):
    """Metadata associated with a single metric."""

    username: str = ""
    build_id: str = ""
    targetrun_id: str = ""
    targetsteprun_id: str = ""
    targetstep_uri: str = ""
    target_name: str = ""
    launch_id: str = ""
    k8s_resource_type: str = ""
    k8s_resource_name: str = ""
    k8s_resource_namespace: str = ""
    k8s_resource_status: str = ""
    expected_status: str = ""


class Metric(BaseModel):
    """A single metric."""

    name: MetricName
    value: Union[str, float, int, datetime]
    units: MetricUnits = MetricUnits.SECONDS
    metadata: Optional[MetricMetadata] = None

    def model_post_init(self: Self, context: Any, /) -> None:
        if isinstance(self.value, (int, float)):
            self.value = str(self.value)  # convert to string for pushing
        elif isinstance(self.value, datetime):
            self.value = self.value.isoformat()  # convert to string for pushing
            self.units = MetricUnits.TIMESTAMP


class Metrics(BaseModel):
    """A collection of metrics."""

    metrics: List[Metric] = Field(default_factory=list)
