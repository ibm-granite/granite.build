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

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel


class RunState(str, Enum):
    START = "START"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    ABORT = "ABORT"
    FAIL = "FAIL"
    OTHER = "OTHER"


class Run(BaseModel):
    runId: str
    facets: Dict[str, Any] = {}


class Job(BaseModel):
    namespace: str
    name: str
    facets: Dict[str, Any] = {}


class Dataset(BaseModel):
    namespace: str
    name: str
    facets: Dict[str, Any] = {}


class LineageDatasetEvent(BaseModel):
    eventTime: str
    producer: str
    schemaURL: Optional[str] = (
        "https://openlineage.io/spec/2-0-2/OpenLineage.json#/$defs/DatasetEvent"
    )
    dataset: Dataset


class LineageJobEvent(BaseModel):
    eventTime: str
    producer: str
    schemaURL: Optional[str] = "https://openlineage.io/spec/2-0-2/OpenLineage.json#/$defs/JobEvent"
    job: Job
    inputs: Optional[list[Dataset]] = []
    outputs: Optional[list[Dataset]] = []


class LineageEvent(BaseModel):
    eventType: RunState
    eventTime: str
    run: Run
    job: Job
    inputs: Optional[list[Dataset]] = []
    outputs: Optional[list[Dataset]] = []
    producer: str
    schemaURL: Optional[str] = "https://openlineage.io/spec/2-0-2/OpenLineage.json#/$defs/RunEvent"


class TagSearchRequest(BaseModel):
    tags: list[str] = []
    limit: int = 10
    offset: int = 0


class ArtifactLineageRequest(BaseModel):
    repo_id: str
    limit: int = 10
    offset: int = 0


class PaginatedResponse(BaseModel):
    count: int
    total: int
    limit: int
    offset: int
    runs: list
