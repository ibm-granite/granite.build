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
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


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
    model_config = ConfigDict(extra="allow")

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
    schemaURL: Optional[str] = (
        "https://openlineage.io/spec/2-0-2/OpenLineage.json#/$defs/JobEvent"
    )
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
    schemaURL: Optional[str] = (
        "https://openlineage.io/spec/2-0-2/OpenLineage.json#/$defs/RunEvent"
    )


class TagSearchRequest(BaseModel):
    tags: list[str] = []
    limit: int = 10
    offset: int = 0


class PaginatedResponse(BaseModel):
    count: int
    total: int
    limit: int
    offset: int
    runs: list


class GraphNodeType(str, Enum):
    ARTIFACT = "artifact"
    RUN = "run"


class GraphNode(BaseModel):
    id: str
    node_type: GraphNodeType
    name: str
    artifact_type: Optional[str] = None
    is_root: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str


class ArtifactGraphRequest(BaseModel):
    artifact_name: Optional[str] = None
    artifact_url: Optional[str] = None
    artifact_type: Optional[str] = None
    max_depth: int = Field(default=10, ge=1, le=50)
    direction: str = "both"


class LineageNodeRef(BaseModel):
    node_type: str
    name: str = ""
    uri: Optional[str] = None
    url: Optional[str] = None
    run_id: Optional[str] = None
    job_name: Optional[str] = None


class ArtifactRunEntry(BaseModel):
    job_name: str = ""
    job_namespace: str = ""
    job_type: str = ""
    run_id: str = ""
    created_at: str = ""
    status: str = ""
    tags: List[str] = Field(default_factory=list)
    inputs: List[LineageNodeRef] = Field(default_factory=list)
    outputs: List[LineageNodeRef] = Field(default_factory=list)


class ArtifactGraphResponse(BaseModel):
    root_id: str
    runs: List[ArtifactRunEntry]
    truncated: bool = False
