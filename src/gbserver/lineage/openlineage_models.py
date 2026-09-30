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

from gbserver.lineage.walk import ABSOLUTE_MAX_NODES_PER_LEVEL


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
    """One node of a walked graph.

    ``id`` is the artifact's normalized URI, or ``run:<job_id>`` for a run node --
    prefixed so a job id can never collide with an artifact URI in the shared id
    space that ``GraphEdge`` references.
    """

    id: str
    node_type: GraphNodeType
    name: str
    artifact_type: Optional[str] = None
    is_root: bool = False
    depth: Optional[int] = Field(
        default=None,
        description=(
            "Hops from the nearest seed, shortest path; 0 for a seed. None for a "
            "run node, which hangs off an edge rather than being walked to, and for "
            "an artifact the walk did not reach."
        ),
    )
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


class BuildGraphRequest(BaseModel):
    """A request for the lineage graph seeded from one build's artifacts.

    A build is not a graph node -- it is a way to seed one -- so this carries no
    node identity, only the build to seed from and how far to expand.
    """

    build_id: str
    max_depth: int = Field(default=10, ge=1, le=50)
    direction: str = "both"


class BuildGraphResponse(BaseModel):
    """The walked graph of a build.

    ``root_id`` is the build id, which names no node: a build-seeded graph has
    several roots, so no node is flagged ``is_root``.
    """

    root_id: str
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    truncated: bool = False
    unexpanded: int = 0


class LineageQueryRequest(BaseModel):
    """A lineage query seeded by an artifact, a job, or both.

    One entry point so a caller can ask however it happens to hold the artifact:
    by URI in any spelling, or by the job that produced it. Each field is optional
    on its own, but one of the two is required -- the route answers 400 otherwise.
    Both map to an indexed text column, so either combination is one predicate.

    There is deliberately no ``build_id`` filter: the index has no such column
    (a build is granite.build's own concept, empty on every imported row), and a
    build-scoped view goes through ``POST /lineage/build``, which resolves the build
    outside the index and seeds this same walk.
    """

    uri: Optional[str] = Field(
        default=None,
        description=(
            "The artifact's URI in any spelling; it is normalized server-side, so a "
            "browser URL and the runtime's own URI resolve to one artifact."
        ),
    )
    job_id: Optional[str] = Field(
        default=None,
        description="Seed from every endpoint of this job execution.",
    )
    max_depth: int = Field(default=10, ge=1, le=50)
    direction: str = "both"
    max_nodes_per_level: Optional[int] = Field(
        default=None,
        ge=1,
        le=ABSOLUTE_MAX_NODES_PER_LEVEL,
        description=(
            "Raise the per-level frontier ceiling for this one request, for an "
            "explicit 'show the full graph' action. Omit for the server default. "
            "Still bounded: a hub artifact's closure is unbounded in practice, so "
            "there is no value meaning 'no limit'."
        ),
    )


class LineageGraphResponse(BaseModel):
    """The walked graph.

    ``root_id`` is the URI the query resolved to, or ``""`` when the query named no
    single artifact (a job-seeded or unfiltered query has several roots, so no node
    is flagged ``is_root``).
    """

    root_id: str = ""
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    truncated: bool = False
    unexpanded: int = 0


class LineageJobEntry(BaseModel):
    """One job execution in the job listing, with what it read and wrote."""

    job_id: str
    job_namespace: str = ""
    space_name: str = ""
    owner: str = ""
    source_system: str = ""
    status: str = ""
    started_at: str = ""
    tags: List[str] = Field(default_factory=list)
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    job: Dict[str, Any] = Field(default_factory=dict)


class LineageJobsResponse(BaseModel):
    """A page of the jobs matching a filter.

    ``total`` is the unpaged count of distinct jobs, so a caller can tell how much is
    left rather than guessing from a short page.
    """

    jobs: List[LineageJobEntry] = Field(default_factory=list)
    total: int = 0
    limit: int = 100
    offset: int = 0


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
    job_id: str = ""
    job_status: str = ""
    job_started_at: str = ""
    job_completed_at: str = ""
    release_id: str = ""
    category: str = ""
    owner: str = ""
    source_code_details: Dict[str, Any] = Field(default_factory=dict)
    job_input_params: Dict[str, Any] = Field(default_factory=dict)
    execution_stats: Dict[str, Any] = Field(default_factory=dict)
    job_output_stats: Dict[str, Any] = Field(default_factory=dict)


class ArtifactGraphResponse(BaseModel):
    root_id: str
    runs: List[ArtifactRunEntry]
    truncated: bool = False
