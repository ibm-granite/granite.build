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

from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from lakehouse.api import JobStats

from gbserver.lineage.openlineage_models import (
    ArtifactGraphRequest,
    ArtifactGraphResponse,
    ArtifactRunEntry,
)
from gbserver.lineage.openlineage_models import LineageEvent as OpenLineageEvent
from gbserver.lineage.openlineage_models import (
    LineageNodeRef,
    PaginatedResponse,
    TagSearchRequest,
)
from gbserver.lineage.openlineage_service import LineageService, LineageServiceFactory
from gbserver.lineage.openlineage_utils import parse_hf_url
from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.constants import GBSERVER_LINEAGE_PROVIDER


def _uri_from_url(url: Optional[str]) -> Optional[str]:
    """Derive an hf:// URI from a huggingface.co URL."""
    if not url:
        return None
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "huggingface.co"
        org, name, artifact_type = parse_hf_url(url)
        type_part = f"{artifact_type}s/" if artifact_type != "model" else ""
        return f"hf://{host}/{type_part}{org}/{name}"
    except Exception:
        return url


lineage_api = FastAPI()


class TargetJobStatsResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    target_id: str
    jobstats: dict[str, list[Any]]


class BuildJobStatsResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    build_id: str
    targets: list[dict[str, list[Any]]]


@lineage_api.get("/build/{build_id}")
def get_build_jobstats(build_id: str) -> BuildJobStatsResponse:
    """Get JobStats for all targets in a build."""
    storage = get_admin_storage()

    from gbserver.lineage.jobstats import get_lineage_store

    # Get the build
    build = storage.build_storage.get_by_uuid(build_id)
    if build is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Build with id {build_id} not found",
        )
    assert isinstance(build, StoredBuild)

    # Get all targets for this build
    row_filter = {"build_id": build_id}
    targets = storage.target_storage.get_by_where(row_filter)

    # Collect JobStats for each target
    jobstats_storage = get_lineage_store()
    target_responses: list[dict[str, list[Any]]] = []

    for target in targets:
        assert isinstance(target, StoredTargetRun)
        _, jobstats_dict = jobstats_storage.create_jobstats_for_target(
            storage, target, build
        )
        target_responses.append(jobstats_dict)

    return BuildJobStatsResponse(build_id=build_id, targets=target_responses)


@lineage_api.get("/target/{target_id}")
def get_target_jobstats(target_id: str) -> TargetJobStatsResponse:
    """Get JobStats for a target run, grouped by output artifact name."""
    storage = get_admin_storage()

    # Get the target run
    target = storage.target_storage.get_by_uuid(target_id)
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target with id {target_id} not found",
        )
    assert isinstance(target, StoredTargetRun)

    from gbserver.lineage.jobstats import get_lineage_store

    # Create JobStats using existing method
    jobstats_storage = get_lineage_store()
    _, jobstats_dict = jobstats_storage.create_jobstats_for_target(storage, target)

    return TargetJobStatsResponse(target_id=target_id, jobstats=jobstats_dict)


# --- OpenLineage endpoints ---

_openlineage_service: Optional[LineageService] = None


def _get_openlineage_service() -> LineageService:
    global _openlineage_service
    if _openlineage_service is None:
        _openlineage_service = LineageServiceFactory.create(GBSERVER_LINEAGE_PROVIDER)
    return _openlineage_service


@lineage_api.post("/")
def ingest_lineage_event(event: OpenLineageEvent):
    service = _get_openlineage_service()
    service.emit_event(event.model_dump())
    return {"status": "accepted"}


@lineage_api.post("/search")
def search_lineage_events(request: TagSearchRequest):
    service = _get_openlineage_service()
    total, results = service.search_lineage_by_tags(
        request.tags, request.limit, request.offset
    )
    return PaginatedResponse(
        count=len(results),
        total=total,
        limit=request.limit,
        offset=request.offset,
        runs=results,
    )


@lineage_api.post("/artifact")
def get_artifact_graph(request: ArtifactGraphRequest):
    """Get the lineage DAG for an artifact, traversing downstream or upstream."""
    if request.direction not in ("downstream", "upstream", "both"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="direction must be 'downstream', 'upstream', or 'both'",
        )

    if not request.artifact_name and not request.artifact_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either artifact_name or artifact_url must be provided",
        )

    service = _get_openlineage_service()
    try:
        result = service.get_artifact_graph(
            artifact_name=request.artifact_name,
            artifact_url=request.artifact_url,
            artifact_type=request.artifact_type,
            max_depth=request.max_depth,
            direction=request.direction,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    if result is None:
        identifier = request.artifact_name or request.artifact_url
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact not found: {identifier}",
        )

    nodes = result.get("nodes", [])
    edges = result.get("edges", [])

    nodes_by_id = {node["id"]: node for node in nodes}
    run_nodes = [n for n in nodes if n.get("node_type") == "run"]

    runs: list[ArtifactRunEntry] = []
    for run in run_nodes:
        node_id = run["id"]
        metadata = run.get("metadata") or {}
        tags = run.get("tags") or []

        inputs: list[LineageNodeRef] = []
        outputs: list[LineageNodeRef] = []
        for edge in edges:
            if edge["target"] == node_id:
                source_node = nodes_by_id.get(edge["source"], {})
                node_type = source_node.get("node_type", "")
                if node_type == "artifact":
                    source_meta = source_node.get("metadata") or {}
                    uri = source_meta.get("uri") or _uri_from_url(
                        source_meta.get("url")
                    )
                    inputs.append(
                        LineageNodeRef(
                            node_type="artifact",
                            name=source_node.get("name", edge["source"]),
                            uri=uri,
                            url=source_meta.get("url"),
                        )
                    )
                elif node_type == "run":
                    source_meta = source_node.get("metadata") or {}
                    inputs.append(
                        LineageNodeRef(
                            node_type="run",
                            name=source_node.get("name", ""),
                            run_id=source_meta.get("run_id"),
                            job_name=source_meta.get("job_name"),
                        )
                    )
            elif edge["source"] == node_id:
                target_node = nodes_by_id.get(edge["target"], {})
                node_type = target_node.get("node_type", "")
                if node_type == "artifact":
                    target_meta = target_node.get("metadata") or {}
                    uri = target_meta.get("uri") or _uri_from_url(
                        target_meta.get("url")
                    )
                    outputs.append(
                        LineageNodeRef(
                            node_type="artifact",
                            name=target_node.get("name", edge["target"]),
                            uri=uri,
                            url=target_meta.get("url"),
                        )
                    )
                elif node_type == "run":
                    target_meta = target_node.get("metadata") or {}
                    outputs.append(
                        LineageNodeRef(
                            node_type="run",
                            name=target_node.get("name", ""),
                            run_id=target_meta.get("run_id"),
                            job_name=target_meta.get("job_name"),
                        )
                    )

        runs.append(
            ArtifactRunEntry(
                job_name=metadata.get("job_name") or run.get("name", ""),
                job_namespace=metadata.get("job_namespace") or "",
                job_type=metadata.get("job_type") or "",
                run_id=metadata.get("run_id") or "",
                created_at=metadata.get("created_at") or "",
                status=metadata.get("state") or "",
                tags=tags,
                inputs=inputs,
                outputs=outputs,
            )
        )

    return ArtifactGraphResponse(
        root_id=result["root_id"],
        runs=runs,
        truncated=result["truncated"],
    )
