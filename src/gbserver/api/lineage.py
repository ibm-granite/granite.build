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

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from lakehouse.api import JobStats

from gbserver.lineage.openlineage_models import (
    ArtifactLineageRequest,
)
from gbserver.lineage.openlineage_models import LineageEvent as OpenLineageEvent
from gbserver.lineage.openlineage_models import (
    PaginatedResponse,
    TagSearchRequest,
)
from gbserver.lineage.openlineage_service import LineageService, LineageServiceFactory
from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.constants import GBSERVER_LINEAGE_PROVIDER

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


@lineage_api.post("/artifact/runs")
def get_lineage_by_artifact(request: ArtifactLineageRequest):
    service = _get_openlineage_service()
    total, results = service.search_runs_by_artifact(
        request.repo_id, request.limit, request.offset
    )
    return PaginatedResponse(
        count=len(results),
        total=total,
        limit=request.limit,
        offset=request.offset,
        runs=results,
    )


@lineage_api.get("/{run_id}")
def get_lineage_event(run_id: str):
    service = _get_openlineage_service()
    result = service.get_run_lineage(run_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run not found: {run_id}",
        )
    return result
