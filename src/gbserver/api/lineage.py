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

from typing import Any, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict

from gbserver.api.build_files_paths import authorize_build_read_access
from gbserver.api.utils import has_space_member_access
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
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# search_lineage_events scans backend pages (see docstring there) to keep
# offset/limit and total accurate for the caller's accessible runs rather
# than the global unfiltered set. Bounds the worst case when a caller's
# accessible fraction is a tiny sliver of a huge global result set.
_SEARCH_SCAN_BACKEND_PAGE_SIZE = 100
_SEARCH_SCAN_MAX_BACKEND_ITEMS = 2000


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
def get_build_jobstats(request: Request, build_id: str) -> BuildJobStatsResponse:
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
    authorize_build_read_access(request, build)

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
def get_target_jobstats(request: Request, target_id: str) -> TargetJobStatsResponse:
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

    # Authorize against the target's parent build, since the target itself
    # carries no owner/space of its own.
    build = storage.build_storage.get_by_uuid(target.build_id)
    if build is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Build with id {target.build_id} not found",
        )
    assert isinstance(build, StoredBuild)
    authorize_build_read_access(request, build)

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
        # Resolve the provider at call time (handles standalone established at
        # runtime; avoids the cached-constant / env-leak fragility).
        from gbserver.lineage.jobstats import _resolve_lineage_provider

        _openlineage_service = LineageServiceFactory.create(_resolve_lineage_provider())
    return _openlineage_service


@lineage_api.post("/")
def ingest_lineage_event(event: OpenLineageEvent):
    service = _get_openlineage_service()
    service.emit_event(event.model_dump())
    return {"status": "accepted"}


@lineage_api.post("/search")
def search_lineage_events(request: Request, body: TagSearchRequest):
    """Search lineage runs by tag.

    An empty tag list matches every run in the shared lineage backend across
    all spaces, so results are access-filtered below. search_lineage_by_tags
    paginates the UNFILTERED backend set, so filtering only the caller's
    requested page would (a) leak the global unfiltered count via `total`
    and (b) break offset/limit pagination for the caller — a page could come
    back near-empty after filtering even though the caller has many
    accessible runs on other backend pages, so `count < limit` would no
    longer reliably mean "no more results".

    To keep pagination correct from the caller's point of view, this scans
    backend pages (in fixed-size chunks, not the caller's own limit) up to
    _SEARCH_SCAN_MAX_BACKEND_ITEMS, collecting every accessible run, then
    applies the caller's offset/limit to that accessible set. `total` is
    always the accessible count, never the global unfiltered count. If the
    scan cap is hit before the backend is exhausted, `total`/`count` become
    a lower bound on the true accessible count (logged when this happens).
    """
    service = _get_openlineage_service()

    accessible: list[dict] = []
    backend_offset = 0
    backend_total: Optional[int] = None
    while backend_total is None or backend_offset < backend_total:
        if backend_offset >= _SEARCH_SCAN_MAX_BACKEND_ITEMS:
            logger.warning(
                "search_lineage_events: hit scan cap of %d backend items for "
                "tags=%s; total/count are a lower bound on the true accessible count",
                _SEARCH_SCAN_MAX_BACKEND_ITEMS,
                body.tags,
            )
            break
        backend_total, page_results = service.search_lineage_by_tags(
            body.tags, _SEARCH_SCAN_BACKEND_PAGE_SIZE, backend_offset
        )
        if not page_results:
            break
        for result in page_results:
            run_facets = (result.get("run") or {}).get("facets") or {}
            owner = (run_facets.get("job_details") or {}).get("owner", "")
            space_name = (run_facets.get("tags") or {}).get("space_name", "")
            has_access, _ = has_space_member_access(
                request, username_on_target=owner, space_name=space_name
            )
            if not has_access:
                continue
            # job_input_params carries raw build.yaml step config and can embed
            # credentials — redact it here too, same as get_build_jobstats, rather
            # than widening who can read pipeline secrets via search.
            run_facets.pop("job_input_params", None)
            accessible.append(result)
        backend_offset += len(page_results)

    page = accessible[body.offset : body.offset + body.limit]
    return PaginatedResponse(
        count=len(page),
        total=len(accessible),
        limit=body.limit,
        offset=body.offset,
        runs=page,
    )


@lineage_api.post("/artifact")
def get_artifact_graph(request: Request, body: ArtifactGraphRequest):
    """Get the lineage DAG for an artifact, traversing downstream or upstream.

    Runs are looked up by artifact name/url in the external lineage backend,
    which is not itself space-scoped, so results can span multiple spaces —
    each run is filtered below to the caller's own runs or spaces they belong
    to, rather than gating the whole request on a single space.
    """
    if body.direction not in ("downstream", "upstream", "both"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="direction must be 'downstream', 'upstream', or 'both'",
        )

    if not body.artifact_name and not body.artifact_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either artifact_name or artifact_url must be provided",
        )

    service = _get_openlineage_service()
    try:
        result = service.get_artifact_graph(
            artifact_name=body.artifact_name,
            artifact_url=body.artifact_url,
            artifact_type=body.artifact_type,
            max_depth=body.max_depth,
            direction=body.direction,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    if result is None:
        identifier = body.artifact_name or body.artifact_url
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

        entry = ArtifactRunEntry(
            job_name=metadata.get("job_name") or run.get("name", ""),
            job_namespace=metadata.get("job_namespace") or "",
            job_type=metadata.get("job_type") or "",
            run_id=metadata.get("run_id") or "",
            created_at=metadata.get("created_at") or "",
            status=metadata.get("state") or "",
            tags=tags,
            inputs=inputs,
            outputs=outputs,
            job_id=metadata.get("job_id") or "",
            job_status=metadata.get("job_status") or "",
            job_started_at=metadata.get("job_started_at") or "",
            job_completed_at=metadata.get("job_completed_at") or "",
            release_id=metadata.get("release_id") or "",
            category=metadata.get("category") or "",
            owner=metadata.get("owner") or "",
            source_code_details=metadata.get("source_code_details") or {},
            job_input_params=metadata.get("job_input_params") or {},
            execution_stats=metadata.get("execution_stats") or {},
            job_output_stats=metadata.get("job_output_stats") or {},
        )
        # job_namespace is written as f"{space_name}/{build_name}" (see
        # wandb_jobstats._build_events_for_target); split on the first "/"
        # to recover the space and drop runs from spaces the caller can't
        # access. Runs missing both an owner and a namespace fail closed.
        run_space_name = entry.job_namespace.split("/", 1)[0]
        has_access, _ = has_space_member_access(
            request, username_on_target=entry.owner, space_name=run_space_name
        )
        if has_access:
            runs.append(entry)

    return ArtifactGraphResponse(
        root_id=result["root_id"],
        runs=runs,
        truncated=result["truncated"],
    )
