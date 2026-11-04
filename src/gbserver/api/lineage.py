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

from typing import Annotated, Any, Literal, Optional

from fastapi import FastAPI, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict

from gbserver.api.build_files_paths import authorize_build_read_access
from gbserver.api.utils import has_space_member_access, is_super_admin
from gbserver.lineage.openlineage_models import LineageEvent as OpenLineageEvent
from gbserver.lineage.openlineage_models import (
    LineageGraphResponse,
    LineageJobDetail,
    LineageJobsResponse,
    LineageQueryRequest,
    PaginatedResponse,
    TagSearchRequest,
)
from gbserver.lineage.openlineage_service import LineageService, LineageServiceFactory
from gbserver.lineage.walk import DEFAULT_MAX_NODES_PER_LEVEL
from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.utils.logger import get_logger
from gbserver.utils.redaction import redact_sensitive

logger = get_logger(__name__)

# search_lineage_events scans backend pages (see docstring there) to keep
# offset/limit and total accurate for the caller's accessible runs rather
# than the global unfiltered set. Bounds the worst case when a caller's
# accessible fraction is a tiny sliver of a huge global result set.
_SEARCH_SCAN_BACKEND_PAGE_SIZE = 100
_SEARCH_SCAN_MAX_BACKEND_ITEMS = 2000
# The deepest walk a caller who is not a super admin may ask for on /graph.
_USER_MAX_DEPTH = 20
# The same cap LineageQueryRequest.job_ids applies to the graph.
_MAX_JOB_IDS = 1000


def get_redacted_job_input_params(source: dict) -> dict:
    """Read ``job_input_params`` from a run mapping, redacted for the read path.

    The member-readable ``search_lineage_events`` endpoint must mask this facet
    unconditionally: the write-side builder
    (``wandb_jobstats._build_events_for_target``) already masks secret-named keys,
    but going through this single accessor also protects any row persisted before
    that write-side masking landed. Non-secret step data (e.g. a ``commit_hash``) surfaces intact;
    ``redact_sensitive`` is idempotent, so re-masking an already-masked row is safe.

    :param source: a run's facets mapping.
    :returns: the ``job_input_params`` mapping with secret-named values redacted,
        or an empty dict when the key is absent/empty.
    """
    return redact_sensitive(source.get("job_input_params") or {})


lineage_api = FastAPI()


class TargetJobStatsResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    target_id: str
    jobstats: dict[str, list[Any]]


class BuildJobStatsResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    build_id: str
    targets: list[dict[str, list[Any]]]
    # Parallel to ``targets``: ``target_ids[i]`` owns ``targets[i]``.
    #
    # Additive, so the existing positional ``targets`` shape is unchanged for
    # current clients. It exists because that shape alone is no longer safe to
    # consume positionally: a target with neither input nor output artifacts now
    # contributes an EMPTY dict (it has no lineage events -- see
    # get_build_jobstats), and a client that drops falsy entries before zipping
    # against its own target list -- `[t for t in resp["targets"] if t]`, or a UI
    # that skips empty objects -- silently shifts every later target onto the
    # wrong lineage. Before the artifact-less skip every target contributed at
    # least one key, so that filter was harmless; it is not any more.
    #
    # Read this instead of inferring identity from position. It is populated in
    # the same loop as ``targets``, so the two cannot drift.
    target_ids: list[str]


@lineage_api.get("/build/{build_id}", tags=["gb admin"])
def get_build_jobstats(request: Request, build_id: str) -> BuildJobStatsResponse:
    """Get JobStats for all targets in a build.

    Every target of the build gets an entry, but a target with neither input nor
    output artifacts contributes an empty dict: it has no lineage events to
    report. This endpoint reaches ``create_jobstats_for_target`` directly, without
    going through ``select_recordable_targets``, so the builder's own artifact-less
    check is what produces that -- not a filter here.

    Because of those empty dicts, ``targets`` must not be consumed positionally
    against a separately-fetched target list: dropping the falsy entries (a
    natural "skip the targets with no lineage" filter) shifts every later entry
    onto the wrong target. ``target_ids`` is the parallel identity list --
    ``target_ids[i]`` owns ``targets[i]`` -- and is the supported way to attribute
    an entry. Both lists are built in one loop and are always the same length.
    """
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
    target_ids: list[str] = []

    for target in targets:
        assert isinstance(target, StoredTargetRun)
        _, jobstats_dict = jobstats_storage.create_jobstats_for_target(
            storage, target, build
        )
        # Appended together so the two lists stay index-aligned by construction;
        # an entry is only ambiguous if these ever separate.
        target_responses.append(jobstats_dict)
        target_ids.append(target.uuid)

    return BuildJobStatsResponse(
        build_id=build_id, targets=target_responses, target_ids=target_ids
    )


@lineage_api.get("/target/{target_id}", tags=["gb admin"])
def get_target_jobstats(request: Request, target_id: str) -> TargetJobStatsResponse:
    """Get JobStats for a target run, grouped by output artifact name.

    ``jobstats`` is empty for a target with neither input nor output artifacts --
    it has no lineage events. See ``get_build_jobstats``.
    """
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


_index_service = None


def _get_index_service():
    """The reader of ``gb_lineage_index``, whatever the configured provider.

    The ``lineage-index`` routes read the index directly: the lineage-indexer
    fills it from ``gb_targets`` or the lineage store, so it has data even when
    the provider is ``wandb`` or ``none``.
    """
    global _index_service
    if _index_service is None:
        # Deferred: importing the DB service at module scope would pull the storage
        # layer into every environment that never serves these routes.
        # pylint: disable=import-outside-toplevel
        from gbserver.lineage.db_service import DBLineageService

        _index_service = DBLineageService()
    return _index_service


@lineage_api.post("/", tags=["lineage store"])
def ingest_lineage_event(event: OpenLineageEvent):
    """Write one OpenLineage event to the configured lineage store.

    **Known risk: no authorization.** Any caller who reaches the API can write an
    event, and nothing ties its namespace, owner or tags to the caller. Under the
    ``wandb`` provider the event becomes a W&B run, and the W&B lineage indexer
    reads every run of the project, so an injected event shows up in the global
    ``POST /lineage/graph`` trace (which is not filtered per space by design). It
    cannot reach space-gated detail: ``GET /lineage/jobs/{id}`` checks the
    caller's access before returning owner, namespace or tags.
    """
    service = _get_openlineage_service()
    service.emit_event(event.model_dump())
    return {"status": "accepted"}


@lineage_api.post("/search", tags=["lineage store"], deprecated=True)
def search_lineage_events(request: Request, body: TagSearchRequest):
    """Search lineage runs by tag.

    **Deprecated.** Kept only for existing external callers (dmg-ng); nothing in
    granite.build (frontend, gbcli, gbmcp) calls it. New callers should resolve
    lineage through the index routes: ``GET /lineage/jobs`` to list jobs (by
    ``tag`` for what this route answered) and ``POST /lineage/graph`` with ``job_ids``
    for their lineage. It will be removed once
    those callers have migrated.

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
        try:
            backend_total, page_results = service.search_lineage_by_tags(
                body.tags, _SEARCH_SCAN_BACKEND_PAGE_SIZE, backend_offset
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Lineage tag search failed")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Lineage search is unavailable",
            ) from exc
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
            # Redact on the read path via the shared accessor (see
            # get_redacted_job_input_params).
            run_facets["job_input_params"] = get_redacted_job_input_params(run_facets)
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


# A POST, not a GET: ``job_ids`` -- a grouped node of a few hundred runs -- overruns
# proxy URL limits in a query string, and one route keeps a single shape to document.
@lineage_api.post("/graph", tags=["lineage-index"])
def query_lineage_graph_post(
    request: Request, body: LineageQueryRequest
) -> LineageGraphResponse:
    """Query the lineage graph by URI, by job(s), or both; one of them is required.

    The HTTP surface for :func:`query_lineage_graph`; see it for the semantics. One
    walk over every ``job_ids`` entry, so the runs are grouped across the whole set
    and the per-level cap and ``job_ids`` limit apply to the expansion as a whole.
    """
    return query_lineage_graph(request, body)


def query_lineage_graph(
    request: Request, body: LineageQueryRequest
) -> LineageGraphResponse:
    """Query the lineage graph seeded by an artifact, a job, or both.

    One entry point so a caller asks however it holds the artifact rather than the
    index dictating a lookup shape:

    - ``uri`` -- that artifact's lineage, up and down. Any spelling: a browser URL
      and the runtime's own URI normalize to one artifact.
    - ``job_id`` -- seeded from every endpoint of that execution.
    - both -- the union of their seeds.

    - ``job_ids`` -- each of several executions: the jobs behind a grouped node.

    One of them is required -- a 422 otherwise: a graph needs somewhere to start. A
    bad ``direction`` or an out-of-range limit is a 422 too.

    This returns the node/edge graph directly, with a ``depth`` per node. It also
    never 404s: an empty graph means "nothing recorded", which is a real answer, and a
    caller must not render it as an error. A storage failure is a 503, never an
    empty graph.

    There is no ``build_id`` filter. The index has no such column -- a build is a
    granite.build process concept, absent from every imported row -- and a
    build-scoped view goes through ``GET /lineage/build/{id}``, which resolves the
    build outside the index and seeds this same walk.

    Reads the lineage index directly, whatever the configured provider.

    **The graph is cross-space and is NOT filtered per space.** That is deliberate,
    and it is the one design decision here worth stating twice.

    A lineage graph carries no access to anything. What any caller sees:

    - artifact nodes: ``id`` (the normalized URI), ``name``, ``artifact_type``,
      ``depth``, ``is_root`` and ``metadata.uri``;
    - run nodes: ``id``, ``name`` and, in ``metadata``, ``job_id``, ``job_name``,
      ``job_type``, ``job_status``, ``job_started_at``, ``job_completed_at``,
      ``category`` and ``source_system``; grouped nodes add ``run_count`` and
      ``jobs_query``, ``representative_job_id``, ``source_uri`` and ``target_uri``
      (the first input and output, the terminal marker for an empty side); a node grouping
      jobs with the same inputs and outputs also adds ``source_uris`` and
      ``target_uris``;
    - the edges.

    Owner, namespace and tags are never on the graph. ``job_id`` is a global
    identifier: the graph needs one to name a run, and for a granite.build job it
    is the target-run uuid, so neither is treated as secret. A single run node gets
    ``metadata.gb_build_id`` only when the caller passes the space check
    ``GET /lineage/jobs/{id}`` applies to that job (:func:`_can_read_job`) -- an id
    the caller could read there anyway; ``metadata.gb_target_run_uuid`` rides along
    with it for the dashboard's build drawer, not as a gated field. Reading a
    model, pulling a dataset or fetching a step config each needs its own
    authorized call, none of which route through here.

    The walk is bounded: a caller who is not a super admin is held to
    ``depth <= 20`` and the default per-level ceiling
    (:data:`~gbserver.lineage.walk.DEFAULT_MAX_NODES_PER_LEVEL`); a super admin
    may raise ``max_nodes_per_level`` up to the absolute ceiling.

    Filtering it would break the question lineage exists to answer. A chain almost
    always crosses spaces -- a shared curated dataset, a platform-team base model --
    so pruning nodes from spaces the caller cannot read makes "what was my model
    trained on?" silently unanswerable: the graph would look complete while stopping
    at the space boundary. A truthful partial answer is not available here, only a
    misleading one.

    The accepted cost: artifact URIs, job names and statuses are visible across
    spaces. That is
    broader than ``GET /artifacts/``, which narrows to the caller's spaces via
    ``scope_space_name_filter``. Provenance is judged worth that asymmetry -- so if
    an artifact URI or a job name is ever itself a secret, this route is the wrong
    place to keep it.
    """
    if not body.uri and not body.job_id and not body.job_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="One of uri, job_id or job_ids must be provided",
        )
    if not is_super_admin(request):
        if body.max_depth > _USER_MAX_DEPTH:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"depth must be at most {_USER_MAX_DEPTH}",
            )
        if (body.max_nodes_per_level or 0) > DEFAULT_MAX_NODES_PER_LEVEL:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    f"max_nodes_per_level must be at most {DEFAULT_MAX_NODES_PER_LEVEL}"
                ),
            )
    service = _get_index_service()

    try:
        result = service.query_graph(
            uri=body.uri,
            job_id=body.job_id,
            job_ids=body.job_ids,
            direction=body.direction,
            max_depth=body.max_depth,
            max_nodes_per_level=body.max_nodes_per_level,
            group_runs=body.group_runs,
            authorize_job=lambda entry: _can_read_job(request, entry),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        ) from e
    except HTTPException:
        # Raised on purpose (e.g. from the authorization callback): not a storage
        # failure, so it keeps its own status.
        raise
    except Exception as e:
        logger.exception("Lineage graph query failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The lineage index could not be read",
        ) from e

    return LineageGraphResponse(
        root_id=result.get("root_id", ""),
        nodes=result.get("nodes", []),
        edges=result.get("edges", []),
        truncated=result.get("truncated", False),
        unexpanded=result.get("unexpanded", 0),
    )


@lineage_api.get("/jobs", tags=["lineage-index"])
def list_lineage_jobs(
    request: Request,
    uri: Optional[str] = None,
    job_id: Optional[str] = None,
    self_loop: bool = False,
    output: Optional[str] = None,
    terminal: Optional[Literal["input", "output"]] = None,
    job_ids: Annotated[Optional[list[str]], Query()] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> LineageJobsResponse:
    """List the job executions matching every given filter, paged.

    Filters AND together. ``uri``, ``job_id`` or ``job_ids`` is required -- the route answers
    422 otherwise: an unfiltered listing would aggregate the whole index.

    - ``uri`` -- jobs that consumed or produced the artifact. The drill-down for a
      graph node's ``run_count``: the graph collapses an artifact's in-place
      rewrites into one node, and this lists them.
    - ``job_id`` -- one execution.
    - ``job_ids`` (repeated) -- those executions: the runs behind a graph node
      grouping jobs with the same inputs and outputs, as its ``jobs_query`` lists
      them. A client pages a long list itself, a slice per request.
    - ``self_loop`` (with ``uri``) -- only the jobs that rewrote the artifact in
      place (same source and target): the runs behind its looped graph node.
    - ``output`` (with ``uri``) -- only the jobs that read ``uri`` and wrote
      ``output``: the runs behind a graph node grouping same-signature jobs.
    - ``terminal`` (with ``uri``) -- ``input``: only the jobs that wrote ``uri`` with
      no recorded input; ``output``: only those that read it with no recorded
      output. The runs behind a grouped node with one empty side.

    Each entry carries only the global fields (``job_id``, ``source_system``,
    ``status``, ``started_at``, ``inputs``, ``outputs`` and the trimmed ``job``
    group; see ``db_responses.GLOBAL_JOB_FIELDS``): owner, namespace, tags and origin
    ids are returned by ``GET /lineage/jobs/{id}`` behind the space check. A tag
    search is ``POST /lineage/search``.

    ``limit`` is ``1..1000``, ``offset`` ``>= 0`` and ``job_ids`` at most 1000 ids
    -- the cap ``LineageQueryRequest`` applies to the graph (422 otherwise). A storage
    failure is a 503, never an empty page that would read as "no jobs".

    A GET because every filter is a scalar or a flat list; there is nothing a body
    would carry that a query string cannot, and a GET stays linkable and cacheable.

    Paged rather than capped, unlike the graph: a flat list has no shape to
    preserve. ``total`` is the unpaged count of distinct jobs.

    Reads the lineage index directly, whatever the configured provider. Cross-space
    by the same decision as the graph -- see :func:`query_lineage_graph` for why a
    lineage answer is not filtered per space, and what that costs.
    """
    if not uri and not job_id and not job_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="One of 'uri', 'job_id' or 'job_ids' is required",
        )
    if job_ids and len(job_ids) > _MAX_JOB_IDS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"job_ids must list at most {_MAX_JOB_IDS} ids",
        )
    service = _get_index_service()

    try:
        result = service.list_jobs(
            job_ids=job_ids,
            uri=uri,
            job_id=job_id,
            limit=limit,
            offset=offset,
            self_loop=self_loop,
            output=output,
            terminal=terminal,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e)
        ) from e
    except HTTPException:
        # Raised on purpose (e.g. from the authorization callback): not a storage
        # failure, so it keeps its own status.
        raise
    except Exception as e:
        logger.exception("Lineage job listing failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The lineage index could not be read",
        ) from e
    return LineageJobsResponse(**result)


@lineage_api.get("/jobs/{job_id:path}", tags=["lineage-index"])
def get_lineage_job_detail(request: Request, job_id: str) -> LineageJobDetail:
    """One job execution with its full content, read from this server's builds.

    The listing (``GET /jobs``) carries only what the index rows hold. When the rows
    name a granite.build build, its build, target run and steps are read from the
    gb DB, whatever store indexed the job; no external system (W&B) is contacted.

    404 only when the index has no such job. A job that cannot be read further (no
    build recorded, build or target not on this server, no read access) still returns 200,
    with ``detail_available`` false and ``detail_error`` saying why. A build's step
    configs and target run are returned only to callers who may read that build.
    """
    detail = _get_index_service().get_job_detail(
        job_id,
        authorize_build=lambda build: authorize_build_read_access(request, build),
        authorize_entry=lambda entry: _can_read_job(request, entry),
    )
    if detail is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found in the lineage index",
        )
    return LineageJobDetail(**detail)


def _can_read_job(request: Request, entry: dict) -> bool:
    """Whether the caller may read a job's space-gated fields, given its entry.

    The one access rule for a granite.build job beyond the global trace: space
    membership off the job's owner and ``space_name`` (the namespace's first
    segment). ``GET /lineage/jobs/{id}`` gates its detail on it and
    ``POST /lineage/graph`` gates the build and target-run ids of a run node on it,
    through this one function so the two cannot drift. A job with no namespace (an
    import) fails closed.
    """
    return has_space_member_access(
        request,
        username_on_target=entry.get("owner", ""),
        space_name=entry.get("space_name", ""),
    )[0]
