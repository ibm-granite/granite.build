# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Job endpoints.

``GET /jobs`` and ``GET /jobs/{id}`` report what exists; ``POST /jobs`` submits
a new one, owned by the calling principal — see the design spec for why the
read path is shaped like the ``autotunex_jobs`` view without reading it.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Query, Response
from fastapi.responses import StreamingResponse

from autotunex.api.deps import (
    AssetServiceDep,
    EstimationServiceDep,
    JobServiceDep,
    LogServiceDep,
    MetricsServiceDep,
    OnDemandReconcilerDep,
    RewardToolsServiceDep,
    TrialServiceDep,
)
from autotunex.models.asset import AssetSummary
from autotunex.models.common import DataScope, Page, ProblemDetail
from autotunex.models.estimation import EstimateUsagesRequest, EstimateUsagesResponse
from autotunex.models.job import JobCreate, JobDetail, JobRead, JobShape, JobSummary
from autotunex.models.log import LogPage
from autotunex.models.metric import MetricPage
from autotunex.models.reward import GenerateTestSolutionsRequest, GenerateTestSolutionsResponse
from autotunex.models.trial import TrialRead

router = APIRouter(prefix="/jobs", tags=["jobs"])

_PROBLEM_RESPONSE = {
    "model": ProblemDetail,
    "content": {"application/problem+json": {}},
}
_AUTH_RESPONSES: dict[int | str, dict[str, Any]] = {
    HTTPStatus.UNAUTHORIZED: _PROBLEM_RESPONSE,
    HTTPStatus.BAD_REQUEST: _PROBLEM_RESPONSE,
}


def _content_disposition(filename: str) -> str:
    """Build an ``attachment`` disposition safe for any filename.

    Emits both a sanitized ASCII ``filename`` (for legacy clients) and an
    RFC 5987 ``filename*`` carrying the exact UTF-8 name, so a filename with
    spaces, quotes, or non-ASCII characters can neither break the header nor be
    silently mangled.
    """
    ascii_name = filename.encode("ascii", "replace").decode("ascii").replace('"', "_")
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"


@router.post(
    "/estimate-usages",
    summary="Estimate resource usage for a tuning run",
    responses={
        HTTPStatus.NOT_FOUND: _PROBLEM_RESPONSE,
        HTTPStatus.UNPROCESSABLE_ENTITY: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def estimate_usages(
    body: EstimateUsagesRequest, service: EstimationServiceDep
) -> EstimateUsagesResponse:
    """Estimate GPU/CPU memory and GPU count for a saved or inline configuration."""
    return await service.estimate(body)


@router.post(
    "/generate-test-solutions",
    summary="Generate sample reward test solutions",
    responses={
        HTTPStatus.SERVICE_UNAVAILABLE: _PROBLEM_RESPONSE,
        HTTPStatus.BAD_GATEWAY: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def generate_test_solutions(
    body: GenerateTestSolutionsRequest, service: RewardToolsServiceDep
) -> GenerateTestSolutionsResponse:
    """LLM-generate one sample solution per prompt to seed reward test cases."""
    return await service.generate_test_solutions(body)


@router.get(
    "/{job_id}",
    summary="Get job status",
    responses={
        HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE,
        HTTPStatus.NOT_FOUND: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def get_job(
    job_id: UUID,
    service: JobServiceDep,
    scope: DataScope = Query(default=DataScope.OWN),
    shape: JobShape = Query(default=JobShape.FULL),
) -> JobRead | JobDetail:
    """Return a job with its current status.

    ``shape=full`` (the default) returns :class:`JobRead`; ``shape=lean`` returns
    :class:`JobDetail`, dropping the nested build ``tasks`` array and the
    ``config_snapshot`` blob — the same shape ``GET /jobs/by-build-id/{build_id}``
    returns. Trials are on neither; they are paged by ``GET /jobs/{id}/trials``.

    The union's member order matters: :class:`JobRead` subclasses
    :class:`JobDetail`, so listing the base first would validate every full
    response down to it and silently strip both fields.
    """
    return await service.get(job_id, scope=scope, shape=shape)


@router.get(
    "/by-build-id/{build_id}",
    summary="Get job status by build id",
    responses={
        HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE,
        HTTPStatus.NOT_FOUND: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def get_job_by_build_id(
    build_id: UUID,
    service: JobServiceDep,
    scope: DataScope = Query(default=DataScope.OWN),
) -> JobDetail:
    """Return a job located by its granite.build build id.

    The static ``by-build-id`` path segment cannot collide with the
    single-segment ``/{job_id}`` route, so route order is irrelevant. Same
    scoping as ``GET /jobs/{id}``.

    Returns the leaner ``JobDetail``: no nested build ``tasks`` array and no
    ``config_snapshot``. A caller that arrived by build id already holds the
    identifier the tasks array exists to expose, and the snapshot embeds the whole
    configuration as it ran. Both are on ``GET /jobs/{id}``, and the job's trials
    are on ``GET /jobs/{id}/trials``. This lookup costs one DB round trip.
    """
    return await service.get_by_build_id(build_id, scope=scope)


@router.get(
    "",
    summary="List jobs",
    responses={HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE, **_AUTH_RESPONSES},
)
async def list_jobs(
    service: JobServiceDep,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    scope: DataScope = Query(default=DataScope.OWN),
    q: str | None = Query(default=None, description="Case-insensitive substring filter"),
) -> Page[JobSummary]:
    """Return one page of jobs, newest first (own jobs unless scope=all)."""
    return await service.list(limit=limit, offset=offset, scope=scope, q=q)


@router.post(
    "",
    summary="Submit a tuning job",
    status_code=HTTPStatus.CREATED,
    responses={
        HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE,
        HTTPStatus.NOT_FOUND: _PROBLEM_RESPONSE,
        HTTPStatus.CONFLICT: _PROBLEM_RESPONSE,
        HTTPStatus.UNPROCESSABLE_ENTITY: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def create_job(body: JobCreate, service: JobServiceDep) -> JobRead:
    """Submit a tuning job owned by the calling principal."""
    return await service.create(body)


@router.post(
    "/{job_id}/reconcile",
    summary="Force-reconcile a job with granite.build",
    responses={
        HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE,
        HTTPStatus.NOT_FOUND: _PROBLEM_RESPONSE,
        HTTPStatus.CONFLICT: _PROBLEM_RESPONSE,
        HTTPStatus.BAD_GATEWAY: _PROBLEM_RESPONSE,
        HTTPStatus.SERVICE_UNAVAILABLE: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def reconcile_job(job_id: UUID, reconciler: OnDemandReconcilerDep) -> JobRead:
    """Force one job to re-sync with granite.build (admin only).

    Admin-gated transitively via the reconciler's reader dependency. Rewrites the
    job's build_status + output artifacts and forces jobs.status to what
    granite.build reports.
    """
    return await reconciler.reconcile(job_id)


@router.post(
    "/{job_id}/cancel",
    summary="Cancel a job",
    responses={
        HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE,
        HTTPStatus.NOT_FOUND: _PROBLEM_RESPONSE,
        HTTPStatus.CONFLICT: _PROBLEM_RESPONSE,
        HTTPStatus.BAD_GATEWAY: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def cancel_job(
    job_id: UUID, service: JobServiceDep, scope: DataScope = Query(default=DataScope.OWN)
) -> JobRead:
    """Cancel a job and drive it to ``terminated``.

    The static ``/{job_id}/cancel`` sub-path cannot collide with ``/{job_id}``, so
    route order is irrelevant. Owner-scoped exactly like ``GET``/``DELETE``.
    """
    return await service.cancel(job_id, scope=scope)


@router.delete(
    "/{job_id}",
    summary="Delete a job",
    status_code=HTTPStatus.NO_CONTENT,
    responses={
        HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE,
        HTTPStatus.NOT_FOUND: _PROBLEM_RESPONSE,
        HTTPStatus.CONFLICT: _PROBLEM_RESPONSE,
        HTTPStatus.BAD_GATEWAY: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def delete_job(
    job_id: UUID, service: JobServiceDep, scope: DataScope = Query(default=DataScope.OWN)
) -> Response:
    """Delete a job, returning an empty ``204``."""
    await service.delete(job_id, scope=scope)
    return Response(status_code=HTTPStatus.NO_CONTENT)


@router.get(
    "/{job_id}/logs",
    summary="Get a job's logs",
    responses={
        HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE,
        HTTPStatus.NOT_FOUND: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def get_job_logs(
    job_id: UUID,
    service: LogServiceDep,
    before_id: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    scope: DataScope = Query(default=DataScope.OWN),
) -> LogPage:
    """Return one keyset page of the job's job-level log lines, newest first."""
    return await service.get_job_logs(job_id, before_id=before_id, limit=limit, scope=scope)


@router.get(
    "/{job_id}/metrics",
    summary="Get a job's per-step training metrics",
    responses={
        HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE,
        HTTPStatus.NOT_FOUND: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def get_job_metrics(
    job_id: UUID,
    service: MetricsServiceDep,
    after_id: int = Query(default=0, ge=0),
    limit: int = Query(default=500, ge=1, le=2000),
    scope: DataScope = Query(default=DataScope.OWN),
) -> MetricPage:
    """Return an ascending keyset page of the job's per-step metrics (all trials)."""
    return await service.get_job_metrics(job_id, after_id=after_id, limit=limit, scope=scope)


@router.get(
    "/{job_id}/result-report",
    summary="List a job's downloadable output assets",
    responses={
        HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE,
        HTTPStatus.NOT_FOUND: _PROBLEM_RESPONSE,
        HTTPStatus.CONFLICT: _PROBLEM_RESPONSE,
        HTTPStatus.BAD_GATEWAY: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def get_result_report(
    job_id: UUID, service: AssetServiceDep, scope: DataScope = Query(default=DataScope.OWN)
) -> list[AssetSummary]:
    """Return the job's output-asset list, computed on read from its artifact source."""
    return await service.list_assets(job_id, scope=scope)


@router.get(
    "/{job_id}/result-report/file",
    summary="Download one of a job's output files",
    responses={
        HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE,
        HTTPStatus.NOT_FOUND: _PROBLEM_RESPONSE,
        HTTPStatus.CONFLICT: _PROBLEM_RESPONSE,
        HTTPStatus.BAD_GATEWAY: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def download_result_file(
    job_id: UUID,
    service: AssetServiceDep,
    path: str = Query(..., min_length=1, description="Relative path from the result-report list"),
    scope: DataScope = Query(default=DataScope.OWN),
) -> StreamingResponse:
    """Stream a single output file as an attachment.

    ``path`` (a query param, so it carries nested ``/`` safely) is the relative
    path reported by the result-report list — not the bare filename, which can
    repeat across directories.
    """
    opened = await service.download_file(job_id, path=path, scope=scope)
    headers = {"Content-Disposition": _content_disposition(opened.filename)}
    if opened.size is not None:
        headers["Content-Length"] = str(opened.size)
    return StreamingResponse(opened.stream, media_type=opened.media_type, headers=headers)


@router.get(
    "/{job_id}/result-report/archive",
    summary="Download all of a job's output files as a ZIP archive",
    responses={
        HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE,
        HTTPStatus.NOT_FOUND: _PROBLEM_RESPONSE,
        HTTPStatus.CONFLICT: _PROBLEM_RESPONSE,
        HTTPStatus.BAD_GATEWAY: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def download_result_archive(
    job_id: UUID, service: AssetServiceDep, scope: DataScope = Query(default=DataScope.OWN)
) -> StreamingResponse:
    """Stream a ZIP of every output file, built on the fly (no temp file)."""
    filename, stream = await service.open_archive(job_id, scope=scope)
    return StreamingResponse(
        stream,
        media_type="application/zip",
        headers={"Content-Disposition": _content_disposition(filename)},
    )


@router.get(
    "/{job_id}/trials",
    summary="List a job's trials",
    responses={
        HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE,
        HTTPStatus.NOT_FOUND: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def list_job_trials(
    job_id: UUID,
    service: TrialServiceDep,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    scope: DataScope = Query(default=DataScope.OWN),
) -> Page[TrialRead]:
    """Return one page of the job's trials, oldest first, each with its metrics.

    Trials live here rather than nested in ``GET /jobs/{job_id}``: they are an
    unbounded child collection, and carrying them on the detail response made
    every read of a job pay for every trial's ``config`` and ``metrics`` blob.
    ``JobRead.num_trials`` reports the configured *budget*, not a row count — this
    page's ``total`` is the count. A visible job with no trials
    yet is an empty page, not a 404.
    """
    return await service.list_trials(job_id, limit=limit, offset=offset, scope=scope)


@router.get(
    "/{job_id}/trials/{trial_id}/logs",
    summary="Get a trial's logs",
    responses={
        HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE,
        HTTPStatus.NOT_FOUND: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def get_trial_logs(
    job_id: UUID,
    trial_id: str,
    service: LogServiceDep,
    before_id: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    scope: DataScope = Query(default=DataScope.OWN),
) -> LogPage:
    """Return one keyset page of the trial's log lines, newest first."""
    return await service.get_trial_logs(
        job_id, trial_id, before_id=before_id, limit=limit, scope=scope
    )


@router.get(
    "/{job_id}/trials/{trial_id}/metrics",
    summary="Get a trial's per-step training metrics",
    responses={
        HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE,
        HTTPStatus.NOT_FOUND: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def get_trial_metrics(
    job_id: UUID,
    trial_id: str,
    service: MetricsServiceDep,
    after_id: int = Query(default=0, ge=0),
    limit: int = Query(default=500, ge=1, le=2000),
    scope: DataScope = Query(default=DataScope.OWN),
) -> MetricPage:
    """Return an ascending keyset page of one trial's per-step metrics."""
    return await service.get_trial_metrics(
        job_id, trial_id, after_id=after_id, limit=limit, scope=scope
    )


@router.get(
    "/{job_id}/gb-logs",
    summary="Get a job's live build (gb) logs",
    responses={
        HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE,
        HTTPStatus.NOT_FOUND: _PROBLEM_RESPONSE,
        HTTPStatus.SERVICE_UNAVAILABLE: _PROBLEM_RESPONSE,
        HTTPStatus.BAD_GATEWAY: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def get_gb_logs(
    job_id: UUID,
    service: LogServiceDep,
    all: bool = Query(default=False, description="Page through all logs, not just the first page"),
    scope: DataScope = Query(default=DataScope.OWN),
) -> list[str]:
    """Return the job's live gb container logs (oldest-first)."""
    return await service.get_gb_logs(job_id, fetch_all=all, scope=scope)
