# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Dataset CRUD and upload endpoints.

Datasets are the second resource (after configurations) this API creates,
updates and deletes, plus the repo's first file upload. Every body is one or two
lines: parse, delegate, serialize. The principal is injected into the service
(via ``DatasetServiceDep``), never taken by the router. Upload returns ``202``
because the heavy work runs off-request in the upload runner.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, File, Form, Query, Request, Response, UploadFile

from autotunex.api.deps import DatasetServiceDep
from autotunex.models.common import DataScope, Page, ProblemDetail
from autotunex.models.dataset import DatasetCreate, DatasetRead

router = APIRouter(prefix="/datasets", tags=["datasets"])

_PROBLEM_RESPONSE = {"model": ProblemDetail, "content": {"application/problem+json": {}}}
_AUTH_RESPONSES: dict[int | str, dict[str, Any]] = {
    HTTPStatus.UNAUTHORIZED: _PROBLEM_RESPONSE,
    HTTPStatus.BAD_REQUEST: _PROBLEM_RESPONSE,
}


@router.post(
    "",
    summary="Create a dataset",
    status_code=HTTPStatus.CREATED,
    responses={
        HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE,
        HTTPStatus.CONFLICT: _PROBLEM_RESPONSE,
        HTTPStatus.UNPROCESSABLE_ENTITY: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def create_dataset(body: DatasetCreate, service: DatasetServiceDep) -> DatasetRead:
    """Create a dataset owned by the calling principal."""
    return await service.create(body)


@router.get(
    "",
    summary="List datasets",
    responses={HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE, **_AUTH_RESPONSES},
)
async def list_datasets(
    service: DatasetServiceDep,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    scope: DataScope = Query(default=DataScope.OWN),
    q: str | None = Query(default=None, description="Case-insensitive substring filter"),
) -> Page[DatasetRead]:
    """Return one page of datasets, newest first (own datasets unless scope=all)."""
    return await service.list(limit=limit, offset=offset, scope=scope, q=q)


@router.get(
    "/{dataset_id}",
    summary="Get a dataset",
    responses={
        HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE,
        HTTPStatus.NOT_FOUND: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def get_dataset(
    dataset_id: UUID,
    service: DatasetServiceDep,
    preview: bool = Query(default=False),
    preview_rows: int = Query(default=10, ge=1, le=100),
    scope: DataScope = Query(default=DataScope.OWN),
) -> DatasetRead:
    """Return a single dataset, optionally with a bounded preview."""
    return await service.get(dataset_id, preview=preview, preview_rows=preview_rows, scope=scope)


@router.put(
    "/{dataset_id}",
    summary="Replace a dataset's metadata",
    responses={
        HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE,
        HTTPStatus.NOT_FOUND: _PROBLEM_RESPONSE,
        HTTPStatus.CONFLICT: _PROBLEM_RESPONSE,
        HTTPStatus.UNPROCESSABLE_ENTITY: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def replace_dataset(
    dataset_id: UUID,
    body: DatasetCreate,
    service: DatasetServiceDep,
    scope: DataScope = Query(default=DataScope.OWN),
) -> DatasetRead:
    """Fully replace a dataset's mutable metadata."""
    return await service.update(dataset_id, body, scope=scope)


@router.delete(
    "/{dataset_id}",
    summary="Delete a dataset",
    status_code=HTTPStatus.NO_CONTENT,
    responses={
        HTTPStatus.FORBIDDEN: _PROBLEM_RESPONSE,
        HTTPStatus.NOT_FOUND: _PROBLEM_RESPONSE,
        HTTPStatus.CONFLICT: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def delete_dataset(
    dataset_id: UUID, service: DatasetServiceDep, scope: DataScope = Query(default=DataScope.OWN)
) -> Response:
    """Delete a dataset, returning an empty ``204``."""
    await service.delete(dataset_id, scope=scope)
    return Response(status_code=HTTPStatus.NO_CONTENT)


@router.post(
    "/{dataset_id}/upload",
    summary="Upload a dataset's file",
    status_code=HTTPStatus.ACCEPTED,
    responses={
        HTTPStatus.NOT_FOUND: _PROBLEM_RESPONSE,
        HTTPStatus.CONFLICT: _PROBLEM_RESPONSE,
        HTTPStatus.REQUEST_ENTITY_TOO_LARGE: _PROBLEM_RESPONSE,
        HTTPStatus.UNSUPPORTED_MEDIA_TYPE: _PROBLEM_RESPONSE,
        HTTPStatus.UNPROCESSABLE_ENTITY: _PROBLEM_RESPONSE,
        **_AUTH_RESPONSES,
    },
)
async def upload_dataset(
    dataset_id: UUID,
    service: DatasetServiceDep,
    request: Request,
    train_file: Annotated[UploadFile, File()],
    validation_file: Annotated[UploadFile | None, File()] = None,
    validation_percentage: Annotated[int | None, Form()] = None,
    column_mapping: Annotated[str | None, Form()] = None,
) -> DatasetRead:
    """Accept a dataset file; return ``202`` while the runner processes it."""
    gzip_encoded = request.headers.get("content-encoding", "").lower() == "gzip"
    return await service.upload(
        dataset_id,
        train=train_file,
        validation=validation_file,
        validation_percentage=validation_percentage,
        column_mapping_json=column_mapping,
        gzip_encoded=gzip_encoded,
    )
