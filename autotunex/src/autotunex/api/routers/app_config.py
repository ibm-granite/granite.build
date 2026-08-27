# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Read-only app configuration endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from autotunex.api.deps import SettingsDep
from autotunex.models.app_config import AppConfigResponse, DatasetUploadConfig

router = APIRouter(prefix="/app-config", tags=["meta"])


@router.get("", summary="Read-only frontend-facing app configuration")
async def get_app_config(settings: SettingsDep) -> AppConfigResponse:
    """Return backend-defined values the frontend needs to behave/render correctly.

    Unauthenticated like ``/health``: every value here is non-sensitive
    operational configuration, not user data, and the frontend needs it before
    (and independent of) any dataset-specific request.
    """
    return AppConfigResponse(
        dataset_upload=DatasetUploadConfig(
            max_bytes=settings.dataset_upload_max_bytes,
            client_gzip_enabled=settings.dataset_client_gzip_enabled,
            client_gzip_min_bytes=settings.dataset_client_gzip_min_bytes,
            client_parquet_preview_max_bytes=settings.dataset_client_parquet_preview_max_bytes,
        )
    )
