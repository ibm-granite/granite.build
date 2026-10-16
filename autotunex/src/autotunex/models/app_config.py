# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Read-only, frontend-facing application configuration."""

from __future__ import annotations

from pydantic import BaseModel


class DatasetUploadConfig(BaseModel):
    """Dataset-upload knobs the frontend needs to behave and render correctly."""

    max_bytes: int
    client_gzip_enabled: bool
    client_gzip_min_bytes: int
    client_parquet_preview_max_bytes: int


class AppConfigResponse(BaseModel):
    """Extensible, read-only app configuration surface for the frontend.

    Grouped by domain (``dataset_upload`` today) so a future domain's knobs
    can be added as a new group without breaking existing readers — the
    frontend reads each group by key, not a flat, ever-growing field list.
    """

    dataset_upload: DatasetUploadConfig
