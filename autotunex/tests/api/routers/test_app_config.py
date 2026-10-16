"""Tests for the read-only app-config endpoint."""

from __future__ import annotations

from fastapi import FastAPI
from httpx import AsyncClient

from autotunex.core.config import Settings, get_settings


async def test_app_config_reports_dataset_upload_defaults(client: AsyncClient) -> None:
    response = await client.get("/api/v1/app-config")

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_upload"] == {
        "max_bytes": 5 * 1024**3,
        "client_gzip_enabled": True,
        "client_gzip_min_bytes": 1024**2,
        "client_parquet_preview_max_bytes": 100 * 1024**2,
    }


async def test_app_config_reflects_overridden_settings(
    app: FastAPI, client: AsyncClient, settings: Settings
) -> None:
    """Override only get_settings to test overridden values.

    Clearing all overrides would also drop the client fixture's get_session
    override and fall back to a real engine.
    """
    overridden = settings.model_copy(
        update={
            "dataset_upload_max_bytes": 10 * 1024**3,
            "dataset_client_gzip_enabled": False,
            "dataset_client_gzip_min_bytes": 2048,
            "dataset_client_parquet_preview_max_bytes": 500,
        }
    )

    app.dependency_overrides[get_settings] = lambda: overridden

    response = await client.get("/api/v1/app-config")

    assert response.status_code == 200
    assert response.json()["dataset_upload"] == {
        "max_bytes": 10 * 1024**3,
        "client_gzip_enabled": False,
        "client_gzip_min_bytes": 2048,
        "client_parquet_preview_max_bytes": 500,
    }


async def test_app_config_requires_no_authentication(client: AsyncClient) -> None:
    """No Authorization header, no cookie — matches /health's unauthenticated shape."""
    response = await client.get("/api/v1/app-config")

    assert response.status_code == 200
