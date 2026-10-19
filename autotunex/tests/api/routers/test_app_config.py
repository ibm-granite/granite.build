"""Tests for the read-only app-config endpoint."""

from __future__ import annotations

import pytest
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


async def test_app_config_reports_the_hf_import_group(
    app: FastAPI, client: AsyncClient, settings: Settings
) -> None:
    """The SPA gates the whole import feature on this group, so it is contract.

    ``available`` is the frontend's only signal, and it is a *derived* value
    (``hf_import_enabled`` AND the deployment shape), so the response is where it
    has to be pinned -- a settings-level test cannot catch the router reading the
    wrong attribute.
    """
    overridden = settings.model_copy(
        update={
            "hf_import_enabled": True,
            "gb_environment": "standalone",
            "lsf_cluster": None,
            "hf_import_max_bytes": 7 * 1024**3,
            "hf_import_max_rows": 1234,
        }
    )
    app.dependency_overrides[get_settings] = lambda: overridden

    response = await client.get("/api/v1/app-config")

    assert response.status_code == 200
    assert response.json()["hf_import"] == {
        "available": True,
        "max_bytes": 7 * 1024**3,
        "max_rows": 1234,
    }


async def test_app_config_reports_hf_import_unavailable_on_an_lsf_standalone(
    app: FastAPI, client: AsyncClient, settings: Settings
) -> None:
    """``lsf_cluster`` set is standalone, but not the shape import supports.

    That variant emits no dataset locator for the build to mount, so an import
    would succeed and the launch would then fail on a null URI. The operator flag
    being on is not enough, and dropping the ``not self.lsf_cluster`` term would
    leave the SPA offering a feature that cannot work.
    """
    overridden = settings.model_copy(
        update={
            "hf_import_enabled": True,
            "gb_environment": "standalone",
            "lsf_cluster": "my-lsf-queue",
        }
    )
    app.dependency_overrides[get_settings] = lambda: overridden

    response = await client.get("/api/v1/app-config")

    assert response.status_code == 200
    assert response.json()["hf_import"]["available"] is False


async def test_app_config_reports_hf_import_unavailable_off_standalone(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cluster that cannot push (no tokens) stores with no locator, so import is unavailable."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GB_TOKEN", raising=False)
    overridden = settings.model_copy(
        update={"hf_import_enabled": True, "gb_environment": "cluster", "lsf_cluster": None}
    )
    app.dependency_overrides[get_settings] = lambda: overridden

    response = await client.get("/api/v1/app-config")

    assert response.status_code == 200
    assert response.json()["hf_import"]["available"] is False


async def test_app_config_requires_no_authentication(client: AsyncClient) -> None:
    """No Authorization header, no cookie — matches /health's unauthenticated shape."""
    response = await client.get("/api/v1/app-config")

    assert response.status_code == 200
