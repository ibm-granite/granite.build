"""The app can serve a built SPA from ``frontend_dir`` with SPA-fallback routing."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from autotunex.core.config import Settings
from autotunex.main import create_app
from tests.conftest import make_settings


def _settings_with_frontend(frontend_dir: Path) -> Settings:
    return make_settings().model_copy(update={"frontend_dir": frontend_dir})


@pytest.fixture
def built_spa(tmp_path: Path) -> Path:
    """A minimal ``adapter-static`` build: an index shell plus one real asset."""
    build = tmp_path / "build"
    (build / "_app").mkdir(parents=True)
    (build / "index.html").write_text("<html><body>SPA shell</body></html>")
    (build / "_app" / "app.js").write_text("console.log('app');")
    return build


async def test_frontend_root_serves_the_index_shell(built_spa: Path) -> None:
    app = create_app(_settings_with_frontend(built_spa))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/autotune/")

    assert response.status_code == 200
    assert "SPA shell" in response.text


async def test_frontend_serves_a_real_asset(built_spa: Path) -> None:
    app = create_app(_settings_with_frontend(built_spa))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/autotune/_app/app.js")

    assert response.status_code == 200
    assert "console.log" in response.text


async def test_frontend_falls_back_to_index_for_a_client_route(built_spa: Path) -> None:
    app = create_app(_settings_with_frontend(built_spa))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/autotune/jobs/42")

    assert response.status_code == 200
    assert "SPA shell" in response.text


async def test_api_is_unaffected_when_a_frontend_is_mounted(built_spa: Path) -> None:
    app = create_app(_settings_with_frontend(built_spa))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200


async def test_no_frontend_mount_when_dir_is_unset() -> None:
    app = create_app(make_settings())

    assert not any(getattr(route, "name", None) == "frontend" for route in app.routes)
