"""GbServerStatusReader against httpx.MockTransport: 200/404/401/timeout/malformed."""

from __future__ import annotations

from uuid import UUID

import httpx
import pytest

from autotunex.services.reconcile.gbserver import GbServerStatusReader
from autotunex.services.reconcile.protocols import (
    BuildNotFoundError,
    BuildStatusAuthError,
    BuildStatusUnavailableError,
    MalformedBuildStatusError,
)

BUILD_ID = UUID("22222222-2222-2222-2222-222222222222")
TOKEN_ENV = "AUTOTUNEX_GB_TEST_TOKEN"

_OK_BODY = {
    "status": {
        "build": {
            "uuid": str(BUILD_ID),
            "status": "running",
            "failure_reason": None,
            "created_time": "2026-08-07T00:00:00Z",
            "updated_time": "2026-08-07T00:05:00Z",
        },
        "target_runs": [],
    },
    "retry_chain": None,
}


def _reader(handler: object, *, token_env: str = TOKEN_ENV) -> GbServerStatusReader:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return GbServerStatusReader(
        http_client=httpx.AsyncClient(transport=transport),
        base_url="https://gbserver.example",
        token_env=token_env,
    )


async def test_reads_and_normalizes_a_200(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TOKEN_ENV, "gb-secret")
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=_OK_BODY)

    state = await _reader(handler).read(BUILD_ID)

    assert seen["url"] == f"https://gbserver.example/api/v1/builds/{BUILD_ID}/status"
    assert seen["auth"] == "Bearer gb-secret"
    assert state.build_id == BUILD_ID
    assert state.status == "running"
    assert state.created_at == "2026-08-07T00:00:00Z"
    assert state.updated_at == "2026-08-07T00:05:00Z"
    assert state.raw == _OK_BODY


async def test_404_raises_build_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "nope"})

    with pytest.raises(BuildNotFoundError):
        await _reader(handler).read(BUILD_ID)


async def test_401_raises_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    with pytest.raises(BuildStatusAuthError):
        await _reader(handler).read(BUILD_ID)


async def test_timeout_raises_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow", request=request)

    with pytest.raises(BuildStatusUnavailableError):
        await _reader(handler).read(BUILD_ID)


async def test_malformed_body_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": {}})  # no build.status

    with pytest.raises(MalformedBuildStatusError):
        await _reader(handler).read(BUILD_ID)


async def test_no_token_sends_no_auth_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=_OK_BODY)

    await _reader(handler).read(BUILD_ID)

    assert seen["auth"] is None


_EVENTS_BODY = {
    "events": [{"build_event": {"timestamp": "2026-08-07T00:01:00Z", "payload": {"msg": "hello"}}}]
}


async def test_read_events_hits_the_events_endpoint_with_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOKEN_ENV, "gb-secret")
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=_EVENTS_BODY)

    events = await _reader(handler).read_events(BUILD_ID)

    assert seen["url"] == f"https://gbserver.example/api/v1/builds/{BUILD_ID}/events"
    assert seen["auth"] == "Bearer gb-secret"
    assert events == _EVENTS_BODY


async def test_read_events_404_raises_build_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "nope"})

    with pytest.raises(BuildNotFoundError):
        await _reader(handler).read_events(BUILD_ID)


async def test_read_events_timeout_raises_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow", request=request)

    with pytest.raises(BuildStatusUnavailableError):
        await _reader(handler).read_events(BUILD_ID)
