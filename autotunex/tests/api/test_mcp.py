"""Tests for the opt-in FastMCP server (:mod:`autotunex.api.mcp`).

Guarded by ``pytest.importorskip("fastmcp")`` — the base test suite must not
require the ``[mcp]`` extra. fastmcp's in-memory ``Client``/``FastMCPTransport``
has no ASGI ``Request`` at all (confirmed empirically while building
``mcp.py`` — see that module's docstring), so an authenticated *tool call*
cannot be exercised through it for this fastmcp version. These tests split
the same way that module's docstring promises: tool *registration* is
verified through the in-memory client, and the per-request authentication
gate (:func:`resolve_mcp_principal`) is tested directly — mirroring the
pattern ``tests/api/test_deps.py`` already uses for the HTTP authenticator's
two stages.
"""

from __future__ import annotations

import hashlib

import pytest

fastmcp = pytest.importorskip("fastmcp")

from fastapi import FastAPI  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker  # noqa: E402

from autotunex.api.mcp import build_mcp, mount_mcp, resolve_mcp_principal  # noqa: E402
from autotunex.core.auth.registry import build_authenticator  # noqa: E402
from autotunex.core.exceptions import InvalidCredentialsError, MissingCredentialsError  # noqa: E402
from autotunex.db.tables import UserTable  # noqa: E402
from autotunex.services.chat.tools import TOOLS  # noqa: E402
from tests.conftest import make_settings  # noqa: E402


def _digest(raw_key: str) -> str:
    """SHA-256 hex digest, matching ``ApiKeyVerifier``'s own hashing."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


@pytest.fixture
def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """A raw session factory bound to the test database.

    ``resolve_mcp_principal`` opens its own short-lived session per call (it
    takes a ``session_factory``, mirroring ``api/deps.get_session``) rather
    than a single already-open ``AsyncSession``, so tests need the factory
    itself — the ``session`` fixture's one already-opened session will not do.
    """
    return async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)


async def test_build_mcp_registers_every_tool_in_the_registry() -> None:
    """Every entry in the shared registry becomes a discoverable MCP tool."""
    mcp = build_mcp(make_settings())

    async with fastmcp.Client(mcp) as client:
        tools = await client.list_tools()

    assert {tool.name for tool in tools} == set(TOOLS.keys())


async def test_build_mcp_derives_a_schema_with_correct_required_and_defaulted_fields() -> None:
    """The __signature__ override (see mcp.py's second docstring bullet) must survive intact.

    ``get_job`` has one required field; ``start_tuning_job`` has a field with a
    default (``model_source``), which must NOT be reported as required.
    """
    mcp = build_mcp(make_settings())

    async with fastmcp.Client(mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}

    get_job_schema = tools["get_job"].inputSchema
    assert get_job_schema["required"] == ["job_id"]
    assert get_job_schema["properties"]["job_id"]["description"] == "The job's UUID."

    start_tuning_schema = tools["start_tuning_job"].inputSchema
    assert "model_source" not in start_tuning_schema.get("required", [])
    assert set(start_tuning_schema["required"]) == {
        "config_id",
        "dataset_id",
        "model",
        "experiment_name",
    }


async def test_build_mcp_reports_a_tool_with_no_arguments_as_taking_no_arguments() -> None:
    """``_NoArgs``-backed tools (e.g. list_jobs) must round-trip to an empty parameter list."""
    mcp = build_mcp(make_settings())

    async with fastmcp.Client(mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}

    assert tools["list_jobs"].inputSchema.get("required", []) == []
    assert tools["list_jobs"].inputSchema.get("properties", {}) == {}


async def test_resolve_mcp_principal_accepts_a_valid_api_key(
    session_factory: async_sessionmaker[AsyncSession], user: UserTable
) -> None:
    raw_key = "s3cr3t-mcp-key"
    settings = make_settings(auth_providers=["api_key"], api_keys={_digest(raw_key): user.email})
    authenticator = build_authenticator(settings)

    principal = await resolve_mcp_principal(
        api_key=raw_key,
        authenticator=authenticator,
        session_factory=session_factory,
        settings=settings,
    )

    assert principal.email == user.email
    assert principal.user_id == user.id
    assert principal.is_admin is False


async def test_resolve_mcp_principal_rejects_a_missing_api_key(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    settings = make_settings(
        auth_providers=["api_key"], api_keys={_digest("configured-key"): "owner@example.com"}
    )
    authenticator = build_authenticator(settings)

    with pytest.raises(MissingCredentialsError):
        await resolve_mcp_principal(
            api_key=None,
            authenticator=authenticator,
            session_factory=session_factory,
            settings=settings,
        )


async def test_resolve_mcp_principal_rejects_an_unrecognized_api_key(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    settings = make_settings(
        auth_providers=["api_key"], api_keys={_digest("the-real-key"): "owner@example.com"}
    )
    authenticator = build_authenticator(settings)

    with pytest.raises(InvalidCredentialsError):
        await resolve_mcp_principal(
            api_key="not-the-real-key",
            authenticator=authenticator,
            session_factory=session_factory,
            settings=settings,
        )


async def test_resolve_mcp_principal_provisions_a_new_user_when_auto_provision_is_enabled(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Mirrors get_principal's JIT-provisioning branch (see api/deps.py)."""
    raw_key = "s3cr3t-mcp-key"
    settings = make_settings(
        auth_providers=["api_key"],
        api_keys={_digest(raw_key): "newcomer@example.com"},
        auto_provision_users=True,
    )
    authenticator = build_authenticator(settings)

    principal = await resolve_mcp_principal(
        api_key=raw_key,
        authenticator=authenticator,
        session_factory=session_factory,
        settings=settings,
    )

    assert principal.user_id is not None
    assert principal.is_admin is False


async def test_resolve_mcp_principal_leaves_user_id_none_with_no_matching_row_and_no_provisioning(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    raw_key = "s3cr3t-mcp-key"
    settings = make_settings(
        auth_providers=["api_key"], api_keys={_digest(raw_key): "ghost@example.com"}
    )
    authenticator = build_authenticator(settings)

    principal = await resolve_mcp_principal(
        api_key=raw_key,
        authenticator=authenticator,
        session_factory=session_factory,
        settings=settings,
    )

    assert principal.user_id is None
    assert principal.is_admin is False


def test_mount_mcp_is_a_noop_when_disabled() -> None:
    """The default (``enable_mcp=False``) must never touch the app or import fastmcp's ASGI bits."""
    app = FastAPI()
    routes_before = list(app.routes)

    mount_mcp(app, make_settings())

    assert list(app.routes) == routes_before


async def test_mount_mcp_serves_the_server_at_exactly_mcp_when_enabled() -> None:
    """End-to-end proof of the mounting pattern documented in mcp.py's last docstring bullet.

    Uses a bare ``FastAPI()`` (not ``create_app()``) and only the MCP protocol
    handshake — calling an actual *tool* through this endpoint would need
    ``get_session_factory()`` (a process-wide singleton) wired to the test
    database, which is out of scope here; the ``resolve_mcp_principal`` tests
    above already cover the authentication gate directly. Drives the app's
    ASGI lifespan by hand (``app.router.lifespan_context(app)``) rather than
    reaching for ``starlette.testclient.TestClient`` — ``httpx.ASGITransport``
    alone never sends lifespan events, and the mounted sub-app's session
    manager needs its ``startup`` run or every request fails with "Task group
    is not initialized" (confirmed while building ``mcp.py``), but the rest of
    this suite's convention (``httpx.AsyncClient`` + ``ASGITransport``, see
    ``conftest.py``'s ``client`` fixture) is otherwise worth keeping.
    """
    app = FastAPI()
    settings = make_settings().model_copy(update={"enable_mcp": True})

    mount_mcp(app, settings)

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://testserver", follow_redirects=True
        ) as client:
            response = await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "0.0"},
                    },
                },
                headers={"Accept": "application/json, text/event-stream"},
            )

    assert response.status_code == 200
