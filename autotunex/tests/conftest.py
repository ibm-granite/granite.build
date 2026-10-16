"""Shared test fixtures.

Tests run against a real in-memory SQLite database rather than a mocked
repository, so the ORM mappings and JSON round-trips are genuinely exercised.
Each test gets a fresh schema.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import ConnectionPoolEntry, StaticPool

from autotunex.api.deps import get_principal, get_session
from autotunex.core.config import AuthProviderName, Settings, get_settings
from autotunex.db.session import create_schema
from autotunex.db.tables import ConfigurationTable, DatasetTable, JobTable, UserTable
from autotunex.main import create_app
from autotunex.models.auth import Principal
from autotunex.models.status import DatasetStatus, RunStatus

API = "/api/v1"
"""Prefix the job endpoints are mounted under in tests."""


def make_settings(
    *,
    standalone_email: str | None = None,
    standalone_role: str = "admin",
    auth_providers: list[AuthProviderName] | None = None,
    api_keys: dict[str, str] | None = None,
    auto_provision_users: bool = False,
    login_activity_throttle_minutes: int = 15,
    oidc_issuer: str | None = None,
    oidc_jwks_uri: str | None = None,
    oidc_audience: str | None = None,
    oidc_client_id: str | None = None,
    oidc_client_secret: str | None = None,
    oidc_authorization_endpoint: str | None = None,
    oidc_token_endpoint: str | None = None,
    oidc_end_session_endpoint: str | None = None,
    public_base_url: str | None = None,
    session_secret: str | None = None,
    session_ttl_hours: int = 8,
    session_cookie_same_site: Literal["lax", "none"] = "lax",
    cors_allow_origins: list[str] | None = None,
    dataset_storage_backend: Literal["auto", "local", "huggingface"] = "auto",
    dataset_storage_dir: Path | None = None,
    gb_environment: str | None = None,
    lsf_cluster: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    llm_model: str | None = None,
    llm_timeout_seconds: float = 30.0,
    llm_max_retries: int = 2,
    llm_max_sample_bytes: int = 8000,
) -> Settings:
    """Build test settings a developer's own ``.env`` cannot influence.

    The single factory every test builds a ``Settings`` from. Four near-identical
    copies of this used to live in ``tests/conftest.py``,
    ``tests/api/test_app_wiring.py``, ``tests/api/test_route_coverage.py`` and
    ``tests/api/test_errors.py``, and they had already drifted apart — which is
    how the auth fields came to be pinned in none of them.

    Two independent levers, and both are needed. ``_env_file=None`` stops
    pydantic-settings reading the repository's ``.env`` file at all; pinning each
    field additionally beats an exported ``AUTOTUNEX_`` environment variable,
    which ``_env_file`` does not affect.

    The three auth fields are pinned for the same reason ``database_url`` is. A
    developer who follows ``README.md`` and puts ``AUTOTUNEX_STANDALONE_EMAIL``
    and ``AUTOTUNEX_STANDALONE_ROLE=user`` in ``.env`` otherwise silently
    re-scopes every job test to one non-admin caller, and the resulting
    ``assert 0 == 1`` and missing-``config_snapshot`` failures point nowhere near
    the cause. CI never noticed because CI has no ``.env``.

    ``auto_create_schema=False`` matters even for apps that never touch the
    database: keeping every app built in a test inert about schema creation is
    the habit that stops a later test from writing to the real
    ``./autotunex.db``.

    Args:
        standalone_email: Narrow the standalone principal to one email. The
            default ``None`` is the unrestricted-admin case most tests want.
        standalone_role: The role a narrowed standalone principal carries.
        auth_providers: Which credential kinds ``Settings.auth_providers``
            accepts. Defaults to ``["disabled"]``, matching every existing
            caller of this factory.
        api_keys: SHA-256 hex digest of a key -> the owner's email, for the
            ``"api_key"`` provider. Defaults to ``{}``, matching every
            existing caller of this factory.
        auto_provision_users: Just-in-time create a ``users`` row for a
            verified-email caller with no matching row. Defaults to ``False``,
            matching ``Settings``'s own default, so existing callers are
            unaffected.
        login_activity_throttle_minutes: How stale ``users.last_login_at`` must
            be before an authenticated request refreshes it. Defaults to
            ``Settings``'s own 15, so existing callers are unaffected; a test
            that cares passes ``0`` to record unconditionally.
        oidc_issuer: The issuer a bearer token must carry, for the ``"oidc"``
            provider. Defaults to ``None``, matching every existing caller.
        oidc_jwks_uri: Where the issuer's signing keys would be fetched from,
            for the ``"oidc"`` provider. Defaults to ``None``. A test enabling
            ``"oidc"`` still must not let this be fetched — replace
            ``app.state.authenticator`` with a stubbed key resolver rather
            than relying on this URI ever resolving.
        oidc_audience: The audience a bearer token must carry, for the
            ``"oidc"`` provider. Defaults to ``None``. ``Settings``'s own
            validator requires all three ``oidc_*`` values together the
            moment ``"oidc"`` is in ``auth_providers`` — startup refuses
            otherwise — so a caller that sets one must set all three.
        oidc_client_id: The BFF's own OIDC client id, for the ``"session"``
            provider. Defaults to ``None``, matching every existing caller.
        oidc_client_secret: The BFF's OIDC client secret, for the
            ``"session"`` provider. Defaults to ``None``. Passed as a plain
            ``str``; ``Settings`` wraps it in ``SecretStr`` itself.
        oidc_authorization_endpoint: Where ``/auth/login`` redirects the
            browser, for the ``"session"`` provider. Defaults to ``None``.
        oidc_token_endpoint: Where ``/auth/callback`` exchanges the
            authorization code, for the ``"session"`` provider. Defaults to
            ``None``.
        oidc_end_session_endpoint: Where ``/auth/logout`` redirects to end the
            upstream session, for the ``"session"`` provider. Defaults to
            ``None`` — no rule requires it even when ``"session"`` is enabled.
        public_base_url: Builds ``redirect_uri``, for the ``"session"``
            provider. Defaults to ``None``.
        session_secret: Signs the session JWT, for the ``"session"``
            provider. Defaults to ``None``. ``Settings`` enforces a
            32-character minimum (RFC 7518 §3.2's own floor for an HS256
            key) the moment this is not ``None`` — a caller enabling
            ``"session"`` must pass something at least that long, or
            construction raises before a test body ever runs.
        session_ttl_hours: How long a minted session cookie remains valid.
            Defaults to ``8``, matching ``Settings``'s own default, so a
            caller that never passes this sees no behaviour change.
        session_cookie_same_site: The session cookie's ``SameSite``
            attribute. Defaults to ``"lax"``, matching ``Settings``'s own
            default.
        cors_allow_origins: Origins ``CORSMiddleware`` allows. Defaults to
            ``None``, resolved below to ``[]`` — matching ``Settings``'s own
            default — so a caller that never passes this sees no behaviour
            change.
        dataset_storage_backend: Which dataset storage backend to force.
            Defaults to ``"auto"``, matching ``Settings``'s own default, so
            existing callers see no behaviour change.
        dataset_storage_dir: Root for persisted local dataset files. Defaults
            to ``None``, in which case ``Settings``'s own default
            (``Path("artifacts/datasets")``) is left untouched — only override
            with a tmp dir when a test would otherwise write real files.
        gb_environment: granite.build's own (unprefixed) environment name, e.g.
            "standalone". Defaults to ``None`` and is pinned explicitly, so an
            exported ``GB_ENVIRONMENT`` cannot leak into a test.
        lsf_cluster: SkyPilot/LSF cluster name. Defaults to ``None`` (the
            same-host bash path); set it to select the remote LSF variant.
        llm_base_url: OpenAI-compatible gateway base URL, for the LLM
            intelligence feature. Defaults to ``None``, matching ``Settings``'s
            own default (feature disabled).
        llm_api_key: Bearer token for the gateway. Defaults to ``None``. Passed
            as a plain ``str``; ``Settings`` wraps it in ``SecretStr`` itself.
            ``Settings._validate_llm`` requires all three of ``llm_base_url``,
            ``llm_api_key`` and ``llm_model`` together or none — a caller that
            sets one must set all three.
        llm_model: Model name passed through to the gateway. Defaults to
            ``None``, matching ``Settings``'s own default (no provider-model
            default, deliberately).
        llm_timeout_seconds: Per-call HTTP timeout for a single chat
            completion. Defaults to ``30.0``, matching ``Settings``'s own
            default.
        llm_max_retries: Bounded parse-strategy self-correction retries.
            Defaults to ``2``, matching ``Settings``'s own default.
        llm_max_sample_bytes: Cap on sample bytes sent to the LLM. Defaults to
            ``8000``, matching ``Settings``'s own default.
    """
    resolved_auth_providers: list[AuthProviderName] = (
        auth_providers if auth_providers is not None else ["disabled"]
    )
    return Settings(
        _env_file=None,
        environment="test",
        database_url="sqlite+aiosqlite:///:memory:",
        auto_create_schema=False,
        log_level="WARNING",
        auth_providers=resolved_auth_providers,
        api_keys=api_keys if api_keys is not None else {},
        auto_provision_users=auto_provision_users,
        login_activity_throttle_minutes=login_activity_throttle_minutes,
        standalone_email=standalone_email,
        standalone_role=standalone_role,
        oidc_issuer=oidc_issuer,
        oidc_jwks_uri=oidc_jwks_uri,
        oidc_audience=oidc_audience,
        oidc_client_id=oidc_client_id,
        oidc_client_secret=oidc_client_secret,
        oidc_authorization_endpoint=oidc_authorization_endpoint,
        oidc_token_endpoint=oidc_token_endpoint,
        oidc_end_session_endpoint=oidc_end_session_endpoint,
        public_base_url=public_base_url,
        session_secret=session_secret,
        session_ttl_hours=session_ttl_hours,
        session_cookie_same_site=session_cookie_same_site,
        cors_allow_origins=cors_allow_origins if cors_allow_origins is not None else [],
        dataset_storage_backend=dataset_storage_backend,
        # Path("artifacts/datasets") is Settings' own default; only override when a
        # test passes a tmp dir, so no test ever writes real dataset files.
        dataset_storage_dir=(
            dataset_storage_dir if dataset_storage_dir is not None else Path("artifacts/datasets")
        ),
        gb_environment=gb_environment,
        lsf_cluster=lsf_cluster,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_timeout_seconds=llm_timeout_seconds,
        llm_max_retries=llm_max_retries,
        llm_max_sample_bytes=llm_max_sample_bytes,
    )


@pytest.fixture
def settings() -> Settings:
    """Test configuration, independent of the developer's local .env."""
    return make_settings()


@pytest.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    """A fresh in-memory database with the schema applied.

    ``StaticPool`` keeps every connection pointed at the same in-memory
    database; without it each connection would see an empty one.

    The ``connect`` listener is not optional. SQLite enforces foreign keys only
    when asked, per connection — without it ``ON DELETE CASCADE`` and
    ``ON DELETE RESTRICT`` are silently inert and the tests covering them would
    pass vacuously.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_foreign_keys(
        dbapi_connection: DBAPIConnection, _record: ConnectionPoolEntry
    ) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    await create_schema(engine)
    yield engine
    await engine.dispose()


@pytest.fixture
async def session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """A session bound to the test database."""
    factory = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    async with factory() as session:
        yield session


@pytest.fixture
def app(session: AsyncSession, settings: Settings) -> Iterator[FastAPI]:
    """The application under test, with test settings and database wired in.

    Extracted from ``client`` because several auth tests need the app itself,
    not just an HTTP client onto it: ``app.state.authenticator`` is built once
    in ``create_app`` (so that ``PyJWKClient`` can cache signing keys on the
    instance) and is therefore not something ``dependency_overrides`` can
    intercept. Reaching it through ``client._transport.app`` works but pins the
    test suite to httpx's private attribute names.
    """
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_session] = lambda: session

    yield app

    app.dependency_overrides.clear()


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """An HTTP client wired to the app.

    ``ASGITransport`` calls the app directly, so no port is bound and the
    lifespan hook (which would create the real database) does not run.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.fixture
def as_principal(app: FastAPI) -> Callable[[Principal], None]:
    """Override the resolved principal for the rest of a test.

    ``get_principal`` is a plain dependency, so overriding it short-circuits
    both stages at once — which is what makes a scoping test independent of
    which credential kind is configured.
    """

    def _override(principal: Principal) -> None:
        app.dependency_overrides[get_principal] = lambda: principal

    return _override


@pytest.fixture
async def user(session: AsyncSession) -> UserTable:
    """A persisted owner for the rest of the graph."""
    user = UserTable(id=uuid4(), email="tester@example.com", role="user")
    session.add(user)
    await session.commit()
    return user


@pytest.fixture
async def configuration(session: AsyncSession, user: UserTable) -> ConfigurationTable:
    """A persisted configuration owned by ``user``."""
    configuration = ConfigurationTable(
        id=uuid4(),
        user_id=str(user.id),
        name="lora-sweep",
        tuner_type="optuna",
        rl_tuner_type=None,
        config_data={"learning_rate": {"kind": "float", "low": 1e-6, "high": 1e-3, "log": True}},
    )
    session.add(configuration)
    await session.commit()
    return configuration


@pytest.fixture
async def dataset(session: AsyncSession, user: UserTable) -> DatasetTable:
    """A persisted dataset owned by ``user``."""
    dataset = DatasetTable(
        id=uuid4(), user_id=str(user.id), name="alpaca", description="Instruction data."
    )
    session.add(dataset)
    await session.commit()
    return dataset


@pytest.fixture
async def ready_dataset(session: AsyncSession, user: UserTable) -> DatasetTable:
    """A persisted dataset already in the ``ready`` state."""
    dataset = DatasetTable(
        id=uuid4(),
        user_id=str(user.id),
        name="ready-ds",
        description="Ready data.",
        data_format="jsonl",
        status=DatasetStatus.READY,
        train_records=100,
        train_file_size=2048,
    )
    session.add(dataset)
    await session.commit()
    await session.refresh(dataset, ["train_file", "validation_file"])
    return dataset


@pytest.fixture
async def job_referencing_dataset(
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    dataset: DatasetTable,
) -> JobTable:
    """A job that references ``dataset`` (for associated-jobs and RESTRICT tests)."""
    job = JobTable(
        id=uuid4(),
        user_id=str(user.id),
        status=RunStatus.RUNNING,
        config_id=configuration.id,
        dataset_id=dataset.id,
        model="ibm-granite/granite-3.0-2b-instruct",
        model_source="huggingface",
        experiment_name="uses-dataset",
    )
    session.add(job)
    await session.commit()
    return job


@pytest.fixture
async def job(
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    dataset: DatasetTable,
) -> JobTable:
    """A persisted job with no trials or tasks.

    Tests that need trials, results or tasks add them, so each test states the
    shape it actually depends on rather than inheriting a large fixture.
    """
    job = JobTable(
        id=uuid4(),
        user_id=str(user.id),
        status=RunStatus.PENDING,
        seed=42,
        config_id=configuration.id,
        dataset_id=dataset.id,
        model="ibm-granite/granite-3.0-2b-instruct",
        model_source="huggingface",
        experiment_name="granite-lora-jul",
        tuning_type="lora",
    )
    session.add(job)
    await session.commit()
    await session.refresh(job, ["configuration"])
    return job
