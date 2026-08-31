# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Dependency providers.

Every request-scoped collaborator is built here, so tests can replace any layer
with ``app.dependency_overrides[...]``. Routers depend on these, never on
concrete constructors.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Annotated

import httpx
from fastapi import Cookie, Depends, Request
from fastapi.security import APIKeyCookie, APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.core.auth.disabled import STANDALONE_PROVIDER
from autotunex.core.auth.impersonation import read_assume_token
from autotunex.core.auth.oidc import OidcBearerVerifier
from autotunex.core.auth.protocols import Authenticator
from autotunex.core.config import ADMIN_ROLE, Settings, get_settings
from autotunex.core.exceptions import (
    AdminRequiredError,
    BuildReconcileUnavailableError,
    LlmNotConfiguredError,
)
from autotunex.db.repositories.protocols import (
    ConfigurationRepository,
    DatasetRepository,
    JobRepository,
    UserRepository,
)
from autotunex.db.repositories.sqlalchemy import (
    SqlAlchemyConfigurationRepository,
    SqlAlchemyDatasetRepository,
    SqlAlchemyJobRepository,
    SqlAlchemyUserRepository,
)
from autotunex.db.session import get_session_factory
from autotunex.models.auth import Principal
from autotunex.services.assets import AssetService
from autotunex.services.autotune import AutotuneCore, AutotuneCoreAdapter
from autotunex.services.chat.memory import ConversationMemory
from autotunex.services.chat.service import ChatService
from autotunex.services.configurations import ConfigurationService
from autotunex.services.dataset_intelligence import DatasetIntelligenceService
from autotunex.services.dataset_runner import DatasetUploadRunner, InProcessDatasetUploadRunner
from autotunex.services.datasets import DatasetService
from autotunex.services.estimation import EstimationService
from autotunex.services.gb_logs.protocols import GbLogReader
from autotunex.services.gb_logs.registry import get_gb_log_reader as build_gb_log_reader
from autotunex.services.jobs import JobService
from autotunex.services.launch.registry import get_build_canceller as build_build_canceller
from autotunex.services.launch.registry import get_tuning_launcher as build_tuning_launcher
from autotunex.services.llm.base import LlmClient
from autotunex.services.llm.registry import get_llm_client as build_llm_client
from autotunex.services.local.runner import LocalJobRunner
from autotunex.services.local.trainer import AutotuneLocalTrainer
from autotunex.services.logs import LogService
from autotunex.services.protocols import JobRunner
from autotunex.services.reconcile.on_demand import OnDemandReconciler
from autotunex.services.reconcile.protocols import BuildStatusReader
from autotunex.services.reconcile.registry import (
    get_build_status_reader as build_status_reader_for,
)
from autotunex.services.reward.protocols import RewardExecutor
from autotunex.services.reward.subprocess_executor import SubprocessRewardExecutor
from autotunex.services.reward.tools import RewardToolsService
from autotunex.services.reward.validation import RewardValidationService
from autotunex.services.runner import InProcessJobRunner, NoOpJobRunner
from autotunex.services.storage.artifacts import FilesystemArtifactLister, HuggingFaceArtifactLister
from autotunex.services.storage.base import StorageBackend
from autotunex.services.storage.registry import get_storage_backend as build_storage_backend
from autotunex.services.users import UserService


async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield a request-scoped database session."""
    factory = get_session_factory()
    async with factory() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_http_client(request: Request) -> httpx.AsyncClient:
    """The shared outbound HTTP client, opened once in ``lifespan``.

    Reused rather than constructed per call so the connection pool (and the
    explicit timeout ``lifespan`` sets) is shared across requests instead of
    paying a new TCP/TLS handshake for every outbound call.
    """
    http_client: httpx.AsyncClient = request.app.state.http_client
    return http_client


def get_id_token_verifier(request: Request) -> OidcBearerVerifier | None:
    """The verifier ``/auth/callback`` uses to check a W3ID ID token.

    ``None`` when the session provider is not enabled.
    """
    id_token_verifier: OidcBearerVerifier | None = request.app.state.id_token_verifier
    return id_token_verifier


bearer_scheme = HTTPBearer(
    auto_error=False,
    scheme_name="BearerAuth",
    description="Bearer token in the Authorization header.",
)
"""Parses ``Authorization: Bearer <token>`` into ``HTTPAuthorizationCredentials``.

``auto_error=False`` is load-bearing, not a style choice: with the default
``True`` this scheme raises its own ``HTTPException`` the instant a bearer
credential is missing or malformed, before the ``Authenticator`` is ever
consulted — bypassing ``core/exceptions.py`` entirely and emitting a
FastAPI-shaped body instead of this API's RFC 9457 ``problem+json``, and
pre-empting both the conflicting-credentials 400 and the deliberate no-403
rule. ``False`` makes a missing or malformed credential read as ``None``
instead, so every accept/reject decision still funnels through
``get_authenticated_principal`` and the ``Authenticator`` below it.

The deleted ``_bearer_token`` helper used to document an RFC 7235 imprecision
it had to normalize by hand: the scheme permits more than one space between
the scheme and the token. FastAPI's internal
``get_authorization_scheme_param`` does that normalization itself —
``scheme, _, param = value.partition(" "); return scheme, param.strip()`` —
so ``"Bearer  tok"`` and ``"Bearer tok "`` both yield ``'tok'`` with no
surrounding whitespace, and a future bearer verifier does not need to
normalize it again. It is now the *library's* behaviour rather than one owned
here. One consequence worth knowing: ``.strip()`` reduces a whitespace-only
token (``"Bearer   "``) to ``''``, which ``HTTPBearer.__call__`` treats as
absent — the same missing-credential outcome as no token at all, and a fourth
instance of the empty-credential-reads-as-absent rule this module and
``tests/api/test_auth.py`` document elsewhere. Pinned there.
"""

api_key_scheme = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    scheme_name="ApiKeyAuth",
    description="Service API key in the X-API-Key header.",
)
"""Parses the ``X-API-Key`` header into a token or ``None``.

See ``bearer_scheme`` for why ``auto_error=False`` is required here too.

Its ``check_api_key`` also normalizes a *present but empty* header value to
``None`` (``if not api_key: return None``) rather than passing ``""`` through.
That is the library's behaviour, not something implemented in this module, and
it is the behaviour this project wants: an empty credential must read as no
credential, so the caller gets the missing-credential 401 and challenge rather
than "invalid token" — the same rule the deleted ``_bearer_token`` helper used
to enforce for the bearer slot alone. Before this scheme swap, the plain
``Header()``/``Cookie()`` parameters it replaced did not apply that rule here —
an empty string reached ``RoutingAuthenticator.authenticate``, which treats
"not None" as "presented" — so this scheme did not introduce an inconsistency
between transports, it removed one. Pinned in ``tests/api/test_auth.py``.
"""

session_scheme = APIKeyCookie(
    name="session",
    auto_error=False,
    scheme_name="SessionCookieAuth",
    description="Session identifier in the session cookie.",
)
"""Parses the ``session`` cookie into a token or ``None``.

See ``bearer_scheme`` for why ``auto_error=False`` is required here too, and
``api_key_scheme`` for why an empty cookie value normalizing to ``None`` (the
same ``check_api_key`` the header scheme uses) is wanted, not merely inherited.
"""


async def get_authenticated_principal(
    request: Request,
    bearer: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)] = None,
    x_api_key: Annotated[str | None, Depends(api_key_scheme)] = None,
    session: Annotated[str | None, Depends(session_scheme)] = None,
) -> Principal:
    """Stage one: turn this request's credentials into a ``Principal``.

    Depends on the three ``fastapi.security`` classes above, rather than plain
    ``Header``/``Cookie`` parameters, purely so FastAPI can recognise this as
    authentication and publish it under ``components.securitySchemes`` in the
    generated OpenAPI schema — the only thing that changes is what
    ``/openapi.json`` and ``/docs`` advertise. Each scheme is ``auto_error=False``
    so a missing, malformed, or unrecognised-scheme credential still reads as
    ``None`` here instead of short-circuiting through the scheme's own
    ``HTTPException``; the accept/reject decision, and its RFC 9457 shape,
    stays entirely with the ``Authenticator``.

    No database access — the app's ``Authenticator`` (built once in
    ``create_app`` and stored on ``app.state``, never per request) does the
    verifying and raises an authentication exception if nothing matches.
    """
    authenticator: Authenticator = request.app.state.authenticator
    token = bearer.credentials if bearer is not None else None
    return await authenticator.authenticate(bearer=token, api_key=x_api_key, session=session)


def get_user_repository(session: SessionDep) -> UserRepository:
    """Provide the user repository implementation."""
    return SqlAlchemyUserRepository(session)


async def get_principal(
    authenticated: Annotated[Principal, Depends(get_authenticated_principal)],
    user_repository: Annotated[UserRepository, Depends(get_user_repository)],
    settings: SettingsDep,
) -> Principal:
    """Stage two: resolve ``user_id`` and ``is_admin`` against ``users``.

    A ``None`` email (unrestricted standalone) has nothing to resolve. For a
    narrowed standalone principal, ``is_admin`` came from ``standalone_role``
    and must not be overwritten by whatever the matched row's ``role`` says —
    the setting wins so a developer can run as non-admin even if their real
    row disagrees. Every other provider resolves ``is_admin`` from the row,
    and an authenticated-but-unprovisioned caller (no matching row) resolves
    to ``user_id=None, is_admin=False`` — the empty-page case ``JobService``
    must guard against.

    When ``settings.auto_provision_users`` is on, that last case instead creates
    the row just-in-time (see ``SqlAlchemyUserRepository.provision``), so a
    first-time caller with a verified email owns what it creates rather than
    getting a 403. Provisioning runs only for a resolvable email and only after a
    miss, so the unrestricted-standalone (``email=None``) path never provisions.
    The standalone provider additionally always provisions on a miss,
    independent of ``auto_provision_users`` — it has no auth to police, so the
    flag (an authorization policy for *real* providers) does not apply to it,
    and the default owner must exist for no-auth mode's writes to succeed.
    Crucially, ``is_admin`` is still derived below from the *row's* ``role`` — a
    freshly provisioned row is ``role='user'``, so JIT never grants admin.

    The ``role == ADMIN_ROLE`` half of that is the branch's central authorization
    decision: being an admin no longer removes the ownership filter by itself —
    it grants the ability to request the cross-user view via ``?scope=all``,
    resolved in ``autotunex.services.scoping.resolve_owner_filter``, which
    hands back the unfiltered ``owner_id=None`` only to an admin who asked for
    it. Existence of a row must never be mistaken for admin-ness here.
    """
    if authenticated.email is None:
        return authenticated
    user = await user_repository.get_by_email(authenticated.email)
    # The standalone owner must exist for no-auth mode to function, so it is
    # provisioned regardless of auto_provision_users — that flag is an
    # authorization policy for *real* providers, and standalone has no auth to
    # police. Real providers still respect it.
    should_provision = (
        settings.auto_provision_users or authenticated.provider == STANDALONE_PROVIDER
    )
    if user is None and should_provision:
        user = await user_repository.provision(authenticated.email)
    if authenticated.provider == STANDALONE_PROVIDER:
        is_admin = authenticated.is_admin
    else:
        is_admin = user is not None and user.role == ADMIN_ROLE
    return authenticated.model_copy(
        update={"user_id": user.id if user is not None else None, "is_admin": is_admin}
    )


async def get_effective_principal(
    real: Annotated[Principal, Depends(get_principal)],
    user_repository: Annotated[UserRepository, Depends(get_user_repository)],
    settings: SettingsDep,
    assume_cookie: Annotated[str | None, Cookie(alias="autotunex_assume")] = None,
) -> Principal:
    """Apply an admin's active impersonation overlay on top of the real principal.

    Every guard falls back to returning ``real`` unchanged, so the *only* way the
    overlay takes effect is a genuinely-admin caller presenting a valid, unexpired
    ``autotunex_assume`` cookie that points at an existing user other than
    themselves. ``real.is_admin`` is the security gate — it is recomputed from the
    DB row each request by ``get_principal``, so an admin demoted mid-session (or
    any non-admin presenting a forged cookie) loses the overlay immediately.
    ``is_admin`` is copied from ``real`` (preserved): an admin keeps admin powers
    while acting as the target, matching the ported feature's semantics.
    """
    if assume_cookie is None:
        return real
    if not real.is_admin:
        return real
    if settings.session_secret is None:
        return real
    target_id = read_assume_token(assume_cookie, secret=settings.session_secret.get_secret_value())
    if target_id is None:
        return real
    target = await user_repository.get(target_id)
    if target is None or target.id == real.user_id:
        return real
    return real.model_copy(
        update={
            "email": target.email,
            "user_id": target.id,
            "impersonator": real.email,
        }
    )


PrincipalDep = Annotated[Principal, Depends(get_effective_principal)]


def require_admin(principal: PrincipalDep) -> Principal:
    """Reject a non-admin caller before an admin-only route body runs.

    A coarse gate over the already-resolved ``principal.is_admin``; the
    fine-grained role-change invariants live in ``UserService``. Applied
    per-route (not router-wide) so ``GET /users/me/metadata`` stays open to any
    authenticated caller.

    Raises:
        AdminRequiredError: the caller is authenticated but not an admin.
    """
    if not principal.is_admin:
        raise AdminRequiredError()
    return principal


def get_user_service(
    repository: Annotated[UserRepository, Depends(get_user_repository)],
    principal: PrincipalDep,
) -> UserService:
    """Provide the user service, scoped to the resolved principal."""
    return UserService(repository=repository, principal=principal)


UserServiceDep = Annotated[UserService, Depends(get_user_service)]


def get_job_repository(session: SessionDep) -> JobRepository:
    """Provide the job repository implementation."""
    return SqlAlchemyJobRepository(session)


def get_job_runner(settings: SettingsDep) -> JobRunner:
    """Provide the job runner selected by ``job_backend``.

    ``none`` (default) → ``NoOpJobRunner``: accepted jobs stay ``pending``, as
    before. ``local`` → ``LocalJobRunner`` running the ``autotune`` HPO in-process
    (no granite.build), driving the job to a terminal state itself. ``llmb`` →
    ``InProcessJobRunner`` submitting builds via the CLI (the local-bash spec when
    ``gb_environment="standalone"``, else custom_code). Each in-process runner opens
    its OWN session factory (the request session is gone by the time it runs),
    mirroring ``get_dataset_runner``. A queue-backed runner later replaces this one
    provider (task-queue open decision, unchanged).
    """
    if settings.job_backend == "none":
        return NoOpJobRunner()
    if settings.job_backend == "local":
        return LocalJobRunner(
            session_factory=get_session_factory(),
            trainer=AutotuneLocalTrainer(ray_address=settings.local_ray_address),
            output_root=settings.local_output_dir,
            dataset_root=settings.dataset_storage_dir,
            cancel_timeout=settings.local_cancel_timeout_seconds,
        )
    return InProcessJobRunner(
        session_factory=get_session_factory(),
        launcher=build_tuning_launcher(settings),
        canceller=build_build_canceller(settings),
    )


def get_job_service(
    repository: Annotated[JobRepository, Depends(get_job_repository)],
    configuration_repository: Annotated[
        ConfigurationRepository, Depends(get_configuration_repository)
    ],
    dataset_repository: Annotated[DatasetRepository, Depends(get_dataset_repository)],
    principal: PrincipalDep,
    runner: Annotated[JobRunner, Depends(get_job_runner)],
) -> JobService:
    """Provide the job service, scoped to the resolved principal.

    The principal arrives by constructor injection rather than as a router
    argument, so ``routers/jobs.py`` needs no signature change and its bodies
    stay one-liners.
    """
    return JobService(
        repository=repository,
        configuration_repository=configuration_repository,
        dataset_repository=dataset_repository,
        principal=principal,
        runner=runner,
    )


JobServiceDep = Annotated[JobService, Depends(get_job_service)]


def get_reward_executor(settings: SettingsDep) -> RewardExecutor:
    """Provide the hardened subprocess reward executor."""
    return SubprocessRewardExecutor(
        timeout_seconds=settings.reward_timeout_seconds,
        memory_bytes=settings.reward_memory_bytes,
    )


def get_reward_validation_service(
    executor: Annotated[RewardExecutor, Depends(get_reward_executor)],
) -> RewardValidationService:
    """Provide the reward-validation service."""
    return RewardValidationService(executor=executor)


RewardValidationServiceDep = Annotated[
    RewardValidationService, Depends(get_reward_validation_service)
]


def get_reward_tools_service(settings: SettingsDep, request: Request) -> RewardToolsService:
    """Provide the generate-test-solutions service (LLM built only when configured).

    Mirrors ``get_dataset_intelligence_service``'s LLM-optional shape: the
    client is built only when ``settings.llm_configured``, so the unconfigured
    path never touches ``app.state.http_client`` and the service itself raises
    the 503 (``LlmNotConfiguredError``) when its method is actually called.
    """
    llm: LlmClient | None = (
        build_llm_client(settings, get_http_client(request)) if settings.llm_configured else None
    )
    return RewardToolsService(llm=llm)


RewardToolsServiceDep = Annotated[RewardToolsService, Depends(get_reward_tools_service)]


def get_estimation_service(
    repository: Annotated[ConfigurationRepository, Depends(get_configuration_repository)],
    principal: PrincipalDep,
) -> EstimationService:
    """Provide the resource-estimation service."""
    return EstimationService(repository, principal)


EstimationServiceDep = Annotated[EstimationService, Depends(get_estimation_service)]


def get_asset_service(
    repository: Annotated[JobRepository, Depends(get_job_repository)],
    principal: PrincipalDep,
    settings: SettingsDep,
) -> AssetService:
    """Provide the result-report asset service."""
    return AssetService(
        job_repository=repository,
        principal=principal,
        settings=settings,
        filesystem=FilesystemArtifactLister(),
        huggingface=HuggingFaceArtifactLister(
            base_url=settings.hf_hub_base_url,
            token=os.environ.get(settings.hf_token_env),
        ),
    )


AssetServiceDep = Annotated[AssetService, Depends(get_asset_service)]


def get_reconcile_reader(
    request: Request,
    settings: SettingsDep,
    _admin: Annotated[Principal, Depends(require_admin)],
) -> BuildStatusReader:
    """Provide the gbserver build-status reader for on-demand reconcile.

    Depends on ``require_admin`` first, so a non-admin is refused with 403 before
    the availability check can reveal anything about the deployment. Reuses the
    reconcile loop's shared httpx client (opened in ``lifespan`` for the ``llmb``
    backend); raises :class:`BuildReconcileUnavailableError` (503) when this
    deployment has no reader.
    """
    client = getattr(request.app.state, "reconcile_http_client", None)
    if settings.job_backend != "llmb" or client is None:
        raise BuildReconcileUnavailableError()
    return build_status_reader_for(settings, client)


def get_on_demand_reconciler(
    repository: Annotated[JobRepository, Depends(get_job_repository)],
    reader: Annotated[BuildStatusReader, Depends(get_reconcile_reader)],
) -> OnDemandReconciler:
    """Provide the on-demand reconciler (admin-gated via the reader dependency)."""
    return OnDemandReconciler(repository=repository, reader=reader)


OnDemandReconcilerDep = Annotated[OnDemandReconciler, Depends(get_on_demand_reconciler)]


def get_gb_log_reader(settings: SettingsDep) -> GbLogReader:
    """Provide the gb log reader chosen by settings (gbcli reader / disabled)."""
    return build_gb_log_reader(settings)


def get_log_service(
    repository: Annotated[JobRepository, Depends(get_job_repository)],
    principal: PrincipalDep,
    gb_log_reader: Annotated[GbLogReader, Depends(get_gb_log_reader)],
) -> LogService:
    """Provide the log service, scoped to the resolved principal."""
    return LogService(repository=repository, principal=principal, gb_log_reader=gb_log_reader)


LogServiceDep = Annotated[LogService, Depends(get_log_service)]


def get_configuration_repository(session: SessionDep) -> ConfigurationRepository:
    """Provide the configuration repository implementation."""
    return SqlAlchemyConfigurationRepository(session)


def get_autotune_core() -> AutotuneCore:
    """Provide the autotune-core seam (reads the optional ``autotune`` package).

    Stateless and cheap to construct — the real import is deferred to the first
    method call and memoized there. Tests override this with a fake ``AutotuneCore``.
    """
    return AutotuneCoreAdapter()


AutotuneCoreDep = Annotated[AutotuneCore, Depends(get_autotune_core)]


def get_configuration_service(
    repository: Annotated[ConfigurationRepository, Depends(get_configuration_repository)],
    principal: PrincipalDep,
    autotune: AutotuneCoreDep,
) -> ConfigurationService:
    """Provide the configuration service, scoped to the resolved principal.

    The principal arrives by constructor injection, as in ``get_job_service``,
    so the router bodies stay one-liners. ``autotune`` backs only ``get_template``
    (the starter template comes from the autotune core); the CRUD path never
    touches it, so constructing the adapter is free until the template is asked for.
    """
    return ConfigurationService(repository=repository, principal=principal, autotune=autotune)


ConfigurationServiceDep = Annotated[ConfigurationService, Depends(get_configuration_service)]


def get_dataset_repository(session: SessionDep) -> DatasetRepository:
    """Provide the dataset repository implementation."""
    return SqlAlchemyDatasetRepository(session)


def get_storage_backend(settings: SettingsDep) -> StorageBackend:
    """Provide the storage backend chosen by settings (local/huggingface)."""
    return build_storage_backend(settings)


@lru_cache(maxsize=1)
def _shared_upload_runner() -> InProcessDatasetUploadRunner:
    """Build the one process-wide upload runner (its semaphore is shared).

    Reads settings and builds storage directly (not via request DI) because the
    runner outlives any request and its concurrency slot-set must be shared. A
    queue-backed runner would replace this. Endpoint tests override
    ``get_dataset_runner``, so this singleton is not exercised under test; if a
    future test does call it across event loops, clear the cache in a fixture.
    """
    settings = get_settings()
    return InProcessDatasetUploadRunner(
        session_factory=get_session_factory(),
        storage=build_storage_backend(settings),
        staging_dir=settings.dataset_staging_dir,
        max_concurrent=settings.dataset_upload_max_concurrent,
        processing_timeout_seconds=settings.dataset_processing_timeout_seconds,
    )


def get_dataset_runner() -> DatasetUploadRunner:
    """Provide the process-wide in-process upload runner (see ``_shared_upload_runner``)."""
    return _shared_upload_runner()


def get_dataset_service(
    repository: Annotated[DatasetRepository, Depends(get_dataset_repository)],
    principal: PrincipalDep,
    storage: Annotated[StorageBackend, Depends(get_storage_backend)],
    runner: Annotated[DatasetUploadRunner, Depends(get_dataset_runner)],
    settings: SettingsDep,
) -> DatasetService:
    """Provide the dataset service, scoped to the resolved principal."""
    return DatasetService(
        repository=repository,
        principal=principal,
        storage=storage,
        runner=runner,
        settings=settings,
    )


DatasetServiceDep = Annotated[DatasetService, Depends(get_dataset_service)]


def get_llm_client(
    settings: SettingsDep,
    http_client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
) -> LlmClient:
    """Provide the configured LLM client, or raise 503 if unconfigured."""
    return build_llm_client(settings, http_client)


def get_dataset_intelligence_service(
    settings: SettingsDep,
    request: Request,
    autotune: AutotuneCoreDep,
) -> DatasetIntelligenceService:
    """Provide the dataset-intelligence service.

    The LLM client is built only when configured (``settings.llm_configured``);
    otherwise the service holds ``None`` and its LLM-using methods raise a 503.
    The dataset-type catalog now comes from the ``autotune`` core, so
    ``GET …/formats`` and ``suggest-mapping`` additionally require ``autotune``
    (503 when it is not installed); ``validate-strategy`` and ``parse-strategy``
    need neither the LLM nor autotune. The shared ``app.state.http_client`` is
    read lazily *inside* the configured branch rather than injected as a
    dependency, so the no-LLM endpoints never require it (and the unconfigured
    path never touches it — the lifespan hook that opens it does not run under
    the test transport).
    """
    llm: LlmClient | None = (
        build_llm_client(settings, get_http_client(request)) if settings.llm_configured else None
    )
    return DatasetIntelligenceService(llm=llm, settings=settings, autotune=autotune)


DatasetIntelligenceServiceDep = Annotated[
    DatasetIntelligenceService, Depends(get_dataset_intelligence_service)
]


_chat_memory: ConversationMemory | None = None
"""Process-wide singleton, lazily constructed on first use.

Mirrors the old process-local langgraph checkpointer it replaces: one bounded
store per worker process, not per request, so conversation state on a
``thread_id`` survives across turns. Built lazily (rather than at import time
or in ``lifespan``) so its size/TTL always come from whatever ``Settings`` a
given call is running under — tests build several ``Settings`` instances in
one process without a running lifespan.
"""


def _get_chat_memory(settings: Settings) -> ConversationMemory:
    """Return the process-wide conversation memory, constructing it once."""
    global _chat_memory
    if _chat_memory is None:
        _chat_memory = ConversationMemory(
            max_threads=settings.chat_max_threads, ttl_seconds=settings.chat_thread_ttl_seconds
        )
    return _chat_memory


def get_chat_service(settings: SettingsDep, request: Request) -> ChatService:
    """Provide the chat service, or raise a 503 if no LLM provider is configured.

    Checks ``settings.llm_configured`` before touching ``get_http_client`` —
    the same lazy shape as ``get_dataset_intelligence_service`` — so the
    unconfigured path never needs ``app.state.http_client`` at all. That
    matters for tests: ``ASGITransport`` never runs ``lifespan``, so that
    attribute does not exist unless a test's own fixture sets it, and the 503
    case must not depend on it existing.

    Raises:
        LlmNotConfiguredError: the ``llm_*`` settings are unset.
    """
    if not settings.llm_configured:
        raise LlmNotConfiguredError()
    llm = build_llm_client(settings, get_http_client(request))
    return ChatService(llm=llm, memory=_get_chat_memory(settings), settings=settings)


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
