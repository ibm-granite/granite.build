# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Builds Principal-scoped services for a single tool call, outside FastAPI.

Mirrors ``api/deps.py`` construction so a chat tool or an MCP tool sees exactly
what the REST API would: same repositories, same services, same runner
selection, same ownership scoping. One database session is opened per tool call.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from autotunex.core.config import Settings
from autotunex.db.repositories.sqlalchemy import (
    SqlAlchemyConfigurationRepository,
    SqlAlchemyDatasetRepository,
    SqlAlchemyJobRepository,
    SqlAlchemyUserRepository,
)
from autotunex.db.session import get_session_factory
from autotunex.models.auth import Principal
from autotunex.services.autotune import AutotuneCoreAdapter
from autotunex.services.configurations import ConfigurationService
from autotunex.services.dataset_runner import InProcessDatasetUploadRunner
from autotunex.services.datasets import DatasetService
from autotunex.services.gb_logs.registry import get_gb_log_reader
from autotunex.services.jobs import JobService
from autotunex.services.launch.registry import get_build_canceller, get_tuning_launcher
from autotunex.services.local.runner import LocalJobRunner
from autotunex.services.local.trainer import AutotuneLocalTrainer
from autotunex.services.logs import LogService
from autotunex.services.protocols import JobRunner
from autotunex.services.runner import InProcessJobRunner, NoOpJobRunner
from autotunex.services.storage.registry import get_storage_backend
from autotunex.services.users import UserService


def _build_job_runner(
    settings: Settings, session_factory: async_sessionmaker[AsyncSession]
) -> JobRunner:
    """Mirror ``api/deps.get_job_runner`` so tool submits match ``POST /jobs``.

    Each in-process runner opens its OWN session factory, exactly as
    ``get_job_runner`` does — the tool-call session closes when ``services()``
    exits, long before a runner's background ``process`` task would run.
    """
    if settings.job_backend == "none":
        return NoOpJobRunner()
    if settings.job_backend == "local":
        return LocalJobRunner(
            session_factory=session_factory,
            trainer=AutotuneLocalTrainer(ray_address=settings.local_ray_address),
            output_root=settings.local_output_dir,
            dataset_root=settings.dataset_storage_dir,
            cancel_timeout=settings.local_cancel_timeout_seconds,
        )
    return InProcessJobRunner(
        session_factory=session_factory,
        launcher=get_tuning_launcher(settings),
        canceller=get_build_canceller(settings),
    )


@dataclass(slots=True)
class ScopedServices:
    """The Principal-scoped services a tool handler calls.

    One instance per tool call, backed by the single database session opened
    by :meth:`ToolContext.services`.
    """

    principal: Principal
    job: JobService
    config: ConfigurationService
    dataset: DatasetService
    user: UserService
    logs: LogService


@dataclass(slots=True)
class ToolContext:
    """Everything a tool needs to build its scoped services for one call.

    Constructed once per tool invocation (outside FastAPI's dependency
    injection — there is no request here), then :meth:`services` opens exactly
    one database session and builds the same repositories/services
    ``api/deps.py`` would for an equivalent HTTP request.
    """

    principal: Principal
    settings: Settings
    session_factory: async_sessionmaker[AsyncSession]

    @classmethod
    def for_principal(cls, principal: Principal, settings: Settings) -> ToolContext:
        """Build a context using the process-wide session factory."""
        return cls(principal=principal, settings=settings, session_factory=get_session_factory())

    @asynccontextmanager
    async def services(self) -> AsyncIterator[ScopedServices]:
        """Open one database session and yield the principal-scoped services.

        The session (and everything built from it) is only valid for the
        duration of this ``async with`` block, matching a single FastAPI
        request's lifetime.
        """
        async with self.session_factory() as session:
            job_repo = SqlAlchemyJobRepository(session)
            config_repo = SqlAlchemyConfigurationRepository(session)
            dataset_repo = SqlAlchemyDatasetRepository(session)
            user_repo = SqlAlchemyUserRepository(session)
            autotune = AutotuneCoreAdapter()
            storage = get_storage_backend(self.settings)
            runner = _build_job_runner(self.settings, self.session_factory)
            dataset_runner = InProcessDatasetUploadRunner(
                session_factory=self.session_factory,
                storage=storage,
                staging_dir=self.settings.dataset_staging_dir,
            )
            yield ScopedServices(
                principal=self.principal,
                job=JobService(
                    repository=job_repo,
                    configuration_repository=config_repo,
                    dataset_repository=dataset_repo,
                    principal=self.principal,
                    runner=runner,
                ),
                config=ConfigurationService(
                    repository=config_repo, principal=self.principal, autotune=autotune
                ),
                dataset=DatasetService(
                    repository=dataset_repo,
                    principal=self.principal,
                    storage=storage,
                    runner=dataset_runner,
                    settings=self.settings,
                ),
                user=UserService(repository=user_repo, principal=self.principal),
                logs=LogService(
                    repository=job_repo,
                    principal=self.principal,
                    gb_log_reader=get_gb_log_reader(self.settings),
                ),
            )
