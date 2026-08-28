# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Configuration business logic.

This layer owns every domain rule about configurations: that ``config_data`` is
a valid search space, that a caller may only see and write its own rows by
default (an admin may opt into every row with ``scope=DataScope.ALL``), and
that a caller with no resolvable identity may not create.
It knows nothing about HTTP — it raises the exceptions in
:mod:`autotunex.core.exceptions` and lets the API layer translate them.

The ownership scoping (:func:`~autotunex.services.scoping.resolve_owner_filter` /
:func:`~autotunex.services.scoping.sees_nothing`) is the same shared seam
:class:`autotunex.services.jobs.JobService` uses, deliberately: one rule, one
place it is written, so the read and write paths cannot drift apart.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from autotunex.core.exceptions import (
    CallerNotProvisionedError,
    ConfigurationNotFoundError,
    InvalidConfigDataError,
)
from autotunex.db.repositories.protocols import ConfigurationRepository
from autotunex.db.tables import JobTable
from autotunex.models.auth import Principal
from autotunex.models.common import DataScope, Page
from autotunex.models.configuration import (
    ConfigurationCreate,
    ConfigurationJobRef,
    ConfigurationRead,
)
from autotunex.services.autotune import AutotuneCore
from autotunex.services.mappers import configuration_to_read
from autotunex.services.scoping import resolve_owner_filter, sees_nothing


class ConfigurationService:
    """Full CRUD over configurations, scoped to the calling principal."""

    def __init__(
        self, repository: ConfigurationRepository, principal: Principal, autotune: AutotuneCore
    ) -> None:
        self._repository = repository
        self._principal = principal
        self._autotune = autotune

    @staticmethod
    def _validate_config_data(config_data: dict[str, Any]) -> None:
        """Reject an empty ``config_data``.

        ``configurations.config_data`` is a schema-less ``JSON`` column, and the
        tuning pipeline writes a far richer, evolving structure than any model
        here owns (``tune_config``, ``tuners_config``, ``training_config`` and
        friends). So the API does not impose an internal shape — validating
        against :mod:`autotunex.models.search_space` would reject every real
        configuration. The one rule kept is that a configuration must carry some
        settings: an empty object is a domain 422, not a stored no-op. FastAPI's
        request validation already rejects a non-object body upstream.

        Raises:
            InvalidConfigDataError: ``config_data`` is empty.
        """
        if not config_data:
            raise InvalidConfigDataError("config_data must be a non-empty JSON object.")

    @staticmethod
    def _job_refs(jobs: list[JobTable]) -> list[ConfigurationJobRef]:
        """Convert referencing jobs to compact refs (already caller-scoped)."""
        return [
            ConfigurationJobRef(id=job.id, experiment_name=job.experiment_name, status=job.status)
            for job in jobs
        ]

    async def _associated(
        self, configuration_id: UUID, *, owner_id: UUID | None
    ) -> list[ConfigurationJobRef]:
        """Return caller-scoped tuning refs for one configuration, using a resolved filter."""
        grouped = await self._repository.jobs_for_config([configuration_id], owner_id=owner_id)
        return self._job_refs(grouped.get(configuration_id, []))

    async def get_template(self) -> dict[str, Any]:
        """Return the starter configuration template from the autotune core.

        The template (autotune's ``get_autotune_config``) is server-global, so —
        unlike every other method here — it is *not* scoped to the principal:
        there is nothing owned to hide. It is the schema-less ``config_data``
        shape a new configuration is built from, returned verbatim.

        Raises:
            AutotuneCoreUnavailableError: the ``autotune`` package is not installed.
        """
        return await self._autotune.get_config_template()

    async def get(
        self, configuration_id: UUID, *, scope: DataScope = DataScope.OWN
    ) -> ConfigurationRead:
        """Return the configuration with ``configuration_id``, scoped to the caller.

        Raises:
            ScopeNotPermittedError: a non-admin requested ``scope=all``.
            ConfigurationNotFoundError: no such configuration, or another owner's.
        """
        owner_id = resolve_owner_filter(self._principal, scope)
        if sees_nothing(self._principal, scope):
            raise ConfigurationNotFoundError(configuration_id)
        configuration = await self._repository.get(configuration_id, owner_id=owner_id)
        if configuration is None:
            raise ConfigurationNotFoundError(configuration_id)
        return configuration_to_read(
            configuration, await self._associated(configuration_id, owner_id=owner_id)
        )

    async def list(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        scope: DataScope = DataScope.OWN,
        q: str | None = None,
    ) -> Page[ConfigurationRead]:
        """Return one page of the caller's configurations, newest first.

        ``q`` is an optional case-insensitive substring filter on ``name``.

        Raises:
            ScopeNotPermittedError: a non-admin requested ``scope=all``.
        """
        owner_id = resolve_owner_filter(self._principal, scope)
        if sees_nothing(self._principal, scope):
            return Page[ConfigurationRead](items=[], total=0, limit=limit, offset=offset)
        configurations, total = await self._repository.list(
            limit=limit, offset=offset, owner_id=owner_id, q=q
        )
        grouped = await self._repository.jobs_for_config(
            [configuration.id for configuration in configurations], owner_id=owner_id
        )
        return Page[ConfigurationRead](
            items=[
                configuration_to_read(
                    configuration, self._job_refs(grouped.get(configuration.id, []))
                )
                for configuration in configurations
            ],
            total=total,
            limit=limit,
            offset=offset,
        )

    async def create(self, data: ConfigurationCreate) -> ConfigurationRead:
        """Create a configuration owned by the calling principal.

        Ownership comes from ``principal.user_id``, never from the request. A
        caller with no resolvable ``user_id`` — an unprovisioned user, or an
        unrestricted standalone admin — has no identity to attach and is
        refused, since ``configurations.user_id`` is ``NOT NULL``.

        Raises:
            CallerNotProvisionedError: the caller has no ``user_id`` to own the row.
            InvalidConfigDataError: ``config_data`` is empty.
            ConfigurationNameConflictError: the caller already owns a
                configuration with this name.
        """
        owner_id = self._principal.user_id
        if owner_id is None:
            raise CallerNotProvisionedError()
        self._validate_config_data(data.config_data)
        configuration = await self._repository.create(
            user_id=str(owner_id),
            name=data.name,
            tuner_type=data.tuner_type,
            rl_tuner_type=data.rl_tuner_type,
            config_data=data.config_data,
        )
        return configuration_to_read(configuration, [])

    async def update(
        self, configuration_id: UUID, data: ConfigurationCreate, *, scope: DataScope = DataScope.OWN
    ) -> ConfigurationRead:
        """Fully replace a configuration's mutable fields, scoped to the caller.

        A ``PUT``: the caller resends the whole representation, and ownership is
        left untouched. A missing configuration and one owned by someone else are
        indistinguishable, as on the read path.

        Raises:
            ScopeNotPermittedError: a non-admin requested ``scope=all``.
            InvalidConfigDataError: ``config_data`` is empty.
            ConfigurationNotFoundError: no such configuration, or not the caller's.
            ConfigurationNameConflictError: the new name collides with another of
                the caller's configurations.
        """
        owner_id = resolve_owner_filter(self._principal, scope)
        if sees_nothing(self._principal, scope):
            raise ConfigurationNotFoundError(configuration_id)
        self._validate_config_data(data.config_data)
        configuration = await self._repository.update(
            configuration_id,
            owner_id=owner_id,
            name=data.name,
            tuner_type=data.tuner_type,
            rl_tuner_type=data.rl_tuner_type,
            config_data=data.config_data,
        )
        if configuration is None:
            raise ConfigurationNotFoundError(configuration_id)
        return configuration_to_read(
            configuration, await self._associated(configuration_id, owner_id=owner_id)
        )

    async def delete(self, configuration_id: UUID, *, scope: DataScope = DataScope.OWN) -> None:
        """Delete a configuration, scoped to the caller.

        Raises:
            ScopeNotPermittedError: a non-admin requested ``scope=all``.
            ConfigurationNotFoundError: no such configuration, or not the caller's.
            ConfigurationInUseError: a job still references the configuration.
        """
        owner_id = resolve_owner_filter(self._principal, scope)
        if sees_nothing(self._principal, scope):
            raise ConfigurationNotFoundError(configuration_id)
        deleted = await self._repository.delete(configuration_id, owner_id=owner_id)
        if not deleted:
            raise ConfigurationNotFoundError(configuration_id)
