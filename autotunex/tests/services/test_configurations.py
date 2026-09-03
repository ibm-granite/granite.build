"""Unit tests for ConfigurationService, isolated from the database.

The fake repository below is a plain class, not a mock: because the seam is a
Protocol, structural typing is enough, and mypy verifies conformance via the
annotated assignment in ``test_doubles_satisfy_their_protocols``. It enforces the
same two constraints the real SQLAlchemy repository does — the ``UNIQUE
(user_id, name)`` collision and the "still referenced by a job" block — so the
service's error-mapping is exercised without a database.
"""

from __future__ import annotations

import builtins
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

from autotunex.core.exceptions import (
    AutotuneCoreUnavailableError,
    CallerNotProvisionedError,
    ConfigurationInUseError,
    ConfigurationNameConflictError,
    ConfigurationNotFoundError,
    InvalidConfigDataError,
    ScopeNotPermittedError,
)
from autotunex.db.repositories.protocols import ConfigurationRepository
from autotunex.db.tables import ConfigurationTable, JobTable
from autotunex.models.auth import Principal
from autotunex.models.common import DataScope
from autotunex.models.configuration import ConfigurationCreate
from autotunex.models.status import RunStatus
from autotunex.services.autotune import AutotuneCore
from autotunex.services.configurations import ConfigurationService

SAMPLE_CONFIG: dict[str, Any] = {
    "tune_config": {"num_samples": {"type": "int", "default": 16, "min_val": 1, "max_val": 10000}}
}
"""A non-empty ``config_data`` blob in the real (schema-less) shape.

Mirrors what the tuning pipeline actually writes: the API accepts any non-empty
JSON object, so the sample must exercise the real shape rather than a tidied-up
one the service never sees.
"""

ADMIN_ID = uuid4()
ADMIN = Principal(email="admin@example.com", provider="session", user_id=ADMIN_ID, is_admin=True)
"""An admin who is also a provisioned user, so it can create as well as read."""


class FakeConfigurationRepository:
    """In-memory configuration store enforcing the two DB constraints."""

    def __init__(self) -> None:
        self.configs: dict[UUID, ConfigurationTable] = {}
        self.referenced_by_a_job: set[UUID] = set()
        self.jobs: dict[UUID, list[JobTable]] = {}

    def seed(self, *, owner_id: str, name: str = "cfg") -> ConfigurationTable:
        """Add a configuration straight into the store."""
        now = datetime.now(UTC)
        config = ConfigurationTable(
            id=uuid4(),
            user_id=owner_id,
            name=name,
            tuner_type=None,
            rl_tuner_type=None,
            config_data=dict(SAMPLE_CONFIG),
            created_at=now,
            updated_at=now,
        )
        self.configs[config.id] = config
        return config

    def _name_taken(self, *, user_id: str, name: str, excluding: UUID | None) -> bool:
        return any(
            config.id != excluding and config.user_id == user_id and config.name == name
            for config in self.configs.values()
        )

    async def get(
        self,
        configuration_id: UUID,
        *,
        owner_id: UUID | None = None,
        include_shared: bool = False,
    ) -> ConfigurationTable | None:
        config = self.configs.get(configuration_id)
        if config is None:
            return None
        if owner_id is not None and config.user_id != str(owner_id):
            return None
        return config

    async def list(
        self,
        *,
        limit: int,
        offset: int,
        owner_id: UUID | None = None,
        q: str | None = None,
        include_shared: bool = False,
    ) -> tuple[Sequence[ConfigurationTable], int]:
        matching = [
            config
            for config in self.configs.values()
            if owner_id is None or config.user_id == str(owner_id)
        ]
        if q:
            needle = q.lower()
            matching = [config for config in matching if needle in config.name.lower()]
        ordered = sorted(matching, key=lambda config: config.created_at, reverse=True)
        return ordered[offset : offset + limit], len(ordered)

    async def create(
        self,
        *,
        user_id: str,
        name: str,
        tuner_type: str | None,
        rl_tuner_type: str | None,
        config_data: dict[str, Any],
    ) -> ConfigurationTable:
        if self._name_taken(user_id=user_id, name=name, excluding=None):
            raise ConfigurationNameConflictError(name)
        now = datetime.now(UTC)
        config = ConfigurationTable(
            id=uuid4(),
            user_id=user_id,
            name=name,
            tuner_type=tuner_type,
            rl_tuner_type=rl_tuner_type,
            config_data=config_data,
            created_at=now,
            updated_at=now,
        )
        self.configs[config.id] = config
        return config

    async def update(
        self,
        configuration_id: UUID,
        *,
        owner_id: UUID | None = None,
        name: str,
        tuner_type: str | None,
        rl_tuner_type: str | None,
        config_data: dict[str, Any],
    ) -> ConfigurationTable | None:
        config = await self.get(configuration_id, owner_id=owner_id)
        if config is None:
            return None
        if self._name_taken(user_id=config.user_id, name=name, excluding=configuration_id):
            raise ConfigurationNameConflictError(name)
        config.name = name
        config.tuner_type = tuner_type
        config.rl_tuner_type = rl_tuner_type
        config.config_data = config_data
        return config

    async def delete(self, configuration_id: UUID, *, owner_id: UUID | None = None) -> bool:
        config = await self.get(configuration_id, owner_id=owner_id)
        if config is None:
            return False
        if configuration_id in self.referenced_by_a_job:
            raise ConfigurationInUseError(configuration_id)
        del self.configs[configuration_id]
        return True

    async def jobs_for_config(
        self, config_ids: Sequence[UUID], *, owner_id: UUID | None = None
    ) -> dict[UUID, builtins.list[JobTable]]:
        result: dict[UUID, builtins.list[JobTable]] = {}
        for config_id in config_ids:
            jobs = self.jobs.get(config_id, [])
            if owner_id is not None:
                jobs = [j for j in jobs if j.user_id == str(owner_id)]
            if jobs:
                result[config_id] = jobs
        return result


class FakeAutotuneCore:
    """In-memory ``AutotuneCore``: returns a canned template, or raises."""

    def __init__(
        self, *, config: dict[str, Any] | None = None, raises: Exception | None = None
    ) -> None:
        self._config = config if config is not None else {}
        self._raises = raises

    async def get_config_template(self) -> dict[str, Any]:
        if self._raises is not None:
            raise self._raises
        return self._config

    async def get_dataset_types(self) -> dict[str, Any]:
        if self._raises is not None:
            raise self._raises
        return {}


@pytest.fixture
def repository() -> FakeConfigurationRepository:
    return FakeConfigurationRepository()


@pytest.fixture
def service(repository: FakeConfigurationRepository) -> ConfigurationService:
    return ConfigurationService(repository=repository, principal=ADMIN, autotune=FakeAutotuneCore())


def _body(
    *, name: str = "lora-sweep", config_data: dict[str, Any] | None = None
) -> ConfigurationCreate:
    return ConfigurationCreate(
        name=name, config_data=SAMPLE_CONFIG if config_data is None else config_data
    )


def test_doubles_satisfy_their_protocols() -> None:
    """A type-level assertion: mypy fails here if the Protocol drifts."""
    repository: ConfigurationRepository = FakeConfigurationRepository()
    autotune: AutotuneCore = FakeAutotuneCore()

    assert repository is not None and autotune is not None


# Template.


async def test_get_template_returns_the_autotune_template(
    repository: FakeConfigurationRepository,
) -> None:
    template = {"tune_config": {"num_samples": {"default": 16, "min_val": 1, "max_val": 10000}}}
    service = ConfigurationService(
        repository=repository, principal=ADMIN, autotune=FakeAutotuneCore(config=template)
    )

    result = await service.get_template()

    assert result == template


async def test_get_template_propagates_autotune_unavailable(
    repository: FakeConfigurationRepository,
) -> None:
    service = ConfigurationService(
        repository=repository,
        principal=ADMIN,
        autotune=FakeAutotuneCore(raises=AutotuneCoreUnavailableError()),
    )

    with pytest.raises(AutotuneCoreUnavailableError):
        await service.get_template()


# Create.


async def test_create_stores_a_configuration_owned_by_the_caller(
    service: ConfigurationService,
) -> None:
    created = await service.create(_body(name="lora-sweep"))

    assert created.name == "lora-sweep"
    assert created.user_id == str(ADMIN_ID)


async def test_create_rejects_an_empty_config_data(service: ConfigurationService) -> None:
    with pytest.raises(InvalidConfigDataError):
        await service.create(_body(config_data={}))


async def test_create_accepts_an_arbitrary_config_data_object(
    service: ConfigurationService,
) -> None:
    """config_data is a schema-less blob: any non-empty object is accepted."""
    blob = {"tuners_config": {"lora": {"hyperparams": {"r": {"type": "int", "default": 32}}}}}

    created = await service.create(_body(config_data=blob))

    assert created.config_data == blob


async def test_create_is_refused_for_an_unprovisioned_caller(
    repository: FakeConfigurationRepository,
) -> None:
    principal = Principal(
        email="ghost@example.com", provider="session", user_id=None, is_admin=False
    )
    service = ConfigurationService(
        repository=repository, principal=principal, autotune=FakeAutotuneCore()
    )

    with pytest.raises(CallerNotProvisionedError):
        await service.create(_body())


async def test_create_is_refused_for_an_admin_with_no_user_id(
    repository: FakeConfigurationRepository,
) -> None:
    """Unrestricted standalone: is_admin=True but no users row to own the config."""
    principal = Principal(email=None, provider="disabled", user_id=None, is_admin=True)
    service = ConfigurationService(
        repository=repository, principal=principal, autotune=FakeAutotuneCore()
    )

    with pytest.raises(CallerNotProvisionedError):
        await service.create(_body())


async def test_create_surfaces_a_name_collision_as_a_conflict(
    service: ConfigurationService, repository: FakeConfigurationRepository
) -> None:
    repository.seed(owner_id=str(ADMIN_ID), name="taken")

    with pytest.raises(ConfigurationNameConflictError):
        await service.create(_body(name="taken"))


# Read.


async def test_get_returns_the_seeded_configuration(
    service: ConfigurationService, repository: FakeConfigurationRepository
) -> None:
    seeded = repository.seed(owner_id=str(ADMIN_ID))

    config = await service.get(seeded.id)

    assert config.id == seeded.id


async def test_get_raises_for_an_unknown_configuration(service: ConfigurationService) -> None:
    with pytest.raises(ConfigurationNotFoundError):
        await service.get(uuid4())


async def test_a_provisioned_user_sees_only_their_own(
    repository: FakeConfigurationRepository,
) -> None:
    owner_id = uuid4()
    mine = repository.seed(owner_id=str(owner_id), name="mine")
    repository.seed(owner_id=str(uuid4()), name="theirs")
    principal = Principal(
        email="u@example.com", provider="session", user_id=owner_id, is_admin=False
    )
    service = ConfigurationService(
        repository=repository, principal=principal, autotune=FakeAutotuneCore()
    )

    page = await service.list(limit=20, offset=0)

    assert page.total == 1
    assert page.items[0].id == mine.id


async def test_a_provisioned_user_cannot_get_anothers(
    repository: FakeConfigurationRepository,
) -> None:
    other = repository.seed(owner_id=str(uuid4()))
    principal = Principal(
        email="u@example.com", provider="session", user_id=uuid4(), is_admin=False
    )
    service = ConfigurationService(
        repository=repository, principal=principal, autotune=FakeAutotuneCore()
    )

    with pytest.raises(ConfigurationNotFoundError):
        await service.get(other.id)


async def test_an_admin_sees_only_their_own_configurations_by_default(
    service: ConfigurationService, repository: FakeConfigurationRepository
) -> None:
    repository.seed(owner_id=str(uuid4()))
    repository.seed(owner_id=str(uuid4()))

    page = await service.list(limit=20, offset=0)

    assert page.total == 0


async def test_an_admin_lists_every_configuration_with_scope_all(
    service: ConfigurationService, repository: FakeConfigurationRepository
) -> None:
    repository.seed(owner_id=str(uuid4()))
    repository.seed(owner_id=str(uuid4()))

    page = await service.list(limit=20, offset=0, scope=DataScope.ALL)

    assert page.total == 2


async def test_a_non_admin_requesting_scope_all_on_list_is_forbidden(
    repository: FakeConfigurationRepository,
) -> None:
    principal = Principal(
        email="u@example.com", provider="session", user_id=uuid4(), is_admin=False
    )
    service = ConfigurationService(
        repository=repository, principal=principal, autotune=FakeAutotuneCore()
    )

    with pytest.raises(ScopeNotPermittedError):
        await service.list(limit=20, offset=0, scope=DataScope.ALL)


async def test_an_admin_cannot_get_a_foreign_configuration_by_default(
    service: ConfigurationService, repository: FakeConfigurationRepository
) -> None:
    other = repository.seed(owner_id=str(uuid4()))

    with pytest.raises(ConfigurationNotFoundError):
        await service.get(other.id)


async def test_an_admin_gets_a_foreign_configuration_with_scope_all(
    service: ConfigurationService, repository: FakeConfigurationRepository
) -> None:
    other = repository.seed(owner_id=str(uuid4()))

    config = await service.get(other.id, scope=DataScope.ALL)

    assert config.id == other.id


async def test_a_non_admin_requesting_scope_all_on_get_is_forbidden(
    repository: FakeConfigurationRepository,
) -> None:
    existing = repository.seed(owner_id=str(uuid4()))
    principal = Principal(
        email="u@example.com", provider="session", user_id=uuid4(), is_admin=False
    )
    service = ConfigurationService(
        repository=repository, principal=principal, autotune=FakeAutotuneCore()
    )

    with pytest.raises(ScopeNotPermittedError):
        await service.get(existing.id, scope=DataScope.ALL)


async def test_an_unprovisioned_caller_sees_an_empty_page(
    repository: FakeConfigurationRepository,
) -> None:
    repository.seed(owner_id=str(uuid4()))
    principal = Principal(
        email="ghost@example.com", provider="session", user_id=None, is_admin=False
    )
    service = ConfigurationService(
        repository=repository, principal=principal, autotune=FakeAutotuneCore()
    )

    page = await service.list(limit=20, offset=0)

    assert page.total == 0
    assert page.items == []


async def test_an_unprovisioned_caller_gets_not_found_on_get(
    repository: FakeConfigurationRepository,
) -> None:
    existing = repository.seed(owner_id=str(uuid4()))
    principal = Principal(
        email="ghost@example.com", provider="session", user_id=None, is_admin=False
    )
    service = ConfigurationService(
        repository=repository, principal=principal, autotune=FakeAutotuneCore()
    )

    with pytest.raises(ConfigurationNotFoundError):
        await service.get(existing.id)


# Update.


async def test_update_replaces_every_field(
    service: ConfigurationService, repository: FakeConfigurationRepository
) -> None:
    seeded = repository.seed(owner_id=str(ADMIN_ID), name="before")
    new_space = {"lora_rank": {"kind": "int", "low": 4, "high": 64, "step": 4}}

    updated = await service.update(
        seeded.id, ConfigurationCreate(name="after", tuner_type="optuna", config_data=new_space)
    )

    assert updated.name == "after"
    assert updated.tuner_type == "optuna"
    assert updated.config_data == new_space


async def test_update_rejects_an_empty_config_data(
    service: ConfigurationService, repository: FakeConfigurationRepository
) -> None:
    seeded = repository.seed(owner_id=str(ADMIN_ID))

    with pytest.raises(InvalidConfigDataError):
        await service.update(seeded.id, _body(config_data={}))


async def test_update_of_an_unknown_configuration_raises_not_found(
    service: ConfigurationService,
) -> None:
    with pytest.raises(ConfigurationNotFoundError):
        await service.update(uuid4(), _body())


async def test_update_is_scoped_to_the_owner(
    repository: FakeConfigurationRepository,
) -> None:
    other = repository.seed(owner_id=str(uuid4()))
    principal = Principal(
        email="u@example.com", provider="session", user_id=uuid4(), is_admin=False
    )
    service = ConfigurationService(
        repository=repository, principal=principal, autotune=FakeAutotuneCore()
    )

    with pytest.raises(ConfigurationNotFoundError):
        await service.update(other.id, _body())


async def test_an_admin_cannot_update_a_foreign_configuration_by_default(
    service: ConfigurationService, repository: FakeConfigurationRepository
) -> None:
    other = repository.seed(owner_id=str(uuid4()))

    with pytest.raises(ConfigurationNotFoundError):
        await service.update(other.id, _body())


async def test_an_admin_updates_a_foreign_configuration_with_scope_all(
    service: ConfigurationService, repository: FakeConfigurationRepository
) -> None:
    other = repository.seed(owner_id=str(uuid4()), name="before")

    updated = await service.update(other.id, _body(name="after"), scope=DataScope.ALL)

    assert updated.name == "after"


async def test_a_non_admin_requesting_scope_all_on_update_is_forbidden(
    repository: FakeConfigurationRepository,
) -> None:
    existing = repository.seed(owner_id=str(uuid4()))
    principal = Principal(
        email="u@example.com", provider="session", user_id=uuid4(), is_admin=False
    )
    service = ConfigurationService(
        repository=repository, principal=principal, autotune=FakeAutotuneCore()
    )

    with pytest.raises(ScopeNotPermittedError):
        await service.update(existing.id, _body(), scope=DataScope.ALL)


async def test_update_into_a_name_collision_is_a_conflict(
    service: ConfigurationService, repository: FakeConfigurationRepository
) -> None:
    repository.seed(owner_id=str(ADMIN_ID), name="taken")
    mine = repository.seed(owner_id=str(ADMIN_ID), name="mine")

    with pytest.raises(ConfigurationNameConflictError):
        await service.update(mine.id, _body(name="taken"))


# Delete.


async def test_delete_removes_the_configuration(
    service: ConfigurationService, repository: FakeConfigurationRepository
) -> None:
    seeded = repository.seed(owner_id=str(ADMIN_ID))

    await service.delete(seeded.id)

    assert seeded.id not in repository.configs


async def test_delete_of_an_unknown_configuration_raises_not_found(
    service: ConfigurationService,
) -> None:
    with pytest.raises(ConfigurationNotFoundError):
        await service.delete(uuid4())


async def test_delete_is_scoped_to_the_owner(repository: FakeConfigurationRepository) -> None:
    other = repository.seed(owner_id=str(uuid4()))
    principal = Principal(
        email="u@example.com", provider="session", user_id=uuid4(), is_admin=False
    )
    service = ConfigurationService(
        repository=repository, principal=principal, autotune=FakeAutotuneCore()
    )

    with pytest.raises(ConfigurationNotFoundError):
        await service.delete(other.id)


async def test_an_admin_cannot_delete_a_foreign_configuration_by_default(
    service: ConfigurationService, repository: FakeConfigurationRepository
) -> None:
    other = repository.seed(owner_id=str(uuid4()))

    with pytest.raises(ConfigurationNotFoundError):
        await service.delete(other.id)


async def test_an_admin_deletes_a_foreign_configuration_with_scope_all(
    service: ConfigurationService, repository: FakeConfigurationRepository
) -> None:
    other = repository.seed(owner_id=str(uuid4()))

    await service.delete(other.id, scope=DataScope.ALL)

    assert other.id not in repository.configs


async def test_a_non_admin_requesting_scope_all_on_delete_is_forbidden(
    repository: FakeConfigurationRepository,
) -> None:
    existing = repository.seed(owner_id=str(uuid4()))
    principal = Principal(
        email="u@example.com", provider="session", user_id=uuid4(), is_admin=False
    )
    service = ConfigurationService(
        repository=repository, principal=principal, autotune=FakeAutotuneCore()
    )

    with pytest.raises(ScopeNotPermittedError):
        await service.delete(existing.id, scope=DataScope.ALL)


async def test_delete_of_a_referenced_configuration_is_a_conflict(
    service: ConfigurationService, repository: FakeConfigurationRepository
) -> None:
    seeded = repository.seed(owner_id=str(ADMIN_ID))
    repository.referenced_by_a_job.add(seeded.id)

    with pytest.raises(ConfigurationInUseError):
        await service.delete(seeded.id)


# Associated jobs ("tunings").


async def test_get_includes_caller_scoped_associated_jobs(
    service: ConfigurationService, repository: FakeConfigurationRepository
) -> None:
    seeded = repository.seed(owner_id=str(ADMIN_ID))
    repository.jobs[seeded.id] = [
        JobTable(
            id=uuid4(),
            user_id=str(ADMIN_ID),
            status=RunStatus.RUNNING,
            config_id=seeded.id,
            dataset_id=uuid4(),
            model="m",
            model_source="huggingface",
            experiment_name="exp",
        ),
    ]

    config = await service.get(seeded.id)

    assert [j.experiment_name for j in config.associated_jobs] == ["exp"]


async def test_get_scopes_associated_jobs_to_a_non_admin_caller(
    repository: FakeConfigurationRepository,
) -> None:
    owner = uuid4()
    seeded = repository.seed(owner_id=str(owner))
    repository.jobs[seeded.id] = [
        JobTable(
            id=uuid4(),
            user_id=str(owner),
            status=RunStatus.RUNNING,
            config_id=seeded.id,
            dataset_id=uuid4(),
            model="m",
            model_source="huggingface",
            experiment_name="mine",
        ),
        JobTable(
            id=uuid4(),
            user_id=str(uuid4()),
            status=RunStatus.RUNNING,
            config_id=seeded.id,
            dataset_id=uuid4(),
            model="m",
            model_source="huggingface",
            experiment_name="theirs",
        ),
    ]
    principal = Principal(email="u@example.com", provider="session", user_id=owner, is_admin=False)
    service = ConfigurationService(
        repository=repository, principal=principal, autotune=FakeAutotuneCore()
    )

    config = await service.get(seeded.id)

    assert [j.experiment_name for j in config.associated_jobs] == ["mine"]


async def test_list_includes_associated_jobs_per_configuration(
    service: ConfigurationService, repository: FakeConfigurationRepository
) -> None:
    seeded = repository.seed(owner_id=str(ADMIN_ID))
    repository.jobs[seeded.id] = [
        JobTable(
            id=uuid4(),
            user_id=str(ADMIN_ID),
            status=RunStatus.RUNNING,
            config_id=seeded.id,
            dataset_id=uuid4(),
            model="m",
            model_source="huggingface",
            experiment_name="exp",
        ),
    ]

    page = await service.list(limit=20, offset=0)

    assert [j.experiment_name for j in page.items[0].associated_jobs] == ["exp"]
