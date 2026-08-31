"""SQLAlchemy configuration repository against a real in-memory database.

These tests exist because the two things most likely to break — the
``UNIQUE (user_id, name)`` collision and the ``ON DELETE RESTRICT`` from
``jobs.config_id`` — are database behaviours a fake cannot prove. The ``engine``
fixture enables ``PRAGMA foreign_keys=ON`` (see ``tests/conftest.py``), so the
RESTRICT is genuinely enforced here rather than silently inert.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.core.exceptions import ConfigurationInUseError, ConfigurationNameConflictError
from autotunex.db.repositories.sqlalchemy import SqlAlchemyConfigurationRepository
from autotunex.db.tables import ConfigurationTable, DatasetTable, JobTable, UserTable
from autotunex.models.status import RunStatus

SPACE: dict[str, Any] = {"lr": {"kind": "float", "low": 1e-6, "high": 1e-3}}


async def _make_user(session: AsyncSession, *, email: str = "a@example.com") -> UserTable:
    """Persist an owner on its own, before any dependents reference it."""
    user = UserTable(id=uuid4(), email=email, role="user")
    session.add(user)
    await session.commit()
    return user


async def _seed_configuration(
    session: AsyncSession, *, user_id: str, name: str = "cfg"
) -> ConfigurationTable:
    configuration = ConfigurationTable(
        id=uuid4(), user_id=user_id, name=name, config_data=dict(SPACE)
    )
    session.add(configuration)
    await session.commit()
    return configuration


async def test_create_persists_and_assigns_an_id(session: AsyncSession) -> None:
    user = await _make_user(session)
    repository = SqlAlchemyConfigurationRepository(session)

    created = await repository.create(
        user_id=str(user.id),
        name="lora-sweep",
        tuner_type="optuna",
        rl_tuner_type=None,
        config_data=dict(SPACE),
    )

    assert isinstance(created.id, UUID)
    assert created.name == "lora-sweep"


async def test_create_is_visible_in_a_later_get(session: AsyncSession) -> None:
    user = await _make_user(session)
    repository = SqlAlchemyConfigurationRepository(session)
    created = await repository.create(
        user_id=str(user.id), name="c", tuner_type=None, rl_tuner_type=None, config_data=dict(SPACE)
    )

    found = await repository.get(created.id)

    assert found is not None
    assert found.id == created.id


async def test_create_with_a_duplicate_name_for_one_owner_conflicts(session: AsyncSession) -> None:
    user = await _make_user(session)
    repository = SqlAlchemyConfigurationRepository(session)
    await repository.create(
        user_id=str(user.id),
        name="dup",
        tuner_type=None,
        rl_tuner_type=None,
        config_data=dict(SPACE),
    )

    with pytest.raises(ConfigurationNameConflictError):
        await repository.create(
            user_id=str(user.id),
            name="dup",
            tuner_type=None,
            rl_tuner_type=None,
            config_data=dict(SPACE),
        )


async def test_the_same_name_under_two_owners_is_allowed(session: AsyncSession) -> None:
    """The UNIQUE constraint is on the pair, not the name alone."""
    one = await _make_user(session, email="one@example.com")
    two = await _make_user(session, email="two@example.com")
    repository = SqlAlchemyConfigurationRepository(session)
    await repository.create(
        user_id=str(one.id),
        name="shared",
        tuner_type=None,
        rl_tuner_type=None,
        config_data=dict(SPACE),
    )

    created = await repository.create(
        user_id=str(two.id),
        name="shared",
        tuner_type=None,
        rl_tuner_type=None,
        config_data=dict(SPACE),
    )

    assert created.name == "shared"


async def test_get_with_a_non_matching_owner_id_returns_none(session: AsyncSession) -> None:
    user = await _make_user(session)
    configuration = await _seed_configuration(session, user_id=str(user.id))
    repository = SqlAlchemyConfigurationRepository(session)

    found = await repository.get(configuration.id, owner_id=uuid4())

    assert found is None


async def test_list_with_an_owner_id_filters_both_page_and_total(session: AsyncSession) -> None:
    mine = await _make_user(session, email="mine@example.com")
    theirs = await _make_user(session, email="theirs@example.com")
    await _seed_configuration(session, user_id=str(mine.id), name="a")
    await _seed_configuration(session, user_id=str(theirs.id), name="b")
    repository = SqlAlchemyConfigurationRepository(session)

    configs, total = await repository.list(limit=20, offset=0, owner_id=mine.id)

    assert total == 1
    assert [config.user_id for config in configs] == [str(mine.id)]


async def test_list_filters_configurations_by_name_substring(session: AsyncSession) -> None:
    user = await _make_user(session)
    await _seed_configuration(session, user_id=str(user.id), name="lora-sweep")
    await _seed_configuration(session, user_id=str(user.id), name="full-finetune")
    repository = SqlAlchemyConfigurationRepository(session)

    configs, total = await repository.list(limit=20, offset=0, q="LORA")

    assert total == 1
    assert [c.name for c in configs] == ["lora-sweep"]


async def test_list_paginates(session: AsyncSession) -> None:
    user = await _make_user(session)
    for index in range(3):
        await _seed_configuration(session, user_id=str(user.id), name=f"c{index}")
    repository = SqlAlchemyConfigurationRepository(session)

    configs, total = await repository.list(limit=2, offset=0)

    assert total == 3
    assert len(configs) == 2


async def test_update_replaces_fields_and_keeps_the_owner(session: AsyncSession) -> None:
    user = await _make_user(session)
    configuration = await _seed_configuration(session, user_id=str(user.id), name="before")
    repository = SqlAlchemyConfigurationRepository(session)
    new_space = {"rank": {"kind": "int", "low": 4, "high": 64, "step": 4}}

    updated = await repository.update(
        configuration.id,
        name="after",
        tuner_type="optuna",
        rl_tuner_type=None,
        config_data=new_space,
    )

    assert updated is not None
    assert updated.name == "after"
    assert updated.config_data == new_space
    assert updated.user_id == str(user.id)


async def test_update_of_an_unknown_configuration_returns_none(session: AsyncSession) -> None:
    repository = SqlAlchemyConfigurationRepository(session)

    result = await repository.update(
        uuid4(), name="x", tuner_type=None, rl_tuner_type=None, config_data=dict(SPACE)
    )

    assert result is None


async def test_update_scoped_to_a_non_owner_returns_none(session: AsyncSession) -> None:
    user = await _make_user(session)
    configuration = await _seed_configuration(session, user_id=str(user.id))
    repository = SqlAlchemyConfigurationRepository(session)

    result = await repository.update(
        configuration.id,
        owner_id=uuid4(),
        name="x",
        tuner_type=None,
        rl_tuner_type=None,
        config_data=dict(SPACE),
    )

    assert result is None


async def test_update_into_a_duplicate_name_conflicts(session: AsyncSession) -> None:
    user = await _make_user(session)
    await _seed_configuration(session, user_id=str(user.id), name="taken")
    mine = await _seed_configuration(session, user_id=str(user.id), name="mine")
    repository = SqlAlchemyConfigurationRepository(session)

    with pytest.raises(ConfigurationNameConflictError):
        await repository.update(
            mine.id, name="taken", tuner_type=None, rl_tuner_type=None, config_data=dict(SPACE)
        )


async def test_delete_removes_the_row(session: AsyncSession) -> None:
    user = await _make_user(session)
    configuration = await _seed_configuration(session, user_id=str(user.id))
    repository = SqlAlchemyConfigurationRepository(session)

    deleted = await repository.delete(configuration.id)

    assert deleted is True
    assert await repository.get(configuration.id) is None


async def test_delete_of_an_unknown_configuration_returns_false(session: AsyncSession) -> None:
    repository = SqlAlchemyConfigurationRepository(session)

    assert await repository.delete(uuid4()) is False


async def test_delete_scoped_to_a_non_owner_returns_false(session: AsyncSession) -> None:
    user = await _make_user(session)
    configuration = await _seed_configuration(session, user_id=str(user.id))
    repository = SqlAlchemyConfigurationRepository(session)

    deleted = await repository.delete(configuration.id, owner_id=uuid4())

    assert deleted is False
    assert await repository.get(configuration.id) is not None


async def test_delete_of_a_configuration_a_job_references_conflicts(session: AsyncSession) -> None:
    """ON DELETE RESTRICT from jobs.config_id, surfaced as a clean 409."""
    user = await _make_user(session)
    configuration = await _seed_configuration(session, user_id=str(user.id))
    dataset = DatasetTable(
        id=uuid4(), user_id=str(user.id), name="ds", description="In-use fixture."
    )
    job = JobTable(
        id=uuid4(),
        user_id=str(user.id),
        status=RunStatus.PENDING,
        config_id=configuration.id,
        dataset_id=dataset.id,
        model="m",
        model_source="huggingface",
        experiment_name="exp",
    )
    session.add_all([dataset, job])
    await session.commit()
    repository = SqlAlchemyConfigurationRepository(session)

    with pytest.raises(ConfigurationInUseError):
        await repository.delete(configuration.id)


async def test_jobs_for_config_returns_referencing_jobs_scoped_to_owner(
    session: AsyncSession,
) -> None:
    owner = UserTable(id=uuid4(), email="owner@example.com", role="user")
    other = UserTable(id=uuid4(), email="other@example.com", role="user")
    session.add_all([owner, other])
    await session.commit()
    repository = SqlAlchemyConfigurationRepository(session)
    configuration = await repository.create(
        user_id=str(owner.id),
        name="cfg",
        tuner_type=None,
        rl_tuner_type=None,
        config_data={"k": 1},
    )
    dataset = DatasetTable(id=uuid4(), user_id=str(owner.id), name="ds")
    my_job = JobTable(
        id=uuid4(),
        user_id=str(owner.id),
        status=RunStatus.RUNNING,
        config_id=configuration.id,
        dataset_id=dataset.id,
        model="m",
        model_source="huggingface",
        experiment_name="mine",
    )
    their_job = JobTable(
        id=uuid4(),
        user_id=str(other.id),
        status=RunStatus.RUNNING,
        config_id=configuration.id,
        dataset_id=dataset.id,
        model="m",
        model_source="huggingface",
        experiment_name="theirs",
    )
    session.add_all([dataset, my_job, their_job])
    await session.commit()

    scoped = await repository.jobs_for_config([configuration.id], owner_id=owner.id)
    unscoped = await repository.jobs_for_config([configuration.id])
    empty = await repository.jobs_for_config([])

    assert {j.experiment_name for j in scoped[configuration.id]} == {"mine"}
    assert {j.experiment_name for j in unscoped[configuration.id]} == {"mine", "theirs"}
    assert empty == {}
