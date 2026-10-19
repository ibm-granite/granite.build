"""Eligibility rules for ``SqlAlchemyJobRepository.tuned_models``.

A tuned model is pickable as a new job's base only when its job completed, its
TUNING task published an ``hf://`` repo in an allowlisted namespace, and the
tuning produced full weights (online RL, or ``sft``/``none``). Each test adds
only the rows its behavior depends on, on top of the shared ``user`` /
``configuration`` / ``dataset`` fixtures.
"""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.db.repositories.sqlalchemy import SqlAlchemyJobRepository
from autotunex.db.tables import (
    ConfigurationTable,
    DatasetTable,
    GbTaskTable,
    JobTable,
    UserTable,
)
from autotunex.models.status import GbTaskType, RunStatus

NAMESPACES = ["example-org"]
URI = "hf://huggingface.co/models/example-org/autotunex_aaaa0001"


async def _tuned_job(
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    dataset: DatasetTable,
    *,
    status: RunStatus = RunStatus.COMPLETED,
    tuning_type: str | None = "sft",
    rl_tuner_type: str | None = None,
    artifact_uris: tuple[str | None, ...] = (URI,),
    experiment_name: str = "sft-granite",
    with_snapshot: bool = True,
    task_type: GbTaskType = GbTaskType.TUNING,
) -> JobTable:
    """Persist a job plus one ``task_type`` task per entry in ``artifact_uris``."""
    job = JobTable(
        id=uuid4(),
        user_id=str(user.id),
        status=status,
        config_id=configuration.id,
        dataset_id=dataset.id,
        model="ibm-granite/granite-4.0-h-micro",
        model_source="huggingface",
        experiment_name=experiment_name,
        tuning_type=tuning_type,
        config_snapshot=(
            {
                "name": "c",
                "tuner_type": tuning_type,
                "rl_tuner_type": rl_tuner_type,
                "config_data": {},
            }
            if with_snapshot
            else None
        ),
    )
    session.add(job)
    for uri in artifact_uris:
        session.add(
            GbTaskTable(
                job_id=job.id,
                type=task_type,
                status=RunStatus.COMPLETED,
                artifact_uri=uri,
                updated_at="2026-09-20T10:00:00Z",
            )
        )
    await session.commit()
    return job


async def _ids(session: AsyncSession, **kwargs: object) -> list[object]:
    """Return the eligible job ids for ``NAMESPACES`` (overridable via kwargs)."""
    kwargs.setdefault("namespaces", NAMESPACES)
    kwargs.setdefault("limit", 100)
    kwargs.setdefault("offset", 0)
    rows, _total = await SqlAlchemyJobRepository(session).tuned_models(**kwargs)  # type: ignore[arg-type]
    return [job.id for job, _finished_at, _uri in rows]


async def test_tuned_models_include_a_completed_sft_job_with_an_allowlisted_repo(
    session: AsyncSession, user: UserTable, configuration: ConfigurationTable, dataset: DatasetTable
) -> None:
    job = await _tuned_job(session, user, configuration, dataset)

    rows, total = await SqlAlchemyJobRepository(session).tuned_models(
        namespaces=NAMESPACES, limit=20, offset=0
    )

    assert total == 1
    assert [(r[0].id, r[1], r[2]) for r in rows] == [(job.id, "2026-09-20T10:00:00Z", URI)]


async def test_tuned_models_include_online_rl_whatever_the_stored_tuning_type(
    session: AsyncSession, user: UserTable, configuration: ConfigurationTable, dataset: DatasetTable
) -> None:
    job = await _tuned_job(
        session, user, configuration, dataset, tuning_type="lora", rl_tuner_type="grpo"
    )

    assert await _ids(session) == [job.id]


async def test_tuned_models_match_tuner_types_case_insensitively(
    session: AsyncSession, user: UserTable, configuration: ConfigurationTable, dataset: DatasetTable
) -> None:
    sft = await _tuned_job(session, user, configuration, dataset, tuning_type="SFT")
    grpo = await _tuned_job(
        session, user, configuration, dataset, tuning_type=None, rl_tuner_type="GRPO"
    )

    assert set(await _ids(session)) == {sft.id, grpo.id}


async def test_tuned_models_include_offline_rl_on_top_of_sft(
    session: AsyncSession, user: UserTable, configuration: ConfigurationTable, dataset: DatasetTable
) -> None:
    job = await _tuned_job(
        session, user, configuration, dataset, tuning_type="sft", rl_tuner_type="dpo"
    )

    assert await _ids(session) == [job.id]


async def test_tuned_models_fall_back_to_the_configuration_rl_type_without_a_snapshot(
    session: AsyncSession, user: UserTable, dataset: DatasetTable
) -> None:
    grpo_config = ConfigurationTable(
        id=uuid4(),
        user_id=str(user.id),
        name="grpo-sweep",
        tuner_type=None,
        rl_tuner_type="grpo",
        config_data={"x": 1},
    )
    session.add(grpo_config)
    await session.commit()
    job = await _tuned_job(
        session, user, grpo_config, dataset, tuning_type=None, with_snapshot=False
    )

    assert await _ids(session) == [job.id]


async def test_tuned_models_trust_the_snapshot_rl_type_over_an_edited_configuration(
    session: AsyncSession, user: UserTable, dataset: DatasetTable
) -> None:
    # The job ran as plain LoRA (the launch reads only the snapshot); its
    # configuration was edited to GRPO afterwards. The output is an adapter.
    edited = ConfigurationTable(
        id=uuid4(),
        user_id=str(user.id),
        name="edited-to-grpo",
        tuner_type="lora",
        rl_tuner_type="grpo",
        config_data={"x": 1},
    )
    session.add(edited)
    await session.commit()
    await _tuned_job(session, user, edited, dataset, tuning_type="lora", rl_tuner_type=None)

    assert await _ids(session) == []


async def test_tuned_models_exclude_adapter_outputs(
    session: AsyncSession, user: UserTable, configuration: ConfigurationTable, dataset: DatasetTable
) -> None:
    await _tuned_job(session, user, configuration, dataset, tuning_type="lora")
    await _tuned_job(session, user, configuration, dataset, tuning_type="lora", rl_tuner_type="dpo")

    assert await _ids(session) == []


async def test_tuned_models_exclude_jobs_that_have_not_completed(
    session: AsyncSession, user: UserTable, configuration: ConfigurationTable, dataset: DatasetTable
) -> None:
    await _tuned_job(session, user, configuration, dataset, status=RunStatus.RUNNING)

    assert await _ids(session) == []


async def test_tuned_models_exclude_jobs_without_a_published_hf_repo(
    session: AsyncSession, user: UserTable, configuration: ConfigurationTable, dataset: DatasetTable
) -> None:
    await _tuned_job(session, user, configuration, dataset, artifact_uris=())
    await _tuned_job(session, user, configuration, dataset, artifact_uris=(None,))
    await _tuned_job(
        session, user, configuration, dataset, artifact_uris=("file:///data/autotune_x/",)
    )

    assert await _ids(session) == []


async def test_tuned_models_exclude_repos_outside_the_allowlist(
    session: AsyncSession, user: UserTable, configuration: ConfigurationTable, dataset: DatasetTable
) -> None:
    await _tuned_job(
        session,
        user,
        configuration,
        dataset,
        artifact_uris=("hf://huggingface.co/models/someone-else/autotunex_bbbb0002",),
    )

    assert await _ids(session) == []


async def test_tuned_models_exclude_a_locator_with_no_repo_name(
    session: AsyncSession, user: UserTable, configuration: ConfigurationTable, dataset: DatasetTable
) -> None:
    await _tuned_job(
        session,
        user,
        configuration,
        dataset,
        artifact_uris=("hf://huggingface.co/models/example-org/",),
    )

    assert await _ids(session) == []


async def test_tuned_models_treat_an_allowlisted_namespace_literally_not_as_a_pattern(
    session: AsyncSession, user: UserTable, configuration: ConfigurationTable, dataset: DatasetTable
) -> None:
    await _tuned_job(
        session,
        user,
        configuration,
        dataset,
        artifact_uris=("hf://huggingface.co/models/exampleXorg/autotunex_cccc0003",),
    )

    assert await _ids(session, namespaces=["example_org"]) == []


async def test_tuned_models_exclude_a_locator_nested_below_owner_and_name(
    session: AsyncSession, user: UserTable, configuration: ConfigurationTable, dataset: DatasetTable
) -> None:
    await _tuned_job(
        session,
        user,
        configuration,
        dataset,
        artifact_uris=("hf://huggingface.co/models/example-org/autotunex_x/extra",),
    )

    assert await _ids(session) == []


async def test_tuned_models_include_a_locator_with_a_trailing_slash(
    session: AsyncSession, user: UserTable, configuration: ConfigurationTable, dataset: DatasetTable
) -> None:
    job = await _tuned_job(
        session,
        user,
        configuration,
        dataset,
        artifact_uris=("hf://huggingface.co/models/example-org/autotunex_y/",),
    )

    assert await _ids(session) == [job.id]


async def test_tuned_models_exclude_an_hf_repo_published_by_a_non_tuning_task(
    session: AsyncSession, user: UserTable, configuration: ConfigurationTable, dataset: DatasetTable
) -> None:
    await _tuned_job(session, user, configuration, dataset, task_type=GbTaskType.DOWNLOAD)

    assert await _ids(session) == []


async def test_tuned_models_return_one_row_for_a_job_with_two_tuning_tasks(
    session: AsyncSession, user: UserTable, configuration: ConfigurationTable, dataset: DatasetTable
) -> None:
    job = await _tuned_job(session, user, configuration, dataset, artifact_uris=(URI, URI))

    rows, total = await SqlAlchemyJobRepository(session).tuned_models(
        namespaces=NAMESPACES, limit=20, offset=0
    )

    assert total == 1
    assert [r[0].id for r in rows] == [job.id]


async def test_tuned_models_return_nothing_for_an_empty_allowlist(
    session: AsyncSession, user: UserTable, configuration: ConfigurationTable, dataset: DatasetTable
) -> None:
    await _tuned_job(session, user, configuration, dataset)

    rows, total = await SqlAlchemyJobRepository(session).tuned_models(
        namespaces=[], limit=20, offset=0
    )

    assert (list(rows), total) == ([], 0)


async def test_tuned_models_filter_to_the_given_owner(
    session: AsyncSession, user: UserTable, configuration: ConfigurationTable, dataset: DatasetTable
) -> None:
    other = UserTable(id=uuid4(), email="not-the-owner@example.com", role="user")
    session.add(other)
    await session.commit()
    mine = await _tuned_job(session, user, configuration, dataset)
    await _tuned_job(session, other, configuration, dataset)

    assert await _ids(session, owner_id=user.id) == [mine.id]


async def test_tuned_models_search_experiment_name_base_model_and_repo(
    session: AsyncSession, user: UserTable, configuration: ConfigurationTable, dataset: DatasetTable
) -> None:
    by_name = await _tuned_job(session, user, configuration, dataset, experiment_name="needle-run")
    by_repo = await _tuned_job(
        session,
        user,
        configuration,
        dataset,
        artifact_uris=("hf://huggingface.co/models/example-org/autotunex_needle01",),
    )
    await _tuned_job(session, user, configuration, dataset, experiment_name="other")

    assert set(await _ids(session, q="NEEDLE")) == {by_name.id, by_repo.id}
    assert len(await _ids(session, q="granite-4.0-h-micro")) == 3


async def test_tuned_models_search_ignores_the_shared_locator_prefix(
    session: AsyncSession, user: UserTable, configuration: ConfigurationTable, dataset: DatasetTable
) -> None:
    await _tuned_job(session, user, configuration, dataset)

    assert await _ids(session, q="huggingface") == []


async def test_tuned_models_total_counts_every_match_not_just_the_page(
    session: AsyncSession, user: UserTable, configuration: ConfigurationTable, dataset: DatasetTable
) -> None:
    for _ in range(3):
        await _tuned_job(session, user, configuration, dataset)

    rows, total = await SqlAlchemyJobRepository(session).tuned_models(
        namespaces=NAMESPACES, limit=2, offset=0
    )

    assert (len(rows), total) == (2, 3)
