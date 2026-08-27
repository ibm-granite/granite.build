"""``GET /jobs/{id}`` — the detail response.

Carries the blobs and trials the list response omits, because the view shipped
config_snapshot and output_artifacts on every row.
"""

from __future__ import annotations

from http import HTTPStatus
from uuid import uuid4

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.db.tables import (
    ConfigurationTable,
    GbTaskTable,
    JobTable,
    ResultTable,
    TrialTable,
)
from autotunex.models.status import GbTaskType, RunStatus
from tests.conftest import API


def _snapshot_of(configuration: ConfigurationTable) -> dict[str, object]:
    """Build the snapshot a job captures at submit — mirrors ``JobService.create``."""
    return {
        "name": configuration.name,
        "tuner_type": configuration.tuner_type,
        "rl_tuner_type": configuration.rl_tuner_type,
        "config_data": configuration.config_data,
    }


async def test_detail_includes_the_config_snapshot(
    client: AsyncClient, session: AsyncSession, job: JobTable
) -> None:
    job.config_snapshot = {"name": "as-it-ran", "precision": "bf16"}
    session.add(job)
    await session.commit()

    response = await client.get(f"{API}/jobs/{job.id}?scope=all")

    assert response.json()["config_snapshot"] == {"name": "as-it-ran", "precision": "bf16"}


async def test_detail_includes_output_artifacts(
    client: AsyncClient, session: AsyncSession, job: JobTable
) -> None:
    job.output_artifacts = {"adapter": "cos://bucket/adapter"}
    session.add(job)
    await session.commit()

    response = await client.get(f"{API}/jobs/{job.id}?scope=all")

    assert response.json()["output_artifacts"] == {"adapter": "cos://bucket/adapter"}


async def test_detail_lists_the_job_s_trials(
    client: AsyncClient, session: AsyncSession, job: JobTable
) -> None:
    session.add_all(
        [
            TrialTable(id="t1", job_id=job.id, status=RunStatus.COMPLETED),
            TrialTable(id="t2", job_id=job.id, status=RunStatus.RUNNING),
        ]
    )
    await session.commit()

    response = await client.get(f"{API}/jobs/{job.id}?scope=all")

    assert {trial["id"] for trial in response.json()["trials"]} == {"t1", "t2"}


async def test_trial_metrics_come_from_the_results_row(
    client: AsyncClient, session: AsyncSession, job: JobTable
) -> None:
    """``metrics`` lives on ``results``, not on ``trials``."""
    session.add(TrialTable(id="t1", job_id=job.id, status=RunStatus.COMPLETED))
    await session.commit()
    session.add(
        ResultTable(
            id=uuid4(),
            job_id=job.id,
            trial_id="t1",
            metric="eval_loss",
            metrics={"eval_loss": 0.42},
        )
    )
    await session.commit()

    response = await client.get(f"{API}/jobs/{job.id}?scope=all")

    trial = response.json()["trials"][0]
    assert trial["metrics"] == {"eval_loss": 0.42}
    assert trial["metric"] == "eval_loss"


async def test_a_trial_with_no_result_has_empty_metrics(
    client: AsyncClient, session: AsyncSession, job: JobTable
) -> None:
    session.add(TrialTable(id="t1", job_id=job.id, status=RunStatus.RUNNING))
    await session.commit()

    response = await client.get(f"{API}/jobs/{job.id}?scope=all")

    trial = response.json()["trials"][0]
    assert trial["metrics"] == {}
    assert trial["metric"] is None


async def test_trial_config_replaces_the_scaffold_s_params(
    client: AsyncClient, session: AsyncSession, job: JobTable
) -> None:
    session.add(
        TrialTable(
            id="t1", job_id=job.id, status=RunStatus.COMPLETED, config={"learning_rate": 3e-5}
        )
    )
    await session.commit()

    response = await client.get(f"{API}/jobs/{job.id}?scope=all")

    trial = response.json()["trials"][0]
    assert trial["config"] == {"learning_rate": 3e-5}
    assert "params" not in trial


async def test_detail_still_carries_the_view_shaped_fields(
    client: AsyncClient, job: JobTable
) -> None:
    response = await client.get(f"{API}/jobs/{job.id}?scope=all")

    body = response.json()
    assert body["user"] == "tester@example.com"
    assert body["config_name"] == "lora-sweep"
    assert body["dataset"] == "alpaca"


async def test_detail_of_an_unknown_job_is_a_404_problem_detail(client: AsyncClient) -> None:
    response = await client.get(f"{API}/jobs/{uuid4()}")

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.headers["content-type"].startswith("application/problem+json")


async def test_detail_carries_model_source_and_num_trials(
    client: AsyncClient, session: AsyncSession, job: JobTable
) -> None:
    session.add(TrialTable(id="t1", job_id=job.id, status=RunStatus.COMPLETED))
    await session.commit()

    response = await client.get(f"{API}/jobs/{job.id}?scope=all")

    body = response.json()
    assert body["model_source"] == "huggingface"
    assert body["num_trials"] == 1


async def test_detail_is_stale_is_false_when_the_snapshot_matches_the_live_config(
    client: AsyncClient, session: AsyncSession, job: JobTable, configuration: ConfigurationTable
) -> None:
    job.config_snapshot = _snapshot_of(configuration)
    session.add(job)
    await session.commit()

    response = await client.get(f"{API}/jobs/{job.id}?scope=all")

    assert response.status_code == 200
    assert response.json().get("is_stale") is False


async def test_detail_is_stale_is_true_when_the_configuration_data_changed(
    client: AsyncClient, session: AsyncSession, job: JobTable, configuration: ConfigurationTable
) -> None:
    job.config_snapshot = _snapshot_of(configuration)
    session.add(job)
    await session.commit()

    configuration.config_data = {"learning_rate": {"kind": "float", "low": 1e-5, "high": 1e-2}}
    session.add(configuration)
    await session.commit()

    response = await client.get(f"{API}/jobs/{job.id}?scope=all")

    assert response.json().get("is_stale") is True


async def test_detail_is_stale_stays_false_after_a_pure_rename(
    client: AsyncClient, session: AsyncSession, job: JobTable, configuration: ConfigurationTable
) -> None:
    """A rename bumps ``configurations.updated_at`` but changes no behaviour.

    This is the case the content comparison exists to get right, and the old
    timestamp proxy got wrong: renaming a configuration must not flag every
    historical job that ran its (unchanged) settings.
    """
    job.config_snapshot = _snapshot_of(configuration)
    session.add(job)
    await session.commit()

    configuration.name = "lora-sweep-renamed"
    session.add(configuration)
    await session.commit()

    response = await client.get(f"{API}/jobs/{job.id}?scope=all")

    assert response.json().get("is_stale") is False


async def test_detail_is_stale_is_true_when_the_tuner_type_changed(
    client: AsyncClient, session: AsyncSession, job: JobTable, configuration: ConfigurationTable
) -> None:
    job.config_snapshot = _snapshot_of(configuration)
    session.add(job)
    await session.commit()

    configuration.tuner_type = "hyperband"
    session.add(configuration)
    await session.commit()

    response = await client.get(f"{API}/jobs/{job.id}?scope=all")

    assert response.json().get("is_stale") is True


async def test_detail_is_stale_is_false_when_the_job_has_no_snapshot(
    client: AsyncClient, job: JobTable
) -> None:
    """A job with no ``config_snapshot`` has no baseline to compare, so is never stale."""
    response = await client.get(f"{API}/jobs/{job.id}?scope=all")

    assert response.json().get("is_stale") is False


async def test_detail_nests_all_tasks_with_the_view_aliases(
    client: AsyncClient, session: AsyncSession, job: JobTable
) -> None:
    session.add(
        GbTaskTable(
            id=uuid4(),
            job_id=job.id,
            status=RunStatus.RUNNING,
            type=GbTaskType.RITS,
            pr_url="https://github.example/pr/1",
            rits_url="https://rits.example/x",
            started_at="2026-07-29 10:12:00",
        )
    )
    session.add_all(
        GbTaskTable(id=uuid4(), job_id=job.id, status=RunStatus.PENDING, type=GbTaskType.TUNING)
        for _ in range(2)
    )
    await session.commit()

    response = await client.get(f"{API}/jobs/{job.id}?scope=all")

    tasks = response.json()["tasks"]
    assert len(tasks) == 3
    rits = next(t for t in tasks if t["task_type"] == "RITS")
    assert rits["task_status"] == "running"
    assert rits["github_pr_url"] == "https://github.example/pr/1"
    assert rits["rits_url"] == "https://rits.example/x"
    assert rits["task_started_at"] == "2026-07-29 10:12:00"
