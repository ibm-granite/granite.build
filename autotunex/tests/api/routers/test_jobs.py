"""Job endpoints.

``GET /jobs`` and ``GET /jobs/{id}`` report what exists; ``POST /jobs``
submits a new one, owned by the calling principal — see
``docs/superpowers/specs/2026-07-29-autotunex-jobs-view-design.md`` for why the
read path was rebuilt on the real schema, and Task A5's brief for why
submission was restored on top of it.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.core.constants import SYSTEM_USER_ID
from autotunex.db.tables import ConfigurationTable, DatasetTable, JobTable, UserTable
from autotunex.models.auth import Principal
from autotunex.models.status import DatasetStatus, RunStatus
from tests.conftest import API

PROBLEM_JSON = "application/problem+json"


def _act_as(as_principal: Callable[[Principal], None], user: UserTable) -> None:
    """Resolve every request to ``user`` — a provisioned, non-admin owner."""
    as_principal(Principal(email=user.email, provider="session", user_id=user.id, is_admin=False))


async def test_list_jobs_returns_an_empty_page_when_there_are_none(client: AsyncClient) -> None:
    response = await client.get(f"{API}/jobs")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"items": [], "total": 0, "limit": 20, "offset": 0}


async def test_get_unknown_job_returns_a_problem_detail(client: AsyncClient) -> None:
    response = await client.get(f"{API}/jobs/{uuid4()}")

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.headers["content-type"].startswith("application/problem+json")


async def test_get_unknown_job_problem_detail_names_the_status(client: AsyncClient) -> None:
    response = await client.get(f"{API}/jobs/{uuid4()}")

    assert response.json()["status"] == HTTPStatus.NOT_FOUND


async def test_list_jobs_rejects_a_limit_above_100(client: AsyncClient) -> None:
    response = await client.get(f"{API}/jobs", params={"limit": 101})

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


async def test_list_jobs_rejects_a_negative_offset(client: AsyncClient) -> None:
    response = await client.get(f"{API}/jobs", params={"offset": -1})

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


async def test_list_jobs_passes_q_filter_to_narrow_results(client: AsyncClient) -> None:
    # An unmatched q must return an empty page with total 0, proving q reaches the query.
    response = await client.get(f"{API}/jobs", params={"q": "zzz-no-such-job"})

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 0
    assert body["items"] == []


# Create.


async def test_create_returns_201_and_the_created_job(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    configuration: ConfigurationTable,
    ready_dataset: DatasetTable,
) -> None:
    _act_as(as_principal, user)

    response = await client.post(
        f"{API}/jobs",
        json={
            "config_id": str(configuration.id),
            "dataset_id": str(ready_dataset.id),
            "model": "ibm/granite",
            "experiment_name": "exp one",
        },
    )

    assert response.status_code == HTTPStatus.CREATED
    body = response.json()
    assert body["status"] == "pending"
    assert body["user_id"] == str(user.id)
    assert body["tuning_type"] == configuration.tuner_type


async def test_create_by_unprovisioned_caller_is_403(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    configuration: ConfigurationTable,
    ready_dataset: DatasetTable,
) -> None:
    as_principal(
        Principal(email="ghost@example.com", provider="session", user_id=None, is_admin=False)
    )

    response = await client.post(
        f"{API}/jobs",
        json={
            "config_id": str(configuration.id),
            "dataset_id": str(ready_dataset.id),
            "model": "m",
            "experiment_name": "e",
        },
    )

    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_create_against_another_users_dataset_is_404(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    configuration: ConfigurationTable,
    ready_dataset: DatasetTable,
) -> None:
    as_principal(
        Principal(email="other@example.com", provider="session", user_id=uuid4(), is_admin=False)
    )

    response = await client.post(
        f"{API}/jobs",
        json={
            "config_id": str(configuration.id),
            "dataset_id": str(ready_dataset.id),
            "model": "m",
            "experiment_name": "e",
        },
    )

    assert response.status_code == HTTPStatus.NOT_FOUND


async def test_create_with_unready_dataset_is_409(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    configuration: ConfigurationTable,
    dataset: DatasetTable,
) -> None:
    _act_as(as_principal, user)

    response = await client.post(
        f"{API}/jobs",
        json={
            "config_id": str(configuration.id),
            "dataset_id": str(dataset.id),
            "model": "m",
            "experiment_name": "e",
        },
    )

    assert response.status_code == HTTPStatus.CONFLICT


async def test_create_with_dmf_model_source_is_422(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    configuration: ConfigurationTable,
    ready_dataset: DatasetTable,
) -> None:
    _act_as(as_principal, user)

    response = await client.post(
        f"{API}/jobs",
        json={
            "config_id": str(configuration.id),
            "dataset_id": str(ready_dataset.id),
            "model": "m",
            "experiment_name": "e",
            "model_source": "dmf",
        },
    )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


async def test_create_cannot_reference_another_users_real_config_or_dataset(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    session: AsyncSession,
) -> None:
    """Cross-user isolation: a real, existing resource of user B's is still a 404 for user A.

    This is stronger than the unknown-id 404 above — it proves the scoped
    lookup, not just a missing row, is what drives the result.
    """
    user_b = UserTable(id=uuid4(), email="userb@example.com", role="user")
    session.add(user_b)
    await session.commit()
    config_b = ConfigurationTable(
        id=uuid4(),
        user_id=str(user_b.id),
        name="b-config",
        tuner_type="optuna",
        rl_tuner_type=None,
        config_data={"learning_rate": {"kind": "float", "low": 1e-6, "high": 1e-3, "log": True}},
    )
    dataset_b = DatasetTable(
        id=uuid4(),
        user_id=str(user_b.id),
        name="b-dataset",
        description="Owned by user B.",
        data_format="jsonl",
        status=DatasetStatus.READY,
        train_records=100,
        train_file_size=2048,
    )
    session.add_all([config_b, dataset_b])
    await session.commit()
    await session.refresh(dataset_b, ["train_file", "validation_file"])
    _act_as(as_principal, user)

    response = await client.post(
        f"{API}/jobs",
        json={
            "config_id": str(config_b.id),
            "dataset_id": str(dataset_b.id),
            "model": "m",
            "experiment_name": "e",
        },
    )

    assert response.status_code == HTTPStatus.NOT_FOUND


async def test_create_against_system_owned_config_and_dataset_succeeds(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    session: AsyncSession,
) -> None:
    """A user may submit a tuning against curated, system-owned starter content."""
    session.add(UserTable(id=SYSTEM_USER_ID, email="system@autotunex.local", role="user"))
    await session.commit()
    system_config = ConfigurationTable(
        id=uuid4(),
        user_id=str(SYSTEM_USER_ID),
        name="starter-config",
        tuner_type="optuna",
        rl_tuner_type=None,
        config_data={"learning_rate": {"kind": "float", "low": 1e-6, "high": 1e-3, "log": True}},
    )
    system_dataset = DatasetTable(
        id=uuid4(),
        user_id=str(SYSTEM_USER_ID),
        name="starter-dataset",
        description="Curated starter dataset.",
        data_format="jsonl",
        status=DatasetStatus.READY,
        train_records=100,
        train_file_size=2048,
    )
    session.add_all([system_config, system_dataset])
    await session.commit()
    await session.refresh(system_dataset, ["train_file", "validation_file"])
    _act_as(as_principal, user)

    response = await client.post(
        f"{API}/jobs",
        json={
            "config_id": str(system_config.id),
            "dataset_id": str(system_dataset.id),
            "model": "m",
            "experiment_name": "e",
        },
    )

    assert response.status_code == HTTPStatus.CREATED
    assert response.json()["user_id"] == str(user.id)


async def test_create_online_rl_with_reward_function_persists_it(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    ready_dataset: DatasetTable,
    session: AsyncSession,
) -> None:
    """The reward function is not in the response body — verify via the DB row instead."""
    rl_config = ConfigurationTable(
        id=uuid4(),
        user_id=str(user.id),
        name="ppo-sweep",
        tuner_type="optuna",
        rl_tuner_type="ppo",
        config_data={"x": 1},
    )
    session.add(rl_config)
    await session.commit()
    _act_as(as_principal, user)
    reward_code = "def compute_score():\n    return 1.0\n"

    response = await client.post(
        f"{API}/jobs",
        json={
            "config_id": str(rl_config.id),
            "dataset_id": str(ready_dataset.id),
            "model": "m",
            "experiment_name": "e",
            "reward_function_code": reward_code,
        },
    )

    assert response.status_code == HTTPStatus.CREATED
    body = response.json()
    job_row = await session.get(JobTable, UUID(body["id"]))
    assert job_row is not None
    assert job_row.reward_function_code == reward_code
    assert job_row.reward_function_name == "compute_score"


# Delete.


async def test_delete_job_returns_204_and_removes_it(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
) -> None:
    _act_as(as_principal, user)

    response = await client.delete(f"{API}/jobs/{job.id}")

    assert response.status_code == HTTPStatus.NO_CONTENT
    assert response.content == b""
    follow_up = await client.get(f"{API}/jobs/{job.id}")
    assert follow_up.status_code == HTTPStatus.NOT_FOUND


async def test_delete_unknown_job_returns_404(
    client: AsyncClient, as_principal: Callable[[Principal], None], user: UserTable
) -> None:
    _act_as(as_principal, user)

    response = await client.delete(f"{API}/jobs/{uuid4()}")

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_delete_running_job_returns_204_and_removes_it(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
    session: AsyncSession,
) -> None:
    """Delete now auto-cancels live work first, then removes the row.

    The default ``NoOpJobRunner.cancel`` is itself a no-op, so this only
    exercises the service's own status write — no more 409 for a running job.
    """
    _act_as(as_principal, user)
    job.status = RunStatus.RUNNING
    await session.commit()

    response = await client.delete(f"{API}/jobs/{job.id}")

    assert response.status_code == HTTPStatus.NO_CONTENT
    follow_up = await client.get(f"{API}/jobs/{job.id}")
    assert follow_up.status_code == HTTPStatus.NOT_FOUND


async def test_delete_paused_job_returns_204_and_removes_it(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
    session: AsyncSession,
) -> None:
    _act_as(as_principal, user)
    job.status = RunStatus.PAUSED
    await session.commit()

    response = await client.delete(f"{API}/jobs/{job.id}")

    assert response.status_code == HTTPStatus.NO_CONTENT
    follow_up = await client.get(f"{API}/jobs/{job.id}")
    assert follow_up.status_code == HTTPStatus.NOT_FOUND


async def test_delete_another_users_job_is_404(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    job: JobTable,
) -> None:
    as_principal(
        Principal(email="other@example.com", provider="session", user_id=uuid4(), is_admin=False)
    )

    response = await client.delete(f"{API}/jobs/{job.id}")

    assert response.status_code == HTTPStatus.NOT_FOUND


# Cancel.


async def test_post_cancel_drives_running_job_to_terminated(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
    session: AsyncSession,
) -> None:
    """Drives a running job to terminated via the cancel endpoint.

    The default ``NoOpJobRunner.cancel`` is itself a no-op, so this only
    exercises the service's own status write — no runner override is needed.
    """
    _act_as(as_principal, user)
    job.status = RunStatus.RUNNING
    await session.commit()

    response = await client.post(f"{API}/jobs/{job.id}/cancel")

    assert response.status_code == HTTPStatus.OK
    assert response.json()["status"] == "terminated"


async def test_post_cancel_on_completed_job_is_409(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
    session: AsyncSession,
) -> None:
    _act_as(as_principal, user)
    job.status = RunStatus.COMPLETED
    await session.commit()

    response = await client.post(f"{API}/jobs/{job.id}/cancel")

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_post_cancel_unknown_job_is_404(
    client: AsyncClient, as_principal: Callable[[Principal], None], user: UserTable
) -> None:
    _act_as(as_principal, user)

    response = await client.post(f"{API}/jobs/{uuid4()}/cancel")

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_post_cancel_another_users_job_is_404(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    job: JobTable,
) -> None:
    as_principal(
        Principal(email="other@example.com", provider="session", user_id=uuid4(), is_admin=False)
    )

    response = await client.post(f"{API}/jobs/{job.id}/cancel")

    assert response.status_code == HTTPStatus.NOT_FOUND


async def test_post_cancel_is_idempotent_for_already_terminated_job(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
    session: AsyncSession,
) -> None:
    _act_as(as_principal, user)
    job.status = RunStatus.TERMINATED
    await session.commit()

    response = await client.post(f"{API}/jobs/{job.id}/cancel")

    assert response.status_code == HTTPStatus.OK
    assert response.json()["status"] == "terminated"


# Logs.


async def _seed_log(session: AsyncSession, job: JobTable, *, id: int, trial_id: str | None) -> None:
    from autotunex.db.tables import LogEntryTable

    session.add(LogEntryTable(id=id, job_id=job.id, trial_id=trial_id, level="INFO", message="x"))
    await session.commit()


async def test_get_job_logs_returns_the_keyset_envelope(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
) -> None:
    _act_as(as_principal, user)
    await _seed_log(session, job, id=1, trial_id=None)

    response = await client.get(f"{API}/jobs/{job.id}/logs")

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["logs"][0]["message"] == "x"
    assert body["has_more"] is False
    assert body["next_before_id"] is None


async def test_get_job_logs_rejects_a_limit_above_500(
    client: AsyncClient, as_principal: Callable[[Principal], None], user: UserTable, job: JobTable
) -> None:
    _act_as(as_principal, user)

    response = await client.get(f"{API}/jobs/{job.id}/logs", params={"limit": 501})

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


async def test_get_job_logs_of_another_users_job_is_404(
    client: AsyncClient, as_principal: Callable[[Principal], None], user: UserTable, job: JobTable
) -> None:
    as_principal(
        Principal(email="other@example.com", provider="session", user_id=uuid4(), is_admin=False)
    )

    response = await client.get(f"{API}/jobs/{job.id}/logs")

    assert response.status_code == HTTPStatus.NOT_FOUND


async def test_admin_reads_another_users_job_logs_with_scope_all(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
) -> None:
    # `job` is owned by `user`; act as a different admin.
    as_principal(
        Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)
    )
    await _seed_log(session, job, id=1, trial_id=None)

    response = await client.get(f"{API}/jobs/{job.id}/logs", params={"scope": "all"})

    assert response.status_code == HTTPStatus.OK
    assert response.json()["logs"][0]["message"] == "x"


async def test_non_admin_requesting_scope_all_on_logs_is_forbidden(
    client: AsyncClient, as_principal: Callable[[Principal], None], user: UserTable, job: JobTable
) -> None:
    _act_as(as_principal, user)

    response = await client.get(f"{API}/jobs/{job.id}/logs", params={"scope": "all"})

    assert response.status_code == HTTPStatus.FORBIDDEN


async def test_get_trial_logs_returns_only_that_trials_lines(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
) -> None:
    _act_as(as_principal, user)
    await _seed_log(session, job, id=1, trial_id=None)
    await _seed_log(session, job, id=2, trial_id="t1")

    response = await client.get(f"{API}/jobs/{job.id}/trials/t1/logs")

    body = response.json()
    assert [row["id"] for row in body["logs"]] == [2]


async def _seed_metric(
    session: AsyncSession,
    job: JobTable,
    *,
    id: int,
    trial_id: str | None,
    global_step: int,
    loss: float,
) -> None:
    from autotunex.db.tables import TrainingMetricTable

    session.add(
        TrainingMetricTable(
            id=id,
            job_id=job.id,
            trial_id=trial_id,
            global_step=global_step,
            loss=loss,
            split="train",
        )
    )
    await session.commit()


async def test_get_job_metrics_returns_ascending_page(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
) -> None:
    _act_as(as_principal, user)
    await _seed_metric(session, job, id=1, trial_id="t1", global_step=10, loss=3.0)
    await _seed_metric(session, job, id=2, trial_id="t1", global_step=20, loss=2.0)

    response = await client.get(f"{API}/jobs/{job.id}/metrics")

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert [p["global_step"] for p in body["metrics"]] == [10, 20]
    assert body["has_more"] is False
    assert body["next_after_id"] is None


async def test_get_trial_metrics_filters_by_trial(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
) -> None:
    _act_as(as_principal, user)
    await _seed_metric(session, job, id=1, trial_id="t1", global_step=1, loss=1.0)
    await _seed_metric(session, job, id=2, trial_id="t2", global_step=1, loss=2.0)

    response = await client.get(f"{API}/jobs/{job.id}/trials/t2/metrics")

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert [p["loss"] for p in body["metrics"]] == [2.0]


async def test_get_job_metrics_of_another_users_job_is_404(
    client: AsyncClient, as_principal: Callable[[Principal], None], user: UserTable, job: JobTable
) -> None:
    as_principal(
        Principal(email="other@example.com", provider="session", user_id=uuid4(), is_admin=False)
    )

    response = await client.get(f"{API}/jobs/{job.id}/metrics")

    assert response.status_code == HTTPStatus.NOT_FOUND


async def test_get_gb_logs_returns_a_list_of_strings(
    client: AsyncClient,
    session: AsyncSession,
    app: FastAPI,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
) -> None:
    from autotunex.api.deps import get_gb_log_reader
    from autotunex.db.tables import GbTaskTable
    from autotunex.models.status import GbTaskType

    class StubReader:
        async def fetch(self, build_id: str, *, fetch_all: bool) -> list[str]:
            return ["gb line 1", "gb line 2"]

    app.dependency_overrides[get_gb_log_reader] = lambda: StubReader()
    _act_as(as_principal, user)
    session.add(
        GbTaskTable(
            job_id=job.id, type=GbTaskType.TUNING, status=RunStatus.RUNNING, build_id=uuid4()
        )
    )
    await session.commit()

    response = await client.get(f"{API}/jobs/{job.id}/gb-logs", params={"all": "true"})

    assert response.status_code == HTTPStatus.OK
    assert response.json() == ["gb line 1", "gb line 2"]


async def test_get_gb_logs_503_when_no_build_handle(
    client: AsyncClient, as_principal: Callable[[Principal], None], user: UserTable, job: JobTable
) -> None:
    _act_as(as_principal, user)

    response = await client.get(f"{API}/jobs/{job.id}/gb-logs")

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE


async def test_get_job_by_build_id_returns_the_job(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
) -> None:
    from autotunex.db.tables import GbTaskTable
    from autotunex.models.status import GbTaskType

    _act_as(as_principal, user)
    build_id = uuid4()
    session.add(GbTaskTable(job_id=job.id, type=GbTaskType.TUNING, build_id=build_id))
    await session.commit()

    response = await client.get(f"{API}/jobs/by-build-id/{build_id}")

    assert response.status_code == HTTPStatus.OK
    assert response.json()["id"] == str(job.id)


async def test_get_job_by_build_id_omits_tasks_and_the_config_snapshot(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
) -> None:
    """The lookup returns ``JobDetail``, not ``JobRead``.

    A caller that arrived by build id already holds the identifier the tasks array
    exists to expose, and ``config_snapshot`` embeds the whole configuration as it
    ran — the heaviest field on the response. Both stay on ``GET /jobs/{id}``.
    """
    from autotunex.db.tables import GbTaskTable
    from autotunex.models.status import GbTaskType

    _act_as(as_principal, user)
    job.config_snapshot = {
        "name": "as-it-ran",
        "config_data": {"lr": 3e-5, "tune_config": {"num_samples": {"default": 12}}},
    }
    session.add(job)
    build_id = uuid4()
    session.add(GbTaskTable(job_id=job.id, type=GbTaskType.TUNING, build_id=build_id))
    await session.commit()

    response = await client.get(f"{API}/jobs/by-build-id/{build_id}")

    body = response.json()
    assert "tasks" not in body
    assert "config_snapshot" not in body
    assert "trials" not in body
    # Everything else the detail shape carries is still here, config_name included —
    # that is read *from* the snapshot, so dropping the blob must not drop the label.
    assert body["config_name"] == "as-it-ran"
    assert body["num_trials"] == 12
    assert body["model_source"] == "huggingface"
    assert body["is_stale"] is True


async def test_get_job_by_build_id_still_reports_finished_at(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
) -> None:
    """``finished_at`` survives dropping ``tasks`` — it comes back as a computed column.

    It is the one field the mapper previously derived from the loaded tasks, so it
    is the one that would silently go ``null`` if the subquery were dropped.
    """
    from autotunex.db.tables import GbTaskTable
    from autotunex.models.status import GbTaskType

    _act_as(as_principal, user)
    build_id = uuid4()
    session.add_all(
        [
            GbTaskTable(
                job_id=job.id,
                type=GbTaskType.TUNING,
                build_id=build_id,
                updated_at="2026-08-07 00:05:00",
            ),
            GbTaskTable(
                job_id=job.id,
                type=GbTaskType.RITS,
                build_id=uuid4(),
                updated_at="2026-08-07 00:09:00",
            ),
        ]
    )
    await session.commit()

    response = await client.get(f"{API}/jobs/by-build-id/{build_id}")

    # The latest across *all* the job's tasks, not just the one matched by build_id.
    assert response.json()["finished_at"] == "2026-08-07 00:09:00"


async def test_get_job_by_build_id_unknown_build_is_404(
    client: AsyncClient, as_principal: Callable[[Principal], None], user: UserTable
) -> None:
    _act_as(as_principal, user)

    response = await client.get(f"{API}/jobs/by-build-id/{uuid4()}")

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_get_job_by_build_id_of_another_users_build_is_404(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    job: JobTable,
) -> None:
    from autotunex.db.tables import GbTaskTable
    from autotunex.models.status import GbTaskType

    build_id = uuid4()
    session.add(GbTaskTable(job_id=job.id, type=GbTaskType.TUNING, build_id=build_id))
    await session.commit()
    as_principal(
        Principal(email="other@example.com", provider="session", user_id=uuid4(), is_admin=False)
    )

    response = await client.get(f"{API}/jobs/by-build-id/{build_id}")

    assert response.status_code == HTTPStatus.NOT_FOUND


async def test_get_job_by_build_id_admin_scope_all_reaches_another_owner(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    job: JobTable,
) -> None:
    from autotunex.db.tables import GbTaskTable
    from autotunex.models.status import GbTaskType

    build_id = uuid4()
    session.add(GbTaskTable(job_id=job.id, type=GbTaskType.TUNING, build_id=build_id))
    await session.commit()
    as_principal(
        Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)
    )

    response = await client.get(f"{API}/jobs/by-build-id/{build_id}", params={"scope": "all"})

    assert response.status_code == HTTPStatus.OK
    assert response.json()["id"] == str(job.id)


async def test_get_job_by_build_id_non_admin_scope_all_is_403(
    client: AsyncClient, as_principal: Callable[[Principal], None], user: UserTable
) -> None:
    _act_as(as_principal, user)

    response = await client.get(f"{API}/jobs/by-build-id/{uuid4()}", params={"scope": "all"})

    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_get_job_by_build_id_malformed_build_id_is_422(
    client: AsyncClient, as_principal: Callable[[Principal], None], user: UserTable
) -> None:
    _act_as(as_principal, user)

    response = await client.get(f"{API}/jobs/by-build-id/not-a-uuid")

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


# Result report.


async def test_result_report_lists_file_scheme_directory(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
    tmp_path: Path,
) -> None:
    from autotunex.db.tables import GbTaskTable
    from autotunex.models.status import GbTaskType

    _act_as(as_principal, user)
    (tmp_path / "best_config.json").write_text("{}")
    session.add(
        GbTaskTable(
            job_id=job.id,
            type=GbTaskType.TUNING,
            artifact_uri=f"file://{tmp_path}",
            build_status={"details": {"status": "success"}},
        )
    )
    await session.commit()

    response = await client.get(f"{API}/jobs/{job.id}/result-report")

    assert response.status_code == HTTPStatus.OK
    assert [a["filename"] for a in response.json()] == ["best_config.json"]


async def test_result_report_not_ready_is_409(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
) -> None:
    from autotunex.db.tables import GbTaskTable
    from autotunex.models.status import GbTaskType

    _act_as(as_principal, user)
    session.add(
        GbTaskTable(
            job_id=job.id,
            type=GbTaskType.TUNING,
            artifact_uri="file:///tmp/x",
            build_status={"details": {"status": "running"}},
        )
    )
    await session.commit()

    response = await client.get(f"{API}/jobs/{job.id}/result-report")

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_result_report_missing_source_is_404(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
    tmp_path: Path,
) -> None:
    from autotunex.db.tables import GbTaskTable
    from autotunex.models.status import GbTaskType

    _act_as(as_principal, user)
    missing = tmp_path / "never-created"
    session.add(
        GbTaskTable(
            job_id=job.id,
            type=GbTaskType.TUNING,
            artifact_uri=f"file://{missing}",
            build_status={"details": {"status": "success"}},
        )
    )
    await session.commit()

    response = await client.get(f"{API}/jobs/{job.id}/result-report")

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_result_report_source_unavailable_is_502(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autotunex.core.exceptions import ArtifactSourceUnavailableError
    from autotunex.db.tables import GbTaskTable
    from autotunex.models.asset import AssetSummary
    from autotunex.models.status import GbTaskType
    from autotunex.services.storage import artifacts as artifacts_mod

    _act_as(as_principal, user)
    session.add(
        GbTaskTable(
            job_id=job.id,
            type=GbTaskType.TUNING,
            artifact_uri="hf://huggingface.co/models/org/repo",
            build_status={"details": {"status": "success"}},
        )
    )
    await session.commit()

    async def _raise(self: object, *, location: str) -> list[AssetSummary]:
        raise ArtifactSourceUnavailableError

    monkeypatch.setattr(artifacts_mod.HuggingFaceArtifactLister, "list_files", _raise)

    response = await client.get(f"{API}/jobs/{job.id}/result-report")

    assert response.status_code == HTTPStatus.BAD_GATEWAY
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


# Result-report downloads (single file + archive).


def _add_tuning_task(
    session: AsyncSession, job: JobTable, *, artifact_uri: str, status: str = "success"
) -> None:
    from autotunex.db.tables import GbTaskTable
    from autotunex.models.status import GbTaskType

    session.add(
        GbTaskTable(
            job_id=job.id,
            type=GbTaskType.TUNING,
            artifact_uri=artifact_uri,
            build_status={"details": {"status": status}},
        )
    )


async def test_result_report_file_downloads_raw_file_as_attachment(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
    tmp_path: Path,
) -> None:
    _act_as(as_principal, user)
    (tmp_path / "final_config.json").write_text('{"lr": 1}')
    _add_tuning_task(session, job, artifact_uri=f"file://{tmp_path}")
    await session.commit()

    response = await client.get(
        f"{API}/jobs/{job.id}/result-report/file", params={"path": "final_config.json"}
    )

    assert response.status_code == HTTPStatus.OK
    assert response.content == b'{"lr": 1}'
    assert response.headers["content-type"].startswith("application/json")
    disposition = response.headers["content-disposition"]
    assert "attachment" in disposition
    assert "final_config.json" in disposition


async def test_result_report_file_disambiguates_same_basename_by_path(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
    tmp_path: Path,
) -> None:
    _act_as(as_principal, user)
    (tmp_path / "run-a").mkdir()
    (tmp_path / "run-b").mkdir()
    (tmp_path / "run-a" / "adapters.safetensors").write_bytes(b"AAAA")
    (tmp_path / "run-b" / "adapters.safetensors").write_bytes(b"BBBBBB")
    _add_tuning_task(session, job, artifact_uri=f"file://{tmp_path}")
    await session.commit()

    first = await client.get(
        f"{API}/jobs/{job.id}/result-report/file", params={"path": "run-a/adapters.safetensors"}
    )
    second = await client.get(
        f"{API}/jobs/{job.id}/result-report/file", params={"path": "run-b/adapters.safetensors"}
    )

    assert first.content == b"AAAA"
    assert second.content == b"BBBBBB"


async def test_result_report_file_rejects_path_traversal_with_404(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
    tmp_path: Path,
) -> None:
    _act_as(as_principal, user)
    (tmp_path / "final_config.json").write_text("{}")
    _add_tuning_task(session, job, artifact_uri=f"file://{tmp_path}")
    await session.commit()

    response = await client.get(
        f"{API}/jobs/{job.id}/result-report/file", params={"path": "../../etc/hosts"}
    )

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_result_report_file_not_ready_is_409(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
) -> None:
    _act_as(as_principal, user)
    _add_tuning_task(session, job, artifact_uri="file:///tmp/x", status="running")
    await session.commit()

    response = await client.get(
        f"{API}/jobs/{job.id}/result-report/file", params={"path": "anything"}
    )

    assert response.status_code == HTTPStatus.CONFLICT


async def test_result_report_archive_streams_a_zip_of_all_files(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
    tmp_path: Path,
) -> None:
    _act_as(as_principal, user)
    (tmp_path / "run-a").mkdir()
    (tmp_path / "final_config.json").write_text('{"a": 1}')
    (tmp_path / "run-a" / "adapters.safetensors").write_bytes(b"AAAA")
    _add_tuning_task(session, job, artifact_uri=f"file://{tmp_path}")
    await session.commit()

    response = await client.get(f"{API}/jobs/{job.id}/result-report/archive")

    assert response.status_code == HTTPStatus.OK
    assert response.headers["content-type"].startswith("application/zip")
    disposition = response.headers["content-disposition"]
    assert "attachment" in disposition
    assert "_assets.zip" in disposition
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {"final_config.json", "run-a/adapters.safetensors"}
        assert archive.read("run-a/adapters.safetensors") == b"AAAA"


async def test_result_report_archive_not_ready_is_409(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
) -> None:
    _act_as(as_principal, user)
    _add_tuning_task(session, job, artifact_uri="file:///tmp/x", status="running")
    await session.commit()

    response = await client.get(f"{API}/jobs/{job.id}/result-report/archive")

    assert response.status_code == HTTPStatus.CONFLICT
