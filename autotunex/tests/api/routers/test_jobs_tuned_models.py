"""``GET /jobs/tuned-models`` over HTTP: shape, route precedence, scope and bounds.

Also covers ``POST /jobs/estimate-usages`` resolving a tuned model to its base model,
which needs this module's allowlisted namespace.
"""

from __future__ import annotations

from collections.abc import Callable
from http import HTTPStatus
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.core.config import Settings
from autotunex.db.tables import (
    ConfigurationTable,
    DatasetTable,
    GbTaskTable,
    JobTable,
    UserTable,
)
from autotunex.models.auth import Principal
from autotunex.models.status import GbTaskType, RunStatus
from tests.conftest import API, make_settings


@pytest.fixture
def settings() -> Settings:
    """Test settings with one allowlisted HF namespace, so outputs are eligible."""
    return make_settings().model_copy(update={"hf_import_namespaces": ["example-org"]})


async def test_tuned_models_lists_the_callers_completed_sft_output(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    dataset: DatasetTable,
) -> None:
    job = JobTable(
        id=uuid4(),
        user_id=str(user.id),
        status=RunStatus.COMPLETED,
        config_id=configuration.id,
        dataset_id=dataset.id,
        model="ibm-granite/granite-4.0-h-micro",
        model_source="huggingface",
        experiment_name="sft-granite",
        tuning_type="sft",
    )
    session.add(job)
    session.add(
        GbTaskTable(
            job_id=job.id,
            type=GbTaskType.TUNING,
            status=RunStatus.COMPLETED,
            artifact_uri="hf://huggingface.co/models/example-org/autotunex_aaaa0001",
            updated_at="2026-09-20T10:00:00Z",
        )
    )
    await session.commit()
    as_principal(Principal(email=user.email, provider="session", user_id=user.id, is_admin=False))

    response = await client.get(f"{API}/jobs/tuned-models")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {
        "items": [
            {
                "job_id": str(job.id),
                "repo_id": "example-org/autotunex_aaaa0001",
                "model_source": "huggingface",
                "experiment_name": "sft-granite",
                "base_model": "ibm-granite/granite-4.0-h-micro",
                "tuning_type": "sft",
                "rl_tuner_type": None,
                "finished_at": "2026-09-20T10:00:00Z",
                "user": "tester@example.com",
            }
        ],
        "total": 1,
        "limit": 20,
        "offset": 0,
    }


async def test_tuned_models_refuses_scope_all_to_a_non_admin(
    client: AsyncClient, as_principal: Callable[[Principal], None], user: UserTable
) -> None:
    as_principal(Principal(email=user.email, provider="session", user_id=user.id, is_admin=False))

    response = await client.get(f"{API}/jobs/tuned-models", params={"scope": "all"})

    assert response.status_code == HTTPStatus.FORBIDDEN


async def test_tuned_models_rejects_a_limit_above_one_hundred(
    client: AsyncClient, as_principal: Callable[[Principal], None], user: UserTable
) -> None:
    as_principal(Principal(email=user.email, provider="session", user_id=user.id, is_admin=False))

    response = await client.get(f"{API}/jobs/tuned-models", params={"limit": 101})

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


async def test_estimate_usages_sizes_a_tuned_model_by_its_base_model(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    dataset: DatasetTable,
) -> None:
    job = JobTable(
        id=uuid4(),
        user_id=str(user.id),
        status=RunStatus.COMPLETED,
        config_id=configuration.id,
        dataset_id=dataset.id,
        model="meta-llama/Llama-3.1-8B",
        model_source="huggingface",
        experiment_name="sft-llama",
        tuning_type="sft",
    )
    session.add(job)
    session.add(
        GbTaskTable(
            job_id=job.id,
            type=GbTaskType.TUNING,
            status=RunStatus.COMPLETED,
            artifact_uri="hf://huggingface.co/models/example-org/autotunex_b482441b",
            updated_at="2026-09-20T10:00:00Z",
        )
    )
    await session.commit()
    as_principal(Principal(email=user.email, provider="session", user_id=user.id, is_admin=False))

    response = await client.post(
        f"{API}/jobs/estimate-usages",
        json={
            "model_name": "example-org/autotunex_b482441b",
            "config_data": {},
            "tuner_type": "sft",
        },
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json()["model_size_billion_params"] == 8.0


async def test_estimate_usages_refuses_scope_all_to_a_non_admin(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
) -> None:
    as_principal(Principal(email=user.email, provider="session", user_id=user.id, is_admin=False))

    response = await client.post(
        f"{API}/jobs/estimate-usages?scope=all",
        json={"model_name": "x-7b", "config_data": {}, "tuner_type": "sft"},
    )

    assert response.status_code == HTTPStatus.FORBIDDEN
