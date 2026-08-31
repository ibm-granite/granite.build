# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""``estimate-usages``, ``generate-test-solutions`` and ``result-report``, over HTTP.

Kept in a separate module from ``test_jobs.py`` so that file stays focused on
the job CRUD/read path.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any
from uuid import uuid4

from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.api.deps import get_reward_tools_service
from autotunex.db.tables import JobTable, UserTable
from autotunex.models.auth import Principal
from autotunex.services.llm.base import ChatDelta
from autotunex.services.reward.tools import RewardToolsService
from tests.conftest import API

PROBLEM_JSON = "application/problem+json"

_CONFIG_DATA = {
    "training_config": {"precision": {"default": "bf16"}, "max_length": {"default": 512}},
    "tuners_config": {
        "sft": {"hyperparams": {"per_device_train_batch_size": {"values": [1, 2, 4]}}}
    },
}


def _act_as(as_principal: Callable[[Principal], None], user: UserTable) -> None:
    """Resolve every request to ``user`` — a provisioned, non-admin owner."""
    as_principal(Principal(email=user.email, provider="session", user_id=user.id, is_admin=False))


# --- estimate-usages ---


async def test_estimate_usages_with_inline_config_returns_all_eight_fields(
    client: AsyncClient,
) -> None:
    response = await client.post(
        f"{API}/jobs/estimate-usages",
        json={
            "model_name": "meta-llama/Llama-2-7b",
            "config_data": _CONFIG_DATA,
            "tuner_type": "sft",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "model_size_billion_params",
        "gpu_memory_gb",
        "cpu_memory_gb",
        "num_gpus",
        "weights_memory",
        "optimizer_memory",
        "gradients_memory",
        "activations_memory",
    }
    assert body["model_size_billion_params"] == 7.0
    assert body["num_gpus"] >= 1


async def test_estimate_usages_rejects_neither_config_source(client: AsyncClient) -> None:
    response = await client.post(
        f"{API}/jobs/estimate-usages", json={"model_name": "meta-llama/Llama-2-7b"}
    )

    assert response.status_code == 422


async def test_estimate_usages_rejects_an_unparseable_model_name(client: AsyncClient) -> None:
    response = await client.post(
        f"{API}/jobs/estimate-usages",
        json={"model_name": "mystery", "config_data": _CONFIG_DATA, "tuner_type": "sft"},
    )

    assert response.status_code == 422
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


# --- generate-test-solutions ---


class _FakeLlmClient:
    """Returns a canned completion; never actually calls out."""

    async def complete(
        self, *, system: str, user: str, response_schema: dict[str, Any] | None = None
    ) -> str:
        return f"answer to: {user}"

    def stream_chat(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[ChatDelta]:  # pragma: no cover - unused by this service
        raise NotImplementedError


async def test_generate_test_solutions_is_503_when_no_llm_is_configured(
    client: AsyncClient,
) -> None:
    response = await client.post(
        f"{API}/jobs/generate-test-solutions",
        json={"prompts": [[{"role": "user", "content": "q1"}]]},
    )

    assert response.status_code == 503
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_generate_test_solutions_returns_one_solution_per_prompt(
    app: FastAPI, client: AsyncClient
) -> None:
    app.dependency_overrides[get_reward_tools_service] = lambda: RewardToolsService(
        llm=_FakeLlmClient()
    )

    response = await client.post(
        f"{API}/jobs/generate-test-solutions",
        json={
            "prompts": [
                [{"role": "user", "content": "q1"}],
                [{"role": "user", "content": "q2"}],
            ]
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["solutions"]) == 2
    assert all(s for s in body["solutions"])


# --- result-report ---


async def test_result_report_returns_the_jobs_output_assets(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
) -> None:
    _act_as(as_principal, user)
    job.output_artifacts = {
        "files": [
            {"filename": "results.csv", "size": 128},
            {"filename": "best_config.json", "file_size": 64},
        ]
    }
    session.add(job)
    await session.commit()

    response = await client.get(f"{API}/jobs/{job.id}/result-report")

    assert response.status_code == 200
    body = response.json()
    assert {a["filename"] for a in body} == {"results.csv", "best_config.json"}


async def test_result_report_of_an_unknown_job_is_404(
    client: AsyncClient, as_principal: Callable[[Principal], None], user: UserTable
) -> None:
    _act_as(as_principal, user)

    response = await client.get(f"{API}/jobs/{uuid4()}/result-report")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_result_report_of_another_users_job_is_404(
    client: AsyncClient, as_principal: Callable[[Principal], None], job: JobTable
) -> None:
    as_principal(
        Principal(email="other@example.com", provider="session", user_id=uuid4(), is_admin=False)
    )

    response = await client.get(f"{API}/jobs/{job.id}/result-report")

    assert response.status_code == 404
