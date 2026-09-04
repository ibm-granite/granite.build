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
from autotunex.core.auth.disabled import SYSTEM_STANDALONE_EMAIL
from autotunex.core.constants import SYSTEM_USER_ID
from autotunex.db.tables import ConfigurationTable, JobTable, UserTable
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


# --- estimate-usages: the saved-configuration (config_id) path ---
#
# Reading a saved configuration is an owner-scoped read, and this endpoint takes no
# `scope` parameter, so it is *always* own-scope plus the shared system tier. The
# repository applies its ownership predicate only when `owner_id is not None`, which
# made an unprovisioned caller's `None` mean "no filter" and handed it any owner's
# `config_data`. These tests pin every principal that reaches the lookup.


async def _seed_config(
    session: AsyncSession, *, owner_id: str, name: str = "saved"
) -> ConfigurationTable:
    """Persist one configuration owned by ``owner_id``."""
    config = ConfigurationTable(
        id=uuid4(),
        user_id=owner_id,
        name=name,
        tuner_type="sft",
        rl_tuner_type=None,
        config_data=_CONFIG_DATA,
    )
    session.add(config)
    await session.commit()
    return config


async def _estimate_with(client: AsyncClient, config_id: object) -> Any:  # noqa: ANN401
    """POST an estimate that resolves its configuration by id."""
    return await client.post(
        f"{API}/jobs/estimate-usages",
        json={"model_name": "meta-llama/Llama-2-7b", "config_id": str(config_id)},
    )


async def test_estimate_usages_reads_the_callers_own_saved_configuration(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    session: AsyncSession,
) -> None:
    _act_as(as_principal, user)
    config = await _seed_config(session, owner_id=str(user.id))

    response = await _estimate_with(client, config.id)

    assert response.status_code == 200
    assert response.json()["model_size_billion_params"] == 7.0


async def test_estimate_usages_reads_a_shared_system_configuration(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    session: AsyncSession,
) -> None:
    _act_as(as_principal, user)
    session.add(UserTable(id=SYSTEM_USER_ID, email="system@autotunex.local", role="user"))
    await session.commit()
    config = await _seed_config(session, owner_id=str(SYSTEM_USER_ID), name="starter")

    response = await _estimate_with(client, config.id)

    assert response.status_code == 200


async def test_estimate_usages_hides_another_owners_saved_configuration(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    session: AsyncSession,
) -> None:
    other = UserTable(id=uuid4(), email="other@example.com", role="user")
    session.add(other)
    await session.commit()
    _act_as(as_principal, user)
    config = await _seed_config(session, owner_id=str(other.id))

    response = await _estimate_with(client, config.id)

    assert response.status_code == 404


async def test_estimate_usages_hides_a_saved_configuration_from_an_unprovisioned_caller(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    session: AsyncSession,
) -> None:
    # An authenticated caller whose verified email has no `users` row (a real
    # provider with `auto_provision_users` off) carries `user_id=None`. That reached
    # the repository's deliberate unscoped branch and read the row outright.
    owner = UserTable(id=uuid4(), email="owner@example.com", role="user")
    session.add(owner)
    await session.commit()
    config = await _seed_config(session, owner_id=str(owner.id))
    as_principal(
        Principal(email="ghost@example.com", provider="session", user_id=None, is_admin=False)
    )

    response = await _estimate_with(client, config.id)

    assert response.status_code == 404
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


# Standalone mode must keep working: its principal is always provisioned (see
# `api.deps.get_principal`, which provisions the standalone owner regardless of
# `auto_provision_users`), so it has a real `user_id` and the guard never fires for
# it. These two drive the *real* authenticator — no `as_principal` override — so a
# regression that refused standalone callers would fail here.


async def test_estimate_usages_in_standalone_mode_reads_its_own_configuration(
    client: AsyncClient, session: AsyncSession
) -> None:
    # Provision the standalone owner exactly as a first request would, then give it a
    # configuration to read back.
    owner = UserTable(id=uuid4(), email=SYSTEM_STANDALONE_EMAIL, role="user")
    session.add(owner)
    await session.commit()
    config = await _seed_config(session, owner_id=str(owner.id))

    response = await _estimate_with(client, config.id)

    assert response.status_code == 200
    assert response.json()["model_size_billion_params"] == 7.0


async def test_estimate_usages_in_standalone_mode_reads_a_shared_system_configuration(
    client: AsyncClient, session: AsyncSession
) -> None:
    session.add(UserTable(id=SYSTEM_USER_ID, email="system@autotunex.local", role="user"))
    await session.commit()
    config = await _seed_config(session, owner_id=str(SYSTEM_USER_ID), name="starter")

    response = await _estimate_with(client, config.id)

    assert response.status_code == 200


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
