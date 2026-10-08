# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Tests for resource estimation."""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.core.constants import SYSTEM_USER_ID
from autotunex.core.exceptions import (
    ConfigurationNotFoundError,
    DomainValidationError,
    ScopeNotPermittedError,
)
from autotunex.db.repositories.sqlalchemy import (
    SqlAlchemyConfigurationRepository,
    SqlAlchemyJobRepository,
)
from autotunex.db.tables import (
    ConfigurationTable,
    DatasetTable,
    GbTaskTable,
    JobTable,
    UserTable,
)
from autotunex.models.auth import Principal
from autotunex.models.common import DataScope
from autotunex.models.estimation import EstimateUsagesRequest
from autotunex.models.status import GbTaskType, RunStatus
from autotunex.services.estimation import (
    EstimationService,
    estimate_memory_usage,
    parse_model_parameters,
)

_CONFIG_DATA = {
    "training_config": {"precision": {"default": "bf16"}, "max_length": {"default": 512}},
    "tuners_config": {
        "sft": {"hyperparams": {"per_device_train_batch_size": {"values": [1, 2, 4]}}}
    },
}


def _principal() -> Principal:
    return Principal(email="u@example.com", provider="test", user_id=uuid4(), is_admin=False)


def test_parse_model_parameters_reads_billions() -> None:
    assert parse_model_parameters("meta-llama/Llama-2-7b") == 7.0
    assert parse_model_parameters("some-500m-model") == 0.5
    assert parse_model_parameters("no-size-here") is None


def test_parse_model_parameters_reads_common_hf_names() -> None:
    assert parse_model_parameters("Qwen/Qwen2.5-0.5B-Instruct") == 0.5
    assert parse_model_parameters("HuggingFaceTB/SmolLM2-135M-Instruct") == 0.135
    assert parse_model_parameters("mistralai/Mixtral-8x7B-v0.1") == 7.0


def test_parse_model_parameters_reads_a_size_glued_to_other_characters() -> None:
    assert parse_model_parameters("google/gemma-3n-E2B-it") == 2.0
    assert parse_model_parameters("Qwen/Qwen1.5-MoE-A2.7B") == 2.7
    assert parse_model_parameters("TheBloke/Mixtral-8X7B-v0.1-GPTQ") == 7.0
    assert parse_model_parameters("bigscience/bloom-7b1") == 7.0
    assert parse_model_parameters("llama7b") == 7.0


def test_parse_model_parameters_finds_no_size_in_a_tuned_model_hash() -> None:
    assert parse_model_parameters("example-org/autotunex_b482441b") is None
    assert parse_model_parameters("example-org/autotunex_1234567b") is None
    assert parse_model_parameters("example-org/autotunex_12b4f00d") is None


def test_parse_model_parameters_falls_back_to_granite_4_lookup() -> None:
    assert parse_model_parameters("ibm-granite/granite-4.0-h-tiny") == 7.0
    assert parse_model_parameters("ibm-granite/granite-4.0-h-small") == 32.0


def test_estimate_memory_usage_is_positive() -> None:
    result = estimate_memory_usage(
        model_size_billion_params=7.0,
        precision="bf16",
        batch_size=4,
        sequence_length=512,
        gpu_size_gb=80,
    )

    assert result["gpu_memory_gb"] > 0
    assert result["num_gpus"] >= 1


def test_estimate_memory_usage_rejects_unsupported_precision() -> None:
    with pytest.raises(ValueError, match="Unsupported precision"):
        estimate_memory_usage(model_size_billion_params=7.0, precision="fp99")


async def test_inline_config_needs_no_db() -> None:
    service = EstimationService(configuration_repository=None, principal=_principal())

    response = await service.estimate(
        EstimateUsagesRequest(
            model_name="meta-llama/Llama-2-7b", config_data=_CONFIG_DATA, tuner_type="sft"
        )
    )

    assert response.model_size_billion_params == 7.0
    assert response.num_gpus >= 1


async def test_unparseable_model_name_is_422() -> None:
    service = EstimationService(configuration_repository=None, principal=_principal())

    with pytest.raises(DomainValidationError):
        await service.estimate(
            EstimateUsagesRequest(model_name="mystery", config_data=_CONFIG_DATA, tuner_type="sft")
        )


async def test_empty_batch_values_does_not_500() -> None:
    bad = {
        "training_config": {"max_length": {"default": 256}},
        "tuners_config": {"sft": {"hyperparams": {"per_device_train_batch_size": {"values": []}}}},
    }
    service = EstimationService(configuration_repository=None, principal=_principal())

    response = await service.estimate(
        EstimateUsagesRequest(model_name="x-7b", config_data=bad, tuner_type="sft")
    )

    assert response.num_gpus >= 1  # falls back to a default batch size, no IndexError


async def test_unsupported_precision_in_config_falls_back_not_500() -> None:
    bad = {
        "training_config": {"precision": {"default": "float8"}, "max_length": {"default": 512}},
        "tuners_config": {"sft": {"hyperparams": {"per_device_train_batch_size": {"values": [4]}}}},
    }
    service = EstimationService(configuration_repository=None, principal=_principal())

    response = await service.estimate(
        EstimateUsagesRequest(model_name="x-7b", config_data=bad, tuner_type="sft")
    )

    assert response.gpu_memory_gb > 0  # unsupported dtype degrades to the default, not a 500


async def test_online_rl_tuner_type_adds_extra_memory() -> None:
    rl_config_data = {
        "training_config": {"precision": {"default": "bf16"}, "max_length": {"default": 512}},
        "tuners_rl_config": {
            "ppo": {"hyperparams": {"per_device_train_batch_size": {"values": [1, 2]}}}
        },
    }
    service = EstimationService(configuration_repository=None, principal=_principal())

    baseline = await service.estimate(
        EstimateUsagesRequest(
            model_name="meta-llama/Llama-2-7b", config_data=_CONFIG_DATA, tuner_type="sft"
        )
    )
    rl_response = await service.estimate(
        EstimateUsagesRequest(
            model_name="meta-llama/Llama-2-7b", config_data=rl_config_data, rl_tuner_type="ppo"
        )
    )

    assert rl_response.gpu_memory_gb > baseline.gpu_memory_gb


async def test_offline_rl_tuner_type_does_not_add_extra_memory() -> None:
    dpo_config_data = {
        "training_config": {"precision": {"default": "bf16"}, "max_length": {"default": 512}},
        "tuners_rl_config": {
            "dpo": {"hyperparams": {"per_device_train_batch_size": {"values": [1, 2, 4]}}}
        },
    }
    service = EstimationService(configuration_repository=None, principal=_principal())

    baseline = await service.estimate(
        EstimateUsagesRequest(
            model_name="meta-llama/Llama-2-7b", config_data=_CONFIG_DATA, tuner_type="sft"
        )
    )
    dpo_response = await service.estimate(
        EstimateUsagesRequest(
            model_name="meta-llama/Llama-2-7b", config_data=dpo_config_data, rl_tuner_type="dpo"
        )
    )

    assert dpo_response.gpu_memory_gb == pytest.approx(baseline.gpu_memory_gb)


class _FakeConfigRepo:
    def __init__(self, config: object | None) -> None:
        self._config = config

    async def get(
        self, config_id: object, *, owner_id: object, include_shared: bool = False
    ) -> object | None:
        return self._config


async def test_saved_config_not_found_is_404() -> None:
    service = EstimationService(
        configuration_repository=_FakeConfigRepo(None),  # type: ignore[arg-type]
        principal=_principal(),
    )

    with pytest.raises(ConfigurationNotFoundError):
        await service.estimate(EstimateUsagesRequest(model_name="x-7b", config_id=uuid4()))


class _FakeConfig:
    def __init__(self, config_data: dict[str, object], tuner_type: str | None) -> None:
        self.config_data = config_data
        self.tuner_type = tuner_type
        self.rl_tuner_type: str | None = None


async def test_saved_config_is_used_when_config_id_is_given() -> None:
    fake_config = _FakeConfig(_CONFIG_DATA, tuner_type="sft")
    service = EstimationService(
        configuration_repository=_FakeConfigRepo(fake_config),  # type: ignore[arg-type]
        principal=_principal(),
    )

    response = await service.estimate(
        EstimateUsagesRequest(model_name="meta-llama/Llama-2-7b", config_id=uuid4())
    )

    assert response.model_size_billion_params == 7.0
    assert response.num_gpus >= 1


async def test_estimate_resolves_a_system_owned_saved_config(session: AsyncSession) -> None:
    session.add(UserTable(id=SYSTEM_USER_ID, email="system@autotunex.local", role="user"))
    await session.commit()
    system_config = ConfigurationTable(
        id=uuid4(),
        user_id=str(SYSTEM_USER_ID),
        name="starter",
        tuner_type="sft",
        rl_tuner_type=None,
        config_data=_CONFIG_DATA,
    )
    session.add(system_config)
    await session.commit()
    repository = SqlAlchemyConfigurationRepository(session)
    service = EstimationService(configuration_repository=repository, principal=_principal())

    response = await service.estimate(
        EstimateUsagesRequest(model_name="x-7b", config_id=system_config.id)
    )

    assert response is not None


_NAMESPACES = ["example-org"]


async def _tuned_job(
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    dataset: DatasetTable,
    *,
    model: str,
    repo_id: str,
) -> None:
    """Persist a completed SFT job whose TUNING task published ``repo_id`` to HF."""
    job = JobTable(
        id=uuid4(),
        user_id=str(user.id),
        status=RunStatus.COMPLETED,
        config_id=configuration.id,
        dataset_id=dataset.id,
        model=model,
        model_source="huggingface",
        experiment_name="sft-run",
        tuning_type="sft",
        config_snapshot={"name": "c", "tuner_type": "sft", "rl_tuner_type": None},
    )
    session.add(job)
    session.add(
        GbTaskTable(
            job_id=job.id,
            type=GbTaskType.TUNING,
            status=RunStatus.COMPLETED,
            artifact_uri=f"hf://huggingface.co/models/{repo_id}",
            updated_at="2026-09-20T10:00:00Z",
        )
    )
    await session.commit()


def _owner(user: UserTable) -> Principal:
    return Principal(email=user.email, provider="test", user_id=user.id, is_admin=False)


async def test_a_tuned_model_is_estimated_at_its_base_model_size(
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    dataset: DatasetTable,
) -> None:
    # The job-id hash "b482441b" reads as 482441B to the name regex.
    await _tuned_job(
        session,
        user,
        configuration,
        dataset,
        model="meta-llama/Llama-3.1-8B",
        repo_id="example-org/autotunex_b482441b",
    )
    service = EstimationService(
        configuration_repository=None,
        principal=_owner(user),
        tuned_model_repository=SqlAlchemyJobRepository(session),
        namespaces=_NAMESPACES,
    )

    response = await service.estimate(
        EstimateUsagesRequest(model_name="example-org/autotunex_b482441b", config_data=_CONFIG_DATA)
    )

    assert response.model_size_billion_params == 8.0


async def test_a_tuned_model_built_on_a_tuned_model_resolves_to_the_original_base(
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    dataset: DatasetTable,
) -> None:
    await _tuned_job(
        session,
        user,
        configuration,
        dataset,
        model="meta-llama/Llama-3.1-8B",
        repo_id="example-org/autotunex_aaaa0001",
    )
    await _tuned_job(
        session,
        user,
        configuration,
        dataset,
        model="example-org/autotunex_aaaa0001",
        repo_id="example-org/autotunex_3b000002",
    )
    service = EstimationService(
        configuration_repository=None,
        principal=_owner(user),
        tuned_model_repository=SqlAlchemyJobRepository(session),
        namespaces=_NAMESPACES,
    )

    response = await service.estimate(
        EstimateUsagesRequest(model_name="example-org/autotunex_3b000002", config_data=_CONFIG_DATA)
    )

    assert response.model_size_billion_params == 8.0


async def test_another_owners_tuned_model_is_not_resolved(
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    dataset: DatasetTable,
) -> None:
    await _tuned_job(
        session,
        user,
        configuration,
        dataset,
        model="meta-llama/Llama-3.1-8B",
        repo_id="example-org/autotunex_aaaa0001",
    )
    service = EstimationService(
        configuration_repository=None,
        principal=_principal(),
        tuned_model_repository=SqlAlchemyJobRepository(session),
        namespaces=_NAMESPACES,
    )

    with pytest.raises(DomainValidationError, match="tuned model"):
        await service.estimate(
            EstimateUsagesRequest(
                model_name="example-org/autotunex_aaaa0001", config_data=_CONFIG_DATA
            )
        )


async def test_a_tuned_model_outside_the_allowlist_by_case_is_not_resolved(
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    dataset: DatasetTable,
) -> None:
    # SQLite's LIKE folds case, so the repository matches this row; the launch
    # path's case-sensitive owner rule would not load it, so neither may this.
    await _tuned_job(
        session,
        user,
        configuration,
        dataset,
        model="meta-llama/Llama-3.1-8B",
        repo_id="example-org/autotunex_aaaa0001",
    )
    service = EstimationService(
        configuration_repository=None,
        principal=_owner(user),
        tuned_model_repository=SqlAlchemyJobRepository(session),
        namespaces=["Example-Org"],
    )

    with pytest.raises(DomainValidationError, match="tuned model"):
        await service.estimate(
            EstimateUsagesRequest(
                model_name="example-org/autotunex_aaaa0001", config_data=_CONFIG_DATA
            )
        )


async def test_an_unparseable_base_model_is_named_in_the_error(
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    dataset: DatasetTable,
) -> None:
    await _tuned_job(
        session,
        user,
        configuration,
        dataset,
        model="microsoft/phi-2",
        repo_id="example-org/autotunex_aaaa0001",
    )
    service = EstimationService(
        configuration_repository=None,
        principal=_owner(user),
        tuned_model_repository=SqlAlchemyJobRepository(session),
        namespaces=_NAMESPACES,
    )

    with pytest.raises(DomainValidationError, match="microsoft/phi-2"):
        await service.estimate(
            EstimateUsagesRequest(
                model_name="example-org/autotunex_aaaa0001", config_data=_CONFIG_DATA
            )
        )


class _CountingTunedModelRepo:
    def __init__(self) -> None:
        self.calls = 0

    async def tuned_models(self, **_: object) -> tuple[list[object], int]:
        self.calls += 1
        return [], 0


async def test_a_name_outside_the_allowlist_is_not_looked_up_as_a_tuned_model() -> None:
    repository = _CountingTunedModelRepo()
    service = EstimationService(
        configuration_repository=None,
        principal=_principal(),
        tuned_model_repository=repository,  # type: ignore[arg-type]
        namespaces=_NAMESPACES,
    )

    await service.estimate(
        EstimateUsagesRequest(model_name="meta-llama/Llama-3.1-8B", config_data=_CONFIG_DATA)
    )

    assert repository.calls == 0


async def test_an_admin_with_scope_all_resolves_another_owners_tuned_model(
    session: AsyncSession,
    user: UserTable,
    configuration: ConfigurationTable,
    dataset: DatasetTable,
) -> None:
    await _tuned_job(
        session,
        user,
        configuration,
        dataset,
        model="meta-llama/Llama-3.1-8B",
        repo_id="example-org/autotunex_aaaa0001",
    )
    admin = Principal(email="a@example.com", provider="test", user_id=uuid4(), is_admin=True)
    service = EstimationService(
        configuration_repository=None,
        principal=admin,
        tuned_model_repository=SqlAlchemyJobRepository(session),
        namespaces=_NAMESPACES,
    )

    response = await service.estimate(
        EstimateUsagesRequest(
            model_name="example-org/autotunex_aaaa0001", config_data=_CONFIG_DATA
        ),
        scope=DataScope.ALL,
    )

    assert response.model_size_billion_params == 8.0


async def test_a_non_admin_requesting_scope_all_is_refused() -> None:
    service = EstimationService(configuration_repository=None, principal=_principal())

    with pytest.raises(ScopeNotPermittedError):
        await service.estimate(
            EstimateUsagesRequest(model_name="x-7b", config_data=_CONFIG_DATA),
            scope=DataScope.ALL,
        )
