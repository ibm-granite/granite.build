# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Tests for resource estimation."""

from __future__ import annotations

from uuid import uuid4

import pytest

from autotunex.core.exceptions import ConfigurationNotFoundError, DomainValidationError
from autotunex.models.auth import Principal
from autotunex.models.estimation import EstimateUsagesRequest
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

    async def get(self, config_id: object, *, owner_id: object) -> object | None:
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
