"""Tests for job API schemas."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from autotunex.models.job import ONLINE_RL_TUNER_TYPES, TERMINAL_JOB_STATUSES, JobCreate
from autotunex.models.status import RunStatus


def test_job_create_applies_defaults() -> None:
    job = JobCreate(
        config_id=uuid4(), dataset_id=uuid4(), model="ibm/granite", experiment_name="exp"
    )

    assert job.model_source == "huggingface"
    assert job.seed == 42
    assert job.autotune is True
    assert job.reward_function_code is None


def test_job_create_rejects_dmf_model_source() -> None:
    with pytest.raises(ValidationError):
        JobCreate(
            config_id=uuid4(),
            dataset_id=uuid4(),
            model="ibm/granite",
            experiment_name="exp",
            model_source="dmf",
        )


def test_job_create_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        JobCreate(  # type: ignore[call-arg]
            config_id=uuid4(),
            dataset_id=uuid4(),
            model="ibm/granite",
            experiment_name="exp",
            additional_info={"x": 1},
        )


def test_job_create_rejects_blank_model() -> None:
    with pytest.raises(ValidationError):
        JobCreate(config_id=uuid4(), dataset_id=uuid4(), model="   ", experiment_name="exp")


def test_online_rl_tuner_types_are_the_reward_requiring_set() -> None:
    assert frozenset({"ppo", "grpo", "dapo"}) == ONLINE_RL_TUNER_TYPES


def test_terminal_job_statuses_are_the_three_states_with_no_transitions() -> None:
    assert (
        frozenset({RunStatus.COMPLETED, RunStatus.ERROR, RunStatus.TERMINATED})
        == TERMINAL_JOB_STATUSES
    )
