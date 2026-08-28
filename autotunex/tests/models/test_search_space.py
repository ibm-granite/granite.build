"""Tests for search space validation.

Search space invariants are the schema's job, not the service's — an invalid
space must never reach the database.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from autotunex.models.search_space import (
    CategoricalParameter,
    FloatParameter,
    IntParameter,
    SearchSpace,
)


def test_parses_each_parameter_kind_by_discriminator() -> None:
    space = SearchSpace.model_validate(
        {
            "learning_rate": {"kind": "float", "low": 1e-6, "high": 1e-3, "log": True},
            "lora_rank": {"kind": "int", "low": 4, "high": 64, "step": 4},
            "scheduler": {"kind": "categorical", "choices": ["linear", "cosine"]},
        }
    )

    assert isinstance(space.root["learning_rate"], FloatParameter)
    assert isinstance(space.root["lora_rank"], IntParameter)
    assert isinstance(space.root["scheduler"], CategoricalParameter)
    assert space.names() == ["learning_rate", "lora_rank", "scheduler"]


def test_rejects_an_empty_space() -> None:
    with pytest.raises(ValidationError, match="at least one parameter"):
        SearchSpace.model_validate({})


def test_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        SearchSpace.model_validate({"lr": {"kind": "gaussian", "mean": 0.0}})


@pytest.mark.parametrize("kind", ["float", "int"])
def test_rejects_bounds_that_are_not_increasing(kind: str) -> None:
    with pytest.raises(ValidationError, match="strictly less than"):
        SearchSpace.model_validate({"p": {"kind": kind, "low": 8, "high": 8}})


@pytest.mark.parametrize("kind", ["float", "int"])
def test_rejects_log_scale_with_non_positive_lower_bound(kind: str) -> None:
    with pytest.raises(ValidationError, match="log-scale"):
        SearchSpace.model_validate({"p": {"kind": kind, "low": 0, "high": 8, "log": True}})


def test_rejects_categorical_without_choices() -> None:
    with pytest.raises(ValidationError):
        SearchSpace.model_validate({"p": {"kind": "categorical", "choices": []}})


def test_rejects_duplicate_categorical_choices() -> None:
    with pytest.raises(ValidationError, match="unique"):
        SearchSpace.model_validate({"p": {"kind": "categorical", "choices": ["a", "a"]}})


def test_rejects_a_negative_int_step() -> None:
    with pytest.raises(ValidationError):
        SearchSpace.model_validate({"p": {"kind": "int", "low": 1, "high": 8, "step": 0}})


def test_serializes_back_to_plain_json() -> None:
    payload = {"lr": {"kind": "float", "low": 0.1, "high": 1.0, "log": False}}

    space = SearchSpace.model_validate(payload)

    assert space.model_dump(mode="json") == payload
