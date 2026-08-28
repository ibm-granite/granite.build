# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Search space schemas.

A *search space* declares which hyperparameters a job may vary and the range
each one is drawn from. It is a mapping of parameter name to a distribution,
discriminated on the ``kind`` field:

```json
{
  "learning_rate": {"kind": "float", "low": 1e-6, "high": 1e-3, "log": true},
  "lora_rank":     {"kind": "int", "low": 4, "high": 64, "step": 4},
  "scheduler":     {"kind": "categorical", "choices": ["linear", "cosine"]}
}
```

A *trial* is one concrete assignment of values drawn from this space.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, RootModel, model_validator

ChoiceValue = str | int | float | bool


class FloatParameter(BaseModel):
    """A continuous parameter sampled from ``[low, high]``."""

    kind: Literal["float"] = "float"
    low: float
    high: float
    log: bool = False
    """Sample on a logarithmic scale. Requires ``low > 0``."""

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.low >= self.high:
            raise ValueError("low must be strictly less than high")
        if self.log and self.low <= 0:
            raise ValueError("log-scale parameters require low > 0")
        return self


class IntParameter(BaseModel):
    """A discrete parameter sampled from ``[low, high]`` in steps of ``step``."""

    kind: Literal["int"] = "int"
    low: int
    high: int
    step: int = Field(default=1, ge=1)
    log: bool = False

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.low >= self.high:
            raise ValueError("low must be strictly less than high")
        if self.log and self.low <= 0:
            raise ValueError("log-scale parameters require low > 0")
        return self


class CategoricalParameter(BaseModel):
    """A parameter drawn from an explicit list of choices."""

    kind: Literal["categorical"] = "categorical"
    choices: list[ChoiceValue] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_unique(self) -> Self:
        seen = [repr(choice) for choice in self.choices]
        if len(set(seen)) != len(seen):
            raise ValueError("choices must be unique")
        return self


Parameter = Annotated[
    FloatParameter | IntParameter | CategoricalParameter,
    Field(discriminator="kind"),
]


class SearchSpace(RootModel[dict[str, Parameter]]):
    """A non-empty mapping of hyperparameter name to its distribution."""

    root: dict[str, Parameter]

    @model_validator(mode="after")
    def _check_non_empty(self) -> Self:
        if not self.root:
            raise ValueError("search_space must declare at least one parameter")
        return self

    def names(self) -> list[str]:
        """Return the declared parameter names."""
        return list(self.root)
