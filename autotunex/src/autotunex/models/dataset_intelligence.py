# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Dataset-intelligence request/response schemas (Phase 2).

Request models forbid extra fields; response models tolerate extra keys because
they are populated from LLM output and validated by
:class:`~autotunex.services.dataset_intelligence.DatasetIntelligenceService`.
``ColumnMappingSuggestion.column_mapping`` is flat ``{target: source}`` — the
exact shape Phase 1's ``POST /datasets/{id}/upload`` accepts — so a client pipes
a suggestion straight into an upload; per-column confidence lives in the
separate ``column_confidence`` sidecar.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ParsingStrategy(BaseModel):
    """How to turn raw records into ``{input, output}`` training pairs.

    Also reused as an *input* embedded in :class:`ValidateStrategyRequest`.
    Tolerates extra keys from the model; the service validates it.
    """

    type: Literal["direct_mapping", "regex", "transformation"]
    description: str = ""
    input_field: str | None = None
    output_field: str | None = None
    input_pattern: str | None = None
    output_pattern: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    sample_extraction: list[dict[str, Any]] | None = None


class ColumnMappingSuggestion(BaseModel):
    """A suggested mapping of a dataset's columns onto a training format.

    ``column_mapping`` is flat ``{target: source}`` and Phase-1-upload-shaped.
    """

    dataset_format: str
    tuning_type: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    column_mapping: dict[str, str]
    column_confidence: dict[str, float] = Field(default_factory=dict)
    reasoning: str = ""

    @field_validator("column_mapping", "column_confidence", mode="before")
    @classmethod
    def _drop_null_entries(cls, value: Any) -> Any:  # noqa: ANN401
        """Drop keys the model left ``null`` before the flat-dict contract applies.

        The LLM maps a target column with no matching source to ``null`` (and may
        report a ``null`` confidence for it). A target with no source simply is not
        in the flat ``{target: source}`` mapping, so those entries are dropped
        rather than failing the ``dict[str, str]`` / ``dict[str, float]`` validation.
        """
        if isinstance(value, dict):
            return {key: item for key, item in value.items() if item is not None}
        return value


class ValidationResult(BaseModel):
    """The outcome of dry-running a parsing strategy against a sample."""

    success: bool
    parsed_count: int = Field(default=0, ge=0)
    sample_results: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ParseStrategyRequest(BaseModel):
    """Body for ``POST /datasets/intelligence/parse-strategy``."""

    model_config = ConfigDict(extra="forbid")

    sample: list[dict[str, Any]] | str
    data_format: str = "jsonl"
    custom_prompt: str | None = None


class SuggestMappingRequest(BaseModel):
    """Body for ``POST /datasets/intelligence/suggest-mapping``."""

    model_config = ConfigDict(extra="forbid")

    column_names: list[str]
    column_samples: dict[str, list[str]] = Field(default_factory=dict)
    sample_data: list[dict[str, Any]] = Field(default_factory=list)
    target_format: str | None = None


class ValidateStrategyRequest(BaseModel):
    """Body for ``POST /datasets/intelligence/validate-strategy`` (no LLM)."""

    model_config = ConfigDict(extra="forbid")

    strategy: ParsingStrategy
    sample: list[dict[str, Any]] | str
