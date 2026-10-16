"""Schema tests for the dataset-intelligence contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from autotunex.models.dataset_intelligence import (
    ColumnMappingSuggestion,
    ParseStrategyRequest,
    ParsingStrategy,
    SuggestMappingRequest,
    ValidateStrategyRequest,
    ValidationResult,
)


def test_parse_request_accepts_rows_or_raw_text() -> None:
    rows = ParseStrategyRequest(sample=[{"q": "a"}], data_format="jsonl")
    text = ParseStrategyRequest(sample="raw text", data_format="txt")

    assert rows.sample == [{"q": "a"}]
    assert text.sample == "raw text"


def test_parse_request_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ParseStrategyRequest(sample=[{"q": "a"}], surprise="x")  # type: ignore[call-arg]


def test_column_mapping_is_flat_with_a_confidence_sidecar() -> None:
    suggestion = ColumnMappingSuggestion(
        dataset_format="sft",
        tuning_type="sft",
        confidence=0.9,
        column_mapping={"prompt": "question", "completion": "answer"},
        column_confidence={"prompt": 0.95},
        reasoning="looks like QA",
    )

    assert suggestion.column_mapping == {"prompt": "question", "completion": "answer"}
    assert suggestion.column_confidence == {"prompt": 0.95}


def test_column_mapping_defaults_confidence_sidecar_to_empty() -> None:
    suggestion = ColumnMappingSuggestion(
        dataset_format="grpo", tuning_type="grpo", column_mapping={"prompt": "q"}
    )

    assert suggestion.column_confidence == {}


def test_parsing_strategy_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValidationError):
        ParsingStrategy(type="direct_mapping", confidence=1.5)


def test_validate_request_embeds_a_parsing_strategy() -> None:
    request = ValidateStrategyRequest(
        strategy=ParsingStrategy(type="direct_mapping", input_field="q", output_field="a"),
        sample=[{"q": "x", "a": "y"}],
    )

    assert request.strategy.type == "direct_mapping"


def test_validation_result_rejects_negative_count() -> None:
    with pytest.raises(ValidationError):
        ValidationResult(success=False, parsed_count=-1)


def test_suggest_request_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SuggestMappingRequest(column_names=["a"], nope=1)  # type: ignore[call-arg]
