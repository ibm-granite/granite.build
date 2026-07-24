# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

from services.datasets.intelligence import DatasetIntelligence


def test_direct_mapping_detected_for_structured_io():
    di = DatasetIntelligence()
    sample = [{"input": "q", "output": "a"}, {"input": "q2", "output": "a2"}]
    assert di._has_io_structure(sample) is True
    strategy = di._create_direct_mapping_strategy(sample)
    assert strategy["type"] == "direct_mapping"
    assert strategy["input_field"] == "input"
    assert strategy["output_field"] == "output"


def test_no_io_structure_for_raw_rows():
    di = DatasetIntelligence()
    assert di._has_io_structure([{"foo": 1, "bar": 2}]) is False


def test_validate_strategy_on_direct_mapping_sample():
    di = DatasetIntelligence()
    strategy = {"type": "direct_mapping", "input_field": "q", "output_field": "a"}
    result = di._validate_strategy_on_sample(strategy, [{"q": "hi", "a": "yo"}])
    assert result["success"] is True
    assert result["parsed_count"] == 1
