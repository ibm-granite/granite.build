"""Tests for autotune.tools.build_gsm8k_dataset — pure helpers."""

import pytest

from autotune.tools.build_gsm8k_dataset import (
    _extract_solution,
    _find_split_file,
    _hf_loader_for,
    _make_record,
)


class TestExtractSolution:
    def test_basic(self):
        assert _extract_solution("the answer is #### 42") == "42"

    def test_negative(self):
        assert _extract_solution("answer #### -7") == "-7"

    def test_strips_commas(self):
        assert _extract_solution("answer #### 1,234") == "1234"

    def test_decimal(self):
        assert _extract_solution("#### 3.14") == "3.14"

    def test_missing_marker_raises(self):
        with pytest.raises(ValueError, match="could not find"):
            _extract_solution("just a sentence with no marker")


class TestMakeRecord:
    def test_schema(self):
        out = _make_record(
            example={"question": "2+2?", "answer": "fourish #### 4"},
            idx=0,
            split="train",
            data_source="gsm8k",
            prompt_key="question",
            answer_key="answer",
            instruction_following="Think step by step.",
            system_prompt=None,
        )
        assert out["data_source"] == "gsm8k"
        assert out["ability"] == "math"
        assert out["reward_model"] == {"style": "rule", "ground_truth": "4"}
        assert out["extra_info"] == {"split": "train", "index": 0}
        # Prompt is a single user message when no system prompt
        assert len(out["prompt"]) == 1
        assert out["prompt"][0]["role"] == "user"
        assert "Think step by step." in out["prompt"][0]["content"]

    def test_with_system_prompt(self):
        out = _make_record(
            example={"q": "x", "a": "#### 1"},
            idx=5,
            split="validation",
            data_source="gsm8k",
            prompt_key="q",
            answer_key="a",
            instruction_following="",
            system_prompt="You are a math tutor.",
        )
        assert len(out["prompt"]) == 2
        assert out["prompt"][0]["role"] == "system"
        assert out["prompt"][1]["role"] == "user"

    def test_no_instruction_following(self):
        out = _make_record(
            example={"q": "what?", "a": "#### 9"},
            idx=0,
            split="train",
            data_source="x",
            prompt_key="q",
            answer_key="a",
            instruction_following="",
            system_prompt=None,
        )
        # User content is the bare question
        assert out["prompt"][0]["content"] == "what?"


class TestFindSplitFile:
    def test_jsonl_preferred(self, tmp_path):
        (tmp_path / "train.jsonl").write_text("")
        (tmp_path / "train.csv").write_text("")
        out = _find_split_file(str(tmp_path), "train")
        assert out.endswith("train.jsonl")

    def test_falls_back_to_json(self, tmp_path):
        (tmp_path / "test.json").write_text("")
        out = _find_split_file(str(tmp_path), "test")
        assert out.endswith("test.json")

    def test_falls_back_to_csv(self, tmp_path):
        (tmp_path / "x.csv").write_text("")
        out = _find_split_file(str(tmp_path), "x")
        assert out.endswith("x.csv")

    def test_falls_back_to_parquet(self, tmp_path):
        (tmp_path / "x.parquet").write_bytes(b"")
        out = _find_split_file(str(tmp_path), "x")
        assert out.endswith("x.parquet")

    def test_none_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _find_split_file(str(tmp_path), "missing")


class TestHfLoaderFor:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("data.json", "json"),
            ("data.jsonl", "json"),
            ("data.csv", "csv"),
            ("data.parquet", "parquet"),
        ],
    )
    def test_known_extensions(self, path, expected):
        assert _hf_loader_for(path) == expected

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="unsupported"):
            _hf_loader_for("data.xml")
