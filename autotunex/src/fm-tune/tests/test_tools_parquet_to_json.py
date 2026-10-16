"""Tests for autotune.tools.parquet_to_json."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autotune.tools.parquet_to_json import (
    _coerce,
    _default_output,
    _parquet_to_records,
    convert,
)


class TestCoerce:
    def test_numpy_scalar(self):
        assert _coerce(np.int64(42)) == 42
        assert isinstance(_coerce(np.int64(42)), int)

    def test_numpy_array(self):
        assert _coerce(np.array([1, 2, 3])) == [1, 2, 3]

    def test_bytes(self):
        assert _coerce(b"hello") == "hello"

    def test_nested(self):
        out = _coerce({"a": np.int64(1), "b": [np.float64(2.5), b"x"]})
        assert out == {"a": 1, "b": [2.5, "x"]}

    def test_passthrough(self):
        assert _coerce("s") == "s"
        assert _coerce(None) is None
        assert _coerce(3) == 3

    def test_result_is_json_serializable(self):
        out = _coerce({"prompt": np.array(["a", "b"]), "n": np.int32(7)})
        json.dumps(out)  # must not raise


class TestDefaultOutput:
    def test_swaps_extension(self):
        assert _default_output("/d/train.parquet", "jsonl") == "/d/train.jsonl"
        assert _default_output("/d/train.parquet", "json") == "/d/train.json"

    def test_no_extension(self):
        assert _default_output("/d/train", "jsonl") == "/d/train.jsonl"


def _write_sample_parquet(path):
    table = pa.table(
        {
            "id": [1, 2],
            "text": ["foo", "bar"],
            "tags": [["a", "b"], ["c"]],
        }
    )
    pq.write_table(table, path)


class TestRoundTrip:
    def test_parquet_to_records(self, tmp_path):
        src = tmp_path / "data.parquet"
        _write_sample_parquet(str(src))
        records = _parquet_to_records(str(src))
        assert records == [
            {"id": 1, "text": "foo", "tags": ["a", "b"]},
            {"id": 2, "text": "bar", "tags": ["c"]},
        ]

    def test_convert_jsonl(self, tmp_path):
        src = tmp_path / "data.parquet"
        _write_sample_parquet(str(src))
        out = convert(str(src), fmt="jsonl", output_path=None, indent=2)
        assert out == str(tmp_path / "data.jsonl")
        lines = (tmp_path / "data.jsonl").read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"id": 1, "text": "foo", "tags": ["a", "b"]}

    def test_convert_json(self, tmp_path):
        src = tmp_path / "data.parquet"
        _write_sample_parquet(str(src))
        out = convert(str(src), fmt="json", output_path=None, indent=2)
        assert out == str(tmp_path / "data.json")
        loaded = json.loads((tmp_path / "data.json").read_text())
        assert loaded[1] == {"id": 2, "text": "bar", "tags": ["c"]}

    def test_explicit_output_path(self, tmp_path):
        src = tmp_path / "data.parquet"
        _write_sample_parquet(str(src))
        dest = tmp_path / "custom.jsonl"
        out = convert(str(src), fmt="jsonl", output_path=str(dest), indent=2)
        assert out == str(dest)
        assert dest.exists()


class TestValidation:
    def test_non_parquet_raises(self, tmp_path):
        bad = tmp_path / "data.csv"
        bad.write_text("x")
        with pytest.raises(ValueError, match="must be a .parquet file"):
            convert(str(bad), fmt="jsonl", output_path=None, indent=2)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            convert(str(tmp_path / "nope.parquet"), fmt="jsonl", output_path=None, indent=2)
