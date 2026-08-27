"""Pure streaming/parse/split/remap helpers, no HTTP or DB.

Each format (jsonl/csv/parquet) is exercised for counting, reading, splitting
and remapping; the gzip path and the size cap are pinned on the streaming
helper; the split is asserted reproducible for one seed and guarded against
empty splits.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autotunex.core.exceptions import (
    DatasetTooLargeError,
    EmptySplitError,
    UnsupportedDatasetFormatError,
)
from autotunex.services.datasets_io import (
    _validation_indices,
    count_records,
    normalize_json_array_to_jsonl,
    read_records,
    remap_records,
    sniff_format,
    split_by_percentage,
    stream_to_staging,
)


class _AsyncBytes:
    """A minimal async-readable over an in-memory buffer (satisfies SupportsAsyncRead)."""

    def __init__(self, data: bytes) -> None:
        self._buffer = io.BytesIO(data)

    async def read(self, size: int) -> bytes:
        return self._buffer.read(size)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_parquet_multi_row_group(
    path: Path, rows: list[dict[str, Any]], *, row_group_size: int
) -> None:
    """Write a parquet file with multiple row groups, to exercise batched streaming."""
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=row_group_size)


def _fail_if_called(name: str) -> Callable[..., Any]:
    """Return a stand-in that fails loudly if called (proves a code path is not taken)."""

    def _boom(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 — stand-in for any call signature
        raise AssertionError(f"{name} must not be called — parquet streaming regressed")

    return _boom


# stream_to_staging.


async def test_stream_writes_plain_bytes_and_returns_the_count(tmp_path: Path) -> None:
    dest = tmp_path / "out.jsonl"

    written = await stream_to_staging(
        _AsyncBytes(b"hello world"), dest, max_bytes=1024, gzip_encoded=False
    )

    assert written == 11
    assert dest.read_bytes() == b"hello world"


async def test_stream_decodes_gzip(tmp_path: Path) -> None:
    dest = tmp_path / "out.jsonl"
    payload = b'{"a": 1}\n'

    written = await stream_to_staging(
        _AsyncBytes(gzip.compress(payload)), dest, max_bytes=1024, gzip_encoded=True
    )

    assert dest.read_bytes() == payload
    assert written == len(payload)


async def test_stream_enforces_the_size_cap(tmp_path: Path) -> None:
    dest = tmp_path / "out.jsonl"

    with pytest.raises(DatasetTooLargeError):
        await stream_to_staging(_AsyncBytes(b"x" * 100), dest, max_bytes=10, gzip_encoded=False)


async def test_stream_enforces_the_size_cap_on_decompressed_gzip_output(
    tmp_path: Path,
) -> None:
    """A tiny, highly-compressible gzip payload must still be capped on decompressed size.

    Guards against a zip bomb: the compressed input here is a few KiB but
    expands past ``max_bytes`` once decompressed, so the cap must trip during
    decompression rather than only on the (small) compressed byte count.
    """
    dest = tmp_path / "out.jsonl"
    max_bytes = 1024
    payload = b"x" * (max_bytes * 10)

    with pytest.raises(DatasetTooLargeError):
        await stream_to_staging(
            _AsyncBytes(gzip.compress(payload)),
            dest,
            max_bytes=max_bytes,
            gzip_encoded=True,
        )


# sniff_format.


def test_sniff_maps_extensions_to_formats() -> None:
    assert sniff_format("data.jsonl") == "jsonl"
    assert sniff_format("data.csv") == "csv"
    assert sniff_format("data.parquet") == "parquet"
    assert sniff_format("data.jsonl.gz") == "jsonl"


def test_sniff_rejects_an_unknown_extension() -> None:
    with pytest.raises(UnsupportedDatasetFormatError):
        sniff_format("data.txt")


# count_records / read_records.


def test_count_and_read_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    _write_jsonl(path, [{"i": 0}, {"i": 1}, {"i": 2}])

    assert count_records(path, "jsonl") == 3
    assert read_records(path, "jsonl", limit=2) == [{"i": 0}, {"i": 1}]


def test_count_csv_excludes_the_header(tmp_path: Path) -> None:
    path = tmp_path / "d.csv"
    _write_csv(path, [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}])

    assert count_records(path, "csv") == 2


def test_count_parquet_uses_metadata(tmp_path: Path) -> None:
    path = tmp_path / "d.parquet"
    _write_parquet(path, [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}])

    assert count_records(path, "parquet") == 4


def test_read_parquet_streams_without_loading_whole_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "d.parquet"
    _write_parquet_multi_row_group(path, [{"a": i} for i in range(50)], row_group_size=10)
    monkeypatch.setattr(
        "autotunex.services.datasets_io.pq.read_table", _fail_if_called("read_table")
    )

    records = read_records(path, "parquet", limit=3)

    assert records == [{"a": 0}, {"a": 1}, {"a": 2}]


def test_read_records_returns_empty_when_limit_is_zero(tmp_path: Path) -> None:
    path = tmp_path / "d.parquet"
    _write_parquet_multi_row_group(path, [{"a": i} for i in range(5)], row_group_size=2)

    assert read_records(path, "parquet", limit=0) == []


# split_by_percentage.


def test_split_is_reproducible_for_a_seed(tmp_path: Path) -> None:
    source = tmp_path / "src.jsonl"
    _write_jsonl(source, [{"i": n} for n in range(10)])
    train_a, val_a = tmp_path / "ta.jsonl", tmp_path / "va.jsonl"
    train_b, val_b = tmp_path / "tb.jsonl", tmp_path / "vb.jsonl"

    counts_a = split_by_percentage(
        source, train_a, val_a, data_format="jsonl", validation_percentage=20, seed=7
    )
    counts_b = split_by_percentage(
        source, train_b, val_b, data_format="jsonl", validation_percentage=20, seed=7
    )

    assert counts_a == (8, 2)
    assert counts_a == counts_b
    assert val_a.read_text() == val_b.read_text()


def test_split_rejects_an_empty_validation_split(tmp_path: Path) -> None:
    source = tmp_path / "src.jsonl"
    _write_jsonl(source, [{"i": n} for n in range(3)])

    with pytest.raises(EmptySplitError):
        split_by_percentage(
            source,
            tmp_path / "t.jsonl",
            tmp_path / "v.jsonl",
            data_format="jsonl",
            validation_percentage=0,
            seed=1,
        )


def test_split_parquet_streams_with_exact_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "d.parquet"
    _write_parquet_multi_row_group(path, [{"a": i} for i in range(20)], row_group_size=5)
    monkeypatch.setattr(
        "autotunex.services.datasets_io.pq.read_table", _fail_if_called("read_table")
    )
    monkeypatch.setattr("autotunex.services.datasets_io.PARQUET_BATCH_SIZE", 7)
    train_out, validation_out = tmp_path / "train.parquet", tmp_path / "val.parquet"

    train_n, validation_n = split_by_percentage(
        path,
        train_out,
        validation_out,
        data_format="parquet",
        validation_percentage=25,
        seed=7,
    )

    assert (train_n, validation_n) == (15, 5)
    assert count_records(train_out, "parquet") == 15
    assert count_records(validation_out, "parquet") == 5


def test_split_parquet_partitions_all_rows_without_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "d.parquet"
    _write_parquet_multi_row_group(path, [{"a": i} for i in range(20)], row_group_size=5)
    monkeypatch.setattr("autotunex.services.datasets_io.PARQUET_BATCH_SIZE", 7)
    train_out, validation_out = tmp_path / "train.parquet", tmp_path / "val.parquet"

    split_by_percentage(
        path,
        train_out,
        validation_out,
        data_format="parquet",
        validation_percentage=25,
        seed=7,
    )

    train_vals = {r["a"] for r in read_records(train_out, "parquet", limit=100)}
    val_vals = {r["a"] for r in read_records(validation_out, "parquet", limit=100)}
    assert train_vals | val_vals == set(range(20))
    assert train_vals & val_vals == set()
    assert val_vals == _validation_indices(20, 25, 7)


# remap_records.


def test_remap_jsonl_renames_keys(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    _write_jsonl(path, [{"instruction": "hi", "extra": "x"}])

    remap_records(path, {"prompt": "instruction"}, data_format="jsonl")

    assert read_records(path, "jsonl", limit=1) == [{"prompt": "hi"}]


def test_remap_csv_renames_columns(tmp_path: Path) -> None:
    path = tmp_path / "d.csv"
    _write_csv(path, [{"instruction": "hi", "extra": "x"}])

    remap_records(path, {"prompt": "instruction"}, data_format="csv")

    assert read_records(path, "csv", limit=1) == [{"prompt": "hi"}]


def test_remap_parquet_renames_columns(tmp_path: Path) -> None:
    path = tmp_path / "d.parquet"
    _write_parquet(path, [{"instruction": "hi", "extra": "x"}])

    remap_records(path, {"prompt": "instruction"}, data_format="parquet")

    assert read_records(path, "parquet", limit=1) == [{"prompt": "hi"}]


def test_remap_parquet_streams_and_keeps_all_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "d.parquet"
    _write_parquet_multi_row_group(
        path,
        [{"instruction": f"q{i}", "extra": "x"} for i in range(50)],
        row_group_size=10,
    )
    monkeypatch.setattr(
        "autotunex.services.datasets_io.pq.read_table", _fail_if_called("read_table")
    )
    monkeypatch.setattr("autotunex.services.datasets_io.PARQUET_BATCH_SIZE", 7)

    remap_records(path, {"prompt": "instruction"}, data_format="parquet")

    rows = read_records(path, "parquet", limit=100)
    assert len(rows) == 50
    assert rows[0] == {"prompt": "q0"}
    assert rows[49] == {"prompt": "q49"}


def test_remap_jsonl_skips_blank_source_instead_of_raising(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    _write_jsonl(path, [{"instruction": "hi"}])

    # An unassigned target (source "") is dropped, not a KeyError: '' (the 2025 behaviour).
    remap_records(path, {"prompt": "instruction", "output": ""}, data_format="jsonl")

    assert read_records(path, "jsonl", limit=1) == [{"prompt": "hi"}]


def test_remap_jsonl_skips_a_source_absent_from_the_row(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    _write_jsonl(path, [{"instruction": "hi"}])

    remap_records(path, {"prompt": "instruction", "answer": "response"}, data_format="jsonl")

    assert read_records(path, "jsonl", limit=1) == [{"prompt": "hi"}]


def test_remap_csv_skips_blank_source(tmp_path: Path) -> None:
    path = tmp_path / "d.csv"
    _write_csv(path, [{"instruction": "hi"}])

    remap_records(path, {"prompt": "instruction", "output": ""}, data_format="csv")

    assert read_records(path, "csv", limit=1) == [{"prompt": "hi"}]


def test_remap_parquet_skips_blank_source(tmp_path: Path) -> None:
    path = tmp_path / "d.parquet"
    _write_parquet(path, [{"instruction": "hi"}])

    remap_records(path, {"prompt": "instruction", "output": ""}, data_format="parquet")

    assert read_records(path, "parquet", limit=1) == [{"prompt": "hi"}]


# normalize_json_array_to_jsonl.
#
# The uploader maps a `.json` upload to the `jsonl` format and streams the raw
# bytes, so a standard (often pretty-printed) JSON array arrives on disk as a
# `.jsonl`-named file that the line-by-line JSONL machinery cannot read. These
# pin the streaming array→JSONL normalization that fixes that.


def test_normalize_converts_a_pretty_printed_json_array_to_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"  # staged name is .jsonl even for a `.json` upload
    rows = [{"text": "one", "label": 0}, {"text": "two", "label": 1}]
    path.write_text(json.dumps(rows, indent=4))

    normalize_json_array_to_jsonl(path)

    assert count_records(path, "jsonl") == 2
    assert read_records(path, "jsonl", limit=2) == rows


def test_normalize_leaves_a_jsonl_file_untouched(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    original = '{"i": 0}\n{"i": 1}\n'
    path.write_text(original)

    normalize_json_array_to_jsonl(path)

    assert path.read_text() == original


def test_normalize_preserves_nested_values_and_bracket_bearing_strings(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    rows = [
        {"meta": {"tags": ["a", "b"]}, "text": "has ], }, and , inside"},
        {"meta": {"tags": []}, "text": "الجزيرة"},
    ]
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    normalize_json_array_to_jsonl(path)

    assert read_records(path, "jsonl", limit=2) == rows


def test_normalize_streams_elements_across_chunk_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A tiny read chunk forces single elements to span many reads, exercising
    # the refill-and-retry path and proving memory stays bounded (one element
    # plus one chunk), never the whole file.
    monkeypatch.setattr("autotunex.services.datasets_io.CHUNK_SIZE", 4)
    path = tmp_path / "d.jsonl"
    rows = [{"i": n, "s": "x" * 10} for n in range(20)]
    path.write_text(json.dumps(rows, indent=2))

    normalize_json_array_to_jsonl(path)

    assert read_records(path, "jsonl", limit=100) == rows


def test_normalize_handles_a_single_line_compact_array(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text('[{"a": 1}, {"a": 2}]')

    normalize_json_array_to_jsonl(path)

    assert read_records(path, "jsonl", limit=2) == [{"a": 1}, {"a": 2}]


def test_normalize_of_an_empty_array_yields_no_records(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text("[]")

    normalize_json_array_to_jsonl(path)

    assert count_records(path, "jsonl") == 0


def test_normalize_raises_on_a_truncated_array(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text('[{"a": 1}, {"a": 2}')  # no closing ]

    with pytest.raises(UnsupportedDatasetFormatError):
        normalize_json_array_to_jsonl(path)
