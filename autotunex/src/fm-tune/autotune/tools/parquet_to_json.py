# coding=utf-8
# Copyright 2023-present International Business Machines Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert a parquet file to JSON or JSONL.

The inverse of the dataset *builders* in this package (e.g.
``build_gsm8k_dataset.py``, which write parquet): given an existing parquet
file — a verl-schema ``train.parquet``, an SFT/RL split, anything — emit a
human-readable export for inspection, diffing, or tools that want JSON.

Two output formats:

- ``jsonl`` (default) — one JSON object per line.
- ``json`` — a single indented array of objects.

Usage:

    python -m autotune.tools.parquet_to_json data/train.parquet
    python -m autotune.tools.parquet_to_json data/train.parquet --format json
    python -m autotune.tools.parquet_to_json in.parquet --output out.jsonl

The output path defaults to the input path with its extension swapped to
``.json``/``.jsonl``; override with ``--output``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional


def _coerce(obj: Any) -> Any:
    """Recursively coerce arrow/numpy values into JSON-serializable natives.

    ``Table.to_pylist()`` returns mostly-native containers, but object/nested
    columns can still surface numpy scalars, numpy arrays, and ``bytes`` — none
    of which ``json.dumps`` handles. Walk the structure and normalize them.
    """
    if isinstance(obj, dict):
        return {k: _coerce(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce(v) for v in obj]
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")

    # numpy scalars expose .item(); numpy arrays expose .tolist(). Import lazily
    # so the helper stays usable without numpy on the path.
    tolist = getattr(obj, "tolist", None)
    if tolist is not None and not isinstance(obj, (str, bytes)):
        return _coerce(tolist())
    item = getattr(obj, "item", None)
    if item is not None and not isinstance(obj, (str, bytes)):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return obj


def _default_output(input_path: str, fmt: str) -> str:
    """Swap the input extension for ``.{fmt}``."""
    root, _ = os.path.splitext(input_path)
    return f"{root}.{fmt}"


def _parquet_to_records(input_path: str) -> List[Dict[str, Any]]:
    """Read a parquet file into a list of JSON-serializable dicts."""
    import pyarrow.parquet as pq

    table = pq.read_table(input_path)
    return [_coerce(row) for row in table.to_pylist()]


def _write_records(
    records: List[Dict[str, Any]],
    output_path: str,
    fmt: str,
    indent: int,
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        if fmt == "jsonl":
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False))
                f.write("\n")
        else:  # json
            json.dump(records, f, ensure_ascii=False, indent=indent)
            f.write("\n")


def convert(
    input_path: str,
    *,
    fmt: str,
    output_path: Optional[str],
    indent: int,
) -> str:
    """Convert ``input_path`` to ``fmt``, returning the written output path."""
    if not input_path.lower().endswith(".parquet"):
        raise ValueError(f"input must be a .parquet file; got {input_path!r}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"input file not found: {input_path}")

    out = output_path or _default_output(input_path, fmt)
    records = _parquet_to_records(input_path)
    _write_records(records, out, fmt, indent)
    print(f"wrote {len(records)} records to {out}")
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="parquet_to_json",
        description="Convert a parquet file to JSON or JSONL.",
    )
    p.add_argument("input", help="Path to the input .parquet file.")
    p.add_argument(
        "--format",
        choices=("json", "jsonl"),
        default="jsonl",
        help="Output format: 'jsonl' (one object per line) or 'json' (array). Default: jsonl.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output path. Defaults to the input path with its extension swapped to .json/.jsonl.",
    )
    p.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation for --format json (ignored for jsonl). Default: 2.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    convert(
        args.input,
        fmt=args.format,
        output_path=args.output,
        indent=args.indent,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
