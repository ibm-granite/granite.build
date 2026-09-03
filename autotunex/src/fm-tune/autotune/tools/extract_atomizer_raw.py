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

"""Extract raw atomizer records from a factuality raw JSON file.

The factuality raw files carry, per record, the query, the assistant response,
and the response's gold decomposition spread across ``a0``/``a1``/``a2``/... keys
(each an object with a ``text`` field). This tool pulls those three pieces out
into the flat shape ``build_atomizer_dataset.py`` expects::

    {"query": ..., "response": ..., "atoms": ["...", "..."]}

Usage:

    python -m autotune.tools.extract_atomizer_raw \
        --input /path/to/eli5_raw_train.json \
        --output-dir /path/to/atomizer

The output filename is derived from the input file: ``eli5_raw_train.json`` ->
``eli5_atomizer_raw_train.json`` (the ``_raw_`` segment is renamed so the source
split is still visible). Records with no usable atoms are dropped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

# Atom keys are ``a`` followed by digits — ``a0``, ``a1``, ... — and must not be
# confused with the ``c_a*`` context keys, which the anchored match excludes.
_ATOM_KEY = re.compile(r"a(\d+)$")


def _atom_sort_key(key: str) -> int:
    """Numeric order for atom keys so a2 precedes a10."""
    m = _ATOM_KEY.fullmatch(key)
    return int(m.group(1)) if m else sys.maxsize


def extract_atoms(record: Dict[str, Any]) -> List[str]:
    """Pull the ``a*`` atom texts out of a record, in numeric key order."""
    keys = sorted(
        (k for k in record if _ATOM_KEY.fullmatch(k) and isinstance(record[k], dict)),
        key=_atom_sort_key,
    )
    atoms = []
    for k in keys:
        text = record[k].get("text")
        if isinstance(text, str) and text.strip():
            atoms.append(text.strip())
    return atoms


def extract_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build one flat atomizer record, or None when it is unusable."""
    query = record.get("query")
    response = record.get("response")
    # The factuality schema nests the response text under ``response.text``.
    if isinstance(response, dict):
        response = response.get("text")
    if not isinstance(query, str) or not query.strip():
        return None
    if not isinstance(response, str) or not response.strip():
        return None
    atoms = extract_atoms(record)
    if not atoms:
        return None
    return {"query": query.strip(), "response": response.strip(), "atoms": atoms}


def output_name_for(input_path: str) -> str:
    """``eli5_raw_train.json`` -> ``eli5_atomizer_raw_train.json``."""
    stem = os.path.splitext(os.path.basename(input_path))[0]
    if "_raw_" in stem:
        head, _, tail = stem.partition("_raw_")
        return f"{head}_atomizer_raw_{tail}.json"
    return f"{stem}_atomizer_raw.json"


def convert(input_path: str, output_path: str) -> None:
    """Extract every usable atomizer record from one raw factuality file."""
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"{input_path}: expected a top-level JSON list")

    out: List[Dict[str, Any]] = []
    skipped = 0
    for dp in records:
        rec = extract_record(dp) if isinstance(dp, dict) else None
        if rec is None:
            skipped += 1
            continue
        out.append(rec)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    total_atoms = sum(len(r["atoms"]) for r in out)
    mean = round(total_atoms / len(out), 1) if out else 0
    print(
        f"[atomizer-raw] wrote {len(out)} records ({total_atoms} atoms, "
        f"{mean} per record) to {output_path} (skipped {skipped} unusable)"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="extract_atomizer_raw",
        description="Extract query/response/atoms from a factuality raw JSON file.",
    )
    p.add_argument("--input", required=True, help="Raw factuality JSON file (top-level list).")
    p.add_argument("--output-dir", required=True, help="Directory to write the raw atomizer JSON into.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not os.path.exists(args.input):
        print(f"[error] {args.input} not found", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    convert(args.input, os.path.join(args.output_dir, output_name_for(args.input)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
