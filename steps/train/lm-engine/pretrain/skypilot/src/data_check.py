# SPDX-License-Identifier: Apache-2.0

"""Fail fast on a bad dataset or tokenizer, on a CPU pod, before ``train`` runs.

``recipes/pretrain`` puts this in a ``data-check`` target ahead of ``train`` so a
mis-mounted PVC, a typo'd sub-dataset name or a tokenizer whose special-token ids
disagree with the model knobs costs seconds on one CPU pod instead of a
multi-node GPU allocation that builds its dataloader and dies.

What it checks, and where each check comes from in the pinned lm-engine
(``steps/train/lm-engine/UPSTREAM``, patch 0001 restores the IBM scalable
dataloader as ``lm_engine/training/data/ibm.py``):

* ``dataset_path`` is a non-empty directory -- ``_StatefulDataset.__init__``
  asserts "Data path ... must be a non-empty folder".
* it holds sub-dataset *subfolders*. ``SamplingDataset.get_datasets_weights()``
  treats ``dataset_path`` as a parent of one folder per sub-dataset: with
  ``sub_datasets`` null it discovers every entry that is not a file and whose
  name does not contain ``meta`` (the optional ``meta/counts.csv`` sidecar), and
  with ``sub_datasets`` set it uses exactly those names. A named sub-dataset that
  does not exist is a silent 0-shard iterator, so every name is checked.
* ``weights``, if set, has one positive number per sub-dataset -- the same
  assertions ``get_datasets_weights()`` makes, which ``render_config.validate``
  does not cover.
* each sub-dataset holds at least one shard file, and the first of them (sorted)
  carries ``col_name``. ``StreamingDocDataset.setup()`` walks the sub-dataset
  recursively and keeps files whose extension contains ``arrow`` (file_type
  ``arrow``) or ``parquet`` (``hf_parquet``), so the walk is recursive here too.
  For ``hf_parquet``, ``ParquetHandler.open()`` additionally asserts that exactly
  one of ``text``/``content``/``contents`` is present and then reads *that*
  column -- a ``col_name`` outside that set would silently train on another
  column, so both are checked.
* the tokenizer directory loads, and its ``bos``/``eos``/``pad`` ids match the
  knobs -- ``model_wrapper/base.py::_setup_model`` asserts all three at model
  construction, each guarded by ``is not None`` on the *tokenizer* side. A
  tokenizer id of ``None`` therefore never trips the assert and is not an error
  here, whatever the knob says; a tokenizer id that is set must equal the knob,
  including when the knob is null.
* the tokenizer is not bigger than ``vocab_size``: equal is fine (ids are
  ``0..vocab_size-1``), one more is not.

Only schemas are read, never a table: these datasets are hundreds of GB on a PVC.
``pyarrow`` and ``transformers`` are imported inside the functions that need them,
so the module -- and ``--help`` -- work without either.

CLI (``run_pretrain.sh`` dispatches ``LME_PHASE=data-check`` here first):

    python data_check.py --knobs '<the config.lme dict, as JSON>'

Every failure is printed, not just the first; exit 1 if there is any.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable

FILE_TYPES = ("arrow", "hf_parquet")
#: Extension substring ``_ShardFileHandler.is_legal()`` matches, per file type.
_SHARD_EXT = {"arrow": "arrow", "hf_parquet": "parquet"}
#: ``ParquetHandler.open()`` requires exactly one of these in the schema.
_PARQUET_LEGAL_FIELDS = ("text", "content", "contents")
_ID_KNOBS = (
    ("bos_token_id", "bos_token_id"),
    ("eos_token_id", "eos_token_id"),
    ("pad_token_id", "pad_token_id"),
)


# ── helpers ─────────────────────────────────────────────────────────────────────


def _is_set(value: Any) -> bool:
    """Whether an optional knob carries a value (``render_config._is_set``)."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in ("", "null", "none")
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    return True


def _splitstrip(value: Any) -> list:
    """``ibm.parse_data_args``'s splitstrip: csv string or list -> list."""
    if isinstance(value, str):
        return [item.strip() for item in value.split(",")]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


# ── dataset ─────────────────────────────────────────────────────────────────────


def _discover(dataset_path: str) -> list[str]:
    """Auto-discovery, as ``SamplingDataset.get_datasets_weights()`` does it."""
    return sorted(
        name
        for name in os.listdir(dataset_path)
        if not os.path.isfile(os.path.join(dataset_path, name)) and "meta" not in name
    )


def _shard_files(folder: str, ext: str) -> list[str]:
    """Shard files under a sub-dataset, as ``StreamingDocDataset.setup()`` walks."""
    found = [
        os.path.join(root, name)
        for root, _dirs, files in os.walk(folder)
        for name in files
        if ext in os.path.splitext(name)[1]
    ]
    found.sort()
    return found


def _arrow_columns(path: str) -> list[str]:
    import pyarrow as pa

    with pa.memory_map(path) as source:
        return list(pa.ipc.open_file(source).schema.names)


def _parquet_columns(path: str) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.read_metadata(path).schema.names)


def _check_columns(path: str, file_type: str, col_name: str) -> list[str]:
    """Schema-only column check for one shard file."""
    try:
        columns = (
            _arrow_columns(path) if file_type == "arrow" else _parquet_columns(path)
        )
    except Exception as exc:  # unreadable, truncated, wrong format
        return [f"cannot read the {file_type} schema of {path}: {exc}"]

    errors: list[str] = []
    if col_name not in columns:
        errors.append(f"{path} has no column {col_name!r} (columns: {sorted(columns)})")
    if file_type == "hf_parquet":
        # ParquetHandler.open() ignores col_name and reads the one legal field.
        overlap = sorted(set(_PARQUET_LEGAL_FIELDS) & set(columns))
        if len(overlap) != 1:
            errors.append(
                f"{path} must have exactly one of {list(_PARQUET_LEGAL_FIELDS)} "
                f"(has {overlap}); lm-engine's ParquetHandler asserts this"
            )
        elif overlap[0] != col_name:
            errors.append(
                f"col_name {col_name!r} is not one of "
                f"{list(_PARQUET_LEGAL_FIELDS)}, so lm-engine would read "
                f"{overlap[0]!r} from {path} instead"
            )
    return errors


def check_data(knobs: dict) -> list[str]:
    """Errors about ``dataset_path``/``sub_datasets``/``weights``/shard files."""
    errors: list[str] = []

    file_type = knobs.get("file_type")
    if file_type not in FILE_TYPES:
        errors.append(
            f"file_type must be one of {list(FILE_TYPES)} (got {file_type!r})"
        )

    dataset_path = str(knobs.get("dataset_path") or "").strip()
    if not dataset_path:
        errors.append("dataset_path is not set; the recipe must pass a dataset path")
        return errors
    if not os.path.isdir(dataset_path):
        what = (
            "is not a directory" if os.path.exists(dataset_path) else "does not exist"
        )
        errors.append(f"dataset_path {dataset_path} {what}")
        return errors

    if _is_set(knobs.get("sub_datasets")):
        sub_datasets = [str(name) for name in _splitstrip(knobs["sub_datasets"])]
        missing = [
            name
            for name in sub_datasets
            if not os.path.isdir(os.path.join(dataset_path, name))
        ]
        if missing:
            errors.append(
                f"sub_datasets not found under {dataset_path}: {missing} "
                f"(present: {_discover(dataset_path)})"
            )
        sub_datasets = [name for name in sub_datasets if name not in missing]
    else:
        sub_datasets = _discover(dataset_path)
        if not sub_datasets:
            errors.append(
                f"dataset_path {dataset_path} holds no sub-dataset subfolder; "
                "lm-engine expects one folder of shard files per sub-dataset"
            )

    errors += _check_weights(knobs, len(sub_datasets))

    if file_type in FILE_TYPES:
        ext = _SHARD_EXT[file_type]
        for name in sub_datasets:
            folder = os.path.join(dataset_path, name)
            shards = _shard_files(folder, ext)
            if not shards:
                errors.append(
                    f"sub-dataset {name} ({folder}) holds no *.{ext} file, "
                    f"which file_type {file_type!r} requires"
                )
                continue
            errors += _check_columns(shards[0], file_type, str(knobs.get("col_name")))
    return errors


def _check_weights(knobs: dict, num_datasets: int) -> list[str]:
    """``get_datasets_weights()``: one positive weight per sub-dataset."""
    if not _is_set(knobs.get("weights")):
        return []
    raw = _splitstrip(knobs["weights"])
    try:
        weights = [float(value) for value in raw]
    except (TypeError, ValueError):
        return [f"weights must all be numbers (got {raw})"]
    if len(weights) != num_datasets:
        return [
            f"weights has {len(weights)} value(s) but there are {num_datasets} "
            "sub-dataset(s); lm-engine asserts the two match"
        ]
    bad = [value for value in weights if value <= 0]
    if bad:
        return [f"every weight must be positive (got {bad})"]
    return []


# ── tokenizer ───────────────────────────────────────────────────────────────────


def load_tokenizer(tokenizer_name: str):
    """The real loader; a seam the tests replace. ``transformers`` is lazy."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_name)


def _vocab_size(tokenizer) -> int | None:
    try:
        return len(tokenizer)
    except TypeError:
        pass
    size = getattr(tokenizer, "vocab_size", None)
    return int(size) if isinstance(size, int) else None


def check_tokenizer(knobs: dict, load: Callable[[str], Any] | None = None) -> list[str]:
    """Errors about the tokenizer directory, its special-token ids and its size."""
    tokenizer_name = str(knobs.get("tokenizer_name") or "").strip()
    if not tokenizer_name:
        return ["tokenizer_name is not set; the recipe must pass a tokenizer path"]

    try:
        tokenizer = (load or load_tokenizer)(tokenizer_name)
    except Exception as exc:
        return [f"cannot load the tokenizer at {tokenizer_name}: {exc}"]

    errors: list[str] = []
    for attr, knob in _ID_KNOBS:
        # model_wrapper/base.py asserts only when the TOKENIZER's id is set.
        tokenizer_id = getattr(tokenizer, attr, None)
        if tokenizer_id is None:
            continue
        expected = knobs.get(knob) if _is_set(knobs.get(knob)) else None
        if expected is None or int(expected) != int(tokenizer_id):
            errors.append(
                f"tokenizer {attr} is {tokenizer_id} but the knob {knob} is "
                f"{knobs.get(knob)!r}; lm-engine asserts they are equal"
            )

    vocab = _vocab_size(tokenizer)
    try:
        vocab_size = int(knobs.get("vocab_size"))
    except (TypeError, ValueError):
        errors.append(
            f"vocab_size must be an integer (got {knobs.get('vocab_size')!r})"
        )
        return errors
    if vocab is not None and vocab > vocab_size:
        errors.append(
            f"the tokenizer at {tokenizer_name} has {vocab} tokens, more than "
            f"vocab_size {vocab_size}; the model has no embedding row for the rest"
        )
    return errors


# ── check ───────────────────────────────────────────────────────────────────────


def check(
    knobs: dict, *, load_tokenizer: Callable[[str], Any] | None = None
) -> list[str]:
    """Every error in the knobs' dataset and tokenizer (empty = ok)."""
    return check_data(knobs) + check_tokenizer(knobs, load_tokenizer)


# ── CLI ─────────────────────────────────────────────────────────────────────────


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--knobs", required=True, help="the config.lme dict, as JSON")
    args = parser.parse_args(argv)

    try:
        knobs = json.loads(args.knobs)
    except json.JSONDecodeError as exc:
        print(f"error: --knobs is not valid JSON: {exc}", file=sys.stderr)
        return 1

    # No loader passed: check_tokenizer() falls back to the module-level
    # load_tokenizer, which a test (and a caller) can swap.
    errors = check(knobs)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("data-check: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
