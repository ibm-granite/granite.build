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

"""Build a GSM8K dataset in the parquet schema verl expects.

Produces ``train.parquet``, ``validation.parquet``, and ``test.parquet`` under
``--output-dir``. Each row matches verl's NaiveRewardManager contract:

    {
        "data_source": <str>,                # reward_fn_key in verl config
        "prompt": [<chat messages>],         # list of {role, content}
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": <extracted number>},
        "extra_info": {"split": <split>, "index": <int>},
    }

Source is auto-detected: if ``--input-dir`` is given, raw files are loaded from
disk (``.json``/``.jsonl``/``.csv``/``.parquet`` with ``question``/``answer``
columns, or column names overridden via ``--prompt-key``/``--answer-key``);
otherwise the builder pulls from the HuggingFace hub (default
``openai/gsm8k``, config ``main``).

Usage:

    python -m autotune.tools.build_gsm8k_dataset --output-dir /path/to/out

Downstream, point ``training_config.train_file`` / ``eval_file`` at the written
parquet files; the verl driver reads them via ``pyarrow`` (see
``autotune/trainers/driver_multi_verl.py``).
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Any, Dict, List, Optional

DEFAULT_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'
_SOLUTION_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def _extract_solution(answer_str: str) -> str:
    """Extract the canonical GSM8K ``#### <number>`` answer, commas stripped."""
    m = _SOLUTION_RE.search(answer_str)
    if m is None:
        raise ValueError(f"could not find '#### <number>' marker in answer: {answer_str!r}")
    return m.group(0).split("#### ", 1)[1].replace(",", "")


def _make_record(
    example: Dict[str, Any],
    idx: int,
    *,
    split: str,
    data_source: str,
    prompt_key: str,
    answer_key: str,
    instruction_following: str,
    system_prompt: Optional[str],
) -> Dict[str, Any]:
    question = example[prompt_key]
    answer = example[answer_key]
    solution = _extract_solution(answer)

    user_content = f"{question} {instruction_following}" if instruction_following else question
    prompt: List[Dict[str, str]] = []
    if system_prompt:
        prompt.append({"role": "system", "content": system_prompt})
    prompt.append({"role": "user", "content": user_content})

    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {"split": split, "index": idx},
    }


def _load_raw(input_dir: Optional[str], hf_name: str, hf_config: Optional[str]):
    """Return a dict with ``train`` and ``test`` HF Dataset objects."""
    from datasets import load_dataset

    if input_dir:
        train_path = _find_split_file(input_dir, "train")
        test_path = _find_split_file(input_dir, "test")
        train = load_dataset(_hf_loader_for(train_path), data_files=train_path, split="train")
        test = load_dataset(_hf_loader_for(test_path), data_files=test_path, split="train")
        return {"train": train, "test": test}

    ds = load_dataset(hf_name, hf_config) if hf_config else load_dataset(hf_name)
    if "train" not in ds or "test" not in ds:
        raise ValueError(
            f"HF dataset {hf_name}/{hf_config} must expose 'train' and 'test' splits; got {list(ds.keys())}"
        )
    return {"train": ds["train"], "test": ds["test"]}


def _find_split_file(input_dir: str, split: str) -> str:
    """Locate a raw split file by trying common extensions."""
    for ext in ("jsonl", "json", "csv", "parquet"):
        candidate = os.path.join(input_dir, f"{split}.{ext}")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"no {split}.{{jsonl,json,csv,parquet}} file found in {input_dir}")


def _hf_loader_for(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext in ("json", "jsonl"):
        return "json"
    if ext == "csv":
        return "csv"
    if ext == "parquet":
        return "parquet"
    raise ValueError(f"unsupported raw file extension: {path}")


def _build_splits(raw: Dict[str, Any], val_ratio: float, seed: int):
    """Carve a validation slice out of train; keep test as-is."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"--val-ratio must be in (0, 1); got {val_ratio}")
    split = raw["train"].train_test_split(test_size=val_ratio, seed=seed)
    return {"train": split["train"], "validation": split["test"], "test": raw["test"]}


def convert(
    output_dir: str,
    *,
    input_dir: Optional[str],
    hf_name: str,
    hf_config: Optional[str],
    data_source: str,
    prompt_key: str,
    answer_key: str,
    val_ratio: float,
    seed: int,
    system_prompt: Optional[str],
    instruction_following: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    raw = _load_raw(input_dir, hf_name, hf_config)
    splits = _build_splits(raw, val_ratio, seed)

    for split_name, ds in splits.items():
        # Map to verl schema; drop every original column so only our keys remain.
        mapped = ds.map(
            lambda ex, idx, split=split_name: _make_record(
                ex,
                idx,
                split=split,
                data_source=data_source,
                prompt_key=prompt_key,
                answer_key=answer_key,
                instruction_following=instruction_following,
                system_prompt=system_prompt,
            ),
            with_indices=True,
            remove_columns=ds.column_names,
        )
        out_path = os.path.join(output_dir, f"{split_name}.parquet")
        mapped.to_parquet(out_path)
        print(f"[gsm8k/{split_name}] wrote {len(mapped)} records to {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_gsm8k_dataset",
        description="Build a GSM8K dataset in verl's parquet schema.",
    )
    p.add_argument("--output-dir", required=True, help="Directory to write parquet splits into.")
    p.add_argument(
        "--input-dir",
        default=None,
        help="Optional local directory containing train.{json,jsonl,csv,parquet} and "
        "test.* files. If omitted, data is pulled from the HF hub.",
    )
    p.add_argument("--hf-name", default="openai/gsm8k", help="HF dataset id (default: openai/gsm8k).")
    p.add_argument("--hf-config", default="main", help="HF dataset config (default: main).")
    p.add_argument(
        "--data-source",
        default="openai/gsm8k",
        help="String stored as the reward_fn_key for each row (default: openai/gsm8k).",
    )
    p.add_argument("--prompt-key", default="question", help="Raw column holding the question.")
    p.add_argument("--answer-key", default="answer", help="Raw column holding the '#### N' answer.")
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of train reserved as validation (default: 0.1).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for val split (default: 42).")
    p.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system message prepended to every prompt.",
    )
    p.add_argument(
        "--instruction-following",
        default=DEFAULT_INSTRUCTION,
        help="Instruction appended to each question (default matches the standard GSM8K rubric).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    convert(
        output_dir=args.output_dir,
        input_dir=args.input_dir,
        hf_name=args.hf_name,
        hf_config=args.hf_config,
        data_source=args.data_source,
        prompt_key=args.prompt_key,
        answer_key=args.answer_key,
        val_ratio=args.val_ratio,
        seed=args.seed,
        system_prompt=args.system_prompt,
        instruction_following=args.instruction_following,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
