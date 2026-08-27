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

"""Build factuality intrinsics datasets (detection / correction) from raw ELI5 splits.

Reads pre-split raw JSON files (``{name}_raw_{split}.json``) and emits JSONL in
one of two shapes:

- ``formatted``: ``input`` is a string rendered through the model's chat
  template; ``output`` is the target JSON string.
- ``chat``: ``input`` is a list of chat messages, ``documents`` is the separate
  documents column, and ``output`` is the target JSON string.

Usage:

    python -m autotune.tools.build_factuality_dataset \
        --input-dir /path/to/raw \
        --output-dir /path/to/out \
        --task detection \
        --format chat

The ``--model`` argument is required when ``--format formatted`` (needed for the
chat template). For ``--format chat`` messages are written raw.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

CRITERIA = {
    "factuality": (
        "A factually incorrect response occurs when the assistant's message contains "
        "one or more factual claims that are unsupported by, inconsistent with, or "
        "directly contradicted by the information provided in the documents or context. "
        "This includes situations where the assistant: introduces details not grounded "
        "in the context, misstates or distorts facts contained within the context, "
        "misinterprets the meaning or implications of the context, supplies erroneous "
        "or conflicting information relative to the context. Even if only a small "
        "portion of the response contains such inaccuracies, the overall message is "
        "considered factually incorrect."
    ),
}

_SYSTEM_PROMPT = (
    "As a judge agent, your role is to help assess whether the provided text meets "
    "the given judging criteria, utilizing all available information, including "
    "conversations, documents, and tools."
)

CRITERIA_ID = "factuality"


def _guardian_block(is_detection: bool, for_prompt: bool) -> str:
    """Build the ``<guardian>`` user-turn content.

    ``is_detection`` toggles between the yes/no detection schema and the
    correction schema. ``for_prompt`` appends the JSON dict schema hint that
    the prompt-style variants use.
    """
    criteria_text = CRITERIA[CRITERIA_ID]
    if is_detection:
        scoring = "If the last assistant's text meets the criteria, return 'yes'; otherwise, return 'no'."
        if for_prompt:
            scoring += ' Provide the final answer as a JSON dict with the following format: {"score": "yes" or "no"}.'
    else:
        scoring = (
            "If the last assistant's text meets the criteria, "
            "return a corrected version of the assistant's message based on the "
            "given context; otherwise, return 'none'."
        )
        if for_prompt:
            scoring += (
                " Provide the final answer as a JSON dict with the following "
                'format: {"correction": "corrected message" or "none"}.'
            )
    return f"<guardian>{_SYSTEM_PROMPT}\n\n### Criteria: {criteria_text}\n\n### Scoring Schema: {scoring}"


def _unique_strings(strings: List[str]) -> List[str]:
    seen = set()
    result = []
    for s in strings:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def _extract_contexts(record: Dict[str, Any]) -> List[str]:
    """Pull c_a* context passages out of a raw record and deduplicate."""
    contexts = []
    for k, v in record.items():
        if k.startswith("c_a") and isinstance(v, dict) and v.get("text") is not None:
            contexts.append(v["text"])
    return _unique_strings(contexts)


def _build_messages(query: str, response: str, is_detection: bool, for_prompt: bool) -> List[Dict[str, str]]:
    return [
        {"role": "user", "content": query},
        {"role": "assistant", "content": response},
        {"role": "user", "content": _guardian_block(is_detection, for_prompt)},
    ]


def _render_input(messages: List[Dict[str, str]], documents: List[Dict[str, str]], tokenizer) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        documents=documents,
    )


def _record_detection(
    dp: Dict[str, Any],
    tokenizer,
    fmt: str,
    for_prompt: bool,
) -> Optional[Dict[str, Any]]:
    query = dp["query"]
    response = dp["response"]["text"]
    label = dp["response"]["label"]

    contexts = _extract_contexts(dp)
    documents = [{"doc_id": "0", "text": "\n\n".join(contexts)}]
    messages = _build_messages(query, response, is_detection=True, for_prompt=for_prompt)
    output_seq = json.dumps({"score": label.lower()})

    if fmt == "formatted":
        return {"input": _render_input(messages, documents, tokenizer), "output": output_seq}
    return {"input": messages, "output": output_seq, "documents": documents}


def _record_correction(
    dp: Dict[str, Any],
    tokenizer,
    fmt: str,
    for_prompt: bool,
    max_length: int,
    include_meta: bool,
) -> Optional[Dict[str, Any]]:
    query = dp["query"]
    response = dp["response"]["text"]
    label = dp["response"]["label"]
    correction = dp["correction"]["text"] if "correction" in dp else "none"

    contexts = _extract_contexts(dp)
    documents = [{"doc_id": "0", "text": "\n\n".join(contexts)}]
    messages = _build_messages(query, response, is_detection=False, for_prompt=for_prompt)
    output_seq = json.dumps({"correction": correction})

    # Length gate the target — mirrors the scratchpad behavior.
    num_tokens = len(tokenizer(output_seq)["input_ids"])
    if num_tokens >= max_length:
        return None

    if fmt == "formatted":
        record: Dict[str, Any] = {
            "input": _render_input(messages, documents, tokenizer),
            "output": output_seq,
        }
        if include_meta:
            record.update({"query": query, "response": response, "label": label})
        return record

    record = {"input": messages, "output": output_seq, "documents": documents}
    if include_meta:
        record.update({"query": query, "response": response})
    return record


def convert_split(
    input_path: str,
    output_path: str,
    task: str,
    fmt: str,
    tokenizer,
    *,
    split: str,
    for_prompt: bool = False,
    max_correction_length: int = 1024,
) -> None:
    """Convert a single raw split file to the target JSONL."""
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"{input_path}: expected a top-level JSON list")

    # Only emit query/response/label metadata on the test split, matching the
    # scratchpad convention so eval can recover the originals.
    include_meta = split == "test"

    written = 0
    skipped_length = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for dp in records:
            if not isinstance(dp, dict):
                continue
            if task == "detection":
                record = _record_detection(dp, tokenizer, fmt, for_prompt)
            else:
                record = _record_correction(
                    dp,
                    tokenizer,
                    fmt,
                    for_prompt,
                    max_correction_length,
                    include_meta,
                )
                if record is None:
                    skipped_length += 1
                    continue
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    msg = f"[{task}/{fmt}/{split}] wrote {written} records to {output_path}"
    if task == "correction":
        msg += f" (skipped {skipped_length} over-length)"
    print(msg)


class _IdentityTokenizer:
    """Stand-in tokenizer for --format chat, which never renders a template."""

    def __call__(self, text: str):
        raise RuntimeError("chat-format conversion should not tokenize")

    def apply_chat_template(self, *args, **kwargs):
        raise RuntimeError("chat-format conversion should not render a template")


def _load_tokenizer(model: Optional[str], fmt: str, task: str):
    # formatted needs a real template; correction length-gates via tokenization.
    needs_tokenizer = fmt == "formatted" or task == "correction"
    if not needs_tokenizer:
        return _IdentityTokenizer()
    if not model:
        raise SystemExit(
            "--model is required for --format formatted or --task correction "
            "(used for chat template rendering and/or output length gating)"
        )
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_factuality_dataset",
        description="Convert raw ELI5 factuality splits into training-ready JSONL.",
    )
    p.add_argument("--input-dir", required=True, help="Directory containing raw split files.")
    p.add_argument("--output-dir", required=True, help="Directory to write JSONL splits into.")
    p.add_argument(
        "--task",
        required=True,
        choices=["detection", "correction"],
        help="Factuality task variant.",
    )
    p.add_argument(
        "--format",
        dest="fmt",
        required=True,
        choices=["formatted", "chat"],
        help="'formatted' renders input via the chat template; 'chat' keeps messages + documents.",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Model name or path for tokenizer / chat template. "
        "Required for --format formatted and for --task correction.",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Which splits to process (default: train val test).",
    )
    p.add_argument(
        "--max-correction-length",
        type=int,
        default=1024,
        help="Skip correction records whose tokenized target is >= this length.",
    )
    p.add_argument(
        "--for-prompt",
        action="store_true",
        help="Append the JSON-dict schema hint to the guardian block.",
    )
    p.add_argument(
        "--dataset-name",
        default="eli5",
        help="Basename for input and output files (default: eli5). Use `bio` for Biographies.",
    )
    p.add_argument(
        "--input-pattern",
        default="{name}_raw_{split}.json",
        help="Input filename pattern; {name} and {split} are substituted.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = _load_tokenizer(args.model, args.fmt, args.task)

    for split in args.splits:
        input_name = args.input_pattern.format(name=args.dataset_name, split=split)
        input_path = os.path.join(args.input_dir, input_name)
        if not os.path.exists(input_path):
            print(f"[skip] {input_path} not found", file=sys.stderr)
            continue
        output_name = f"{args.dataset_name}_{args.task}_{args.fmt}_{split}.jsonl"
        output_path = os.path.join(args.output_dir, output_name)
        convert_split(
            input_path=input_path,
            output_path=output_path,
            task=args.task,
            fmt=args.fmt,
            tokenizer=tokenizer,
            split=split,
            for_prompt=args.for_prompt,
            max_correction_length=args.max_correction_length,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
