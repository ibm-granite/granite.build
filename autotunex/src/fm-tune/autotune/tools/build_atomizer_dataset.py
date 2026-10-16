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

"""Build an SFT dataset that teaches a model to decompose text into atomic units.

The input is a JSON file holding a top-level list of records, each with:

- ``query``: the conversation turn the response answers.
- ``response``: the text to be decomposed.
- ``atoms``: the list of atomic units (gold decomposition) of that text.

Each record becomes one SFT example whose prompt instructs the model to
decompose the response into atomic units and whose target is the atoms, one per
line. The prompt itself comes from ``autotune.tools._chat_utils`` so the training
and eval prompts can never drift. Two output shapes (matching
``build_factuality_dataset.py``):

- ``formatted``: ``input`` is a string rendered through the model's chat
  template; ``output`` is the newline-joined atoms.
- ``chat``: ``input`` is the raw list of chat messages; ``output`` is the
  newline-joined atoms.

Usage:

    python -m autotune.tools.build_atomizer_dataset \
        --input /path/to/atoms_raw.json \
        --output-dir /path/to/out \
        --model ibm-granite/granite-4.1-3b

``--format`` defaults to ``both``, which writes one file per shape into
``--output-dir``, naming each from the input stem and the format:
``atoms_raw.json`` yields ``atoms_raw_formatted.jsonl`` and
``atoms_raw_chat.jsonl``. The ``--model`` argument is required when rendering the chat
template (``formatted``/``both``) or whenever the output-length gate is active
(it tokenizes the target). The output schema matches ``docs/dataset-sft.md``.

Records whose output sequence (the atoms target) tokenizes to at least
``--max-output-length`` tokens (default 1024) are dropped, in both formats,
matching the length gate in ``build_factuality_dataset.py``. Set
``--max-output-length 0`` to disable the gate (which also lets ``--format chat``
run without a model again).

Granite 4.2 moved to ChatML and defaults to ``enable_thinking=True``, which would
end the prompt inside an open ``<think>`` block and train the atoms as reasoning
content. Thinking is therefore disabled unless ``--thinking`` is passed; see
``autotune/tools/_chat_utils.py``.

For ``--format formatted`` (and only then, since it is the only mode with a
real tokenizer) the tool also prints token-length stats — count/min/max/mean
and p50/p90/p95/p99 — for the input sequence, the output sequence, and the two
combined, to help size ``max_seq_length``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from ._chat_utils import (
    IdentityTokenizer,
    get_atomizer_messages,
    load_tokenizer,
    num_tokens,
    output_name_for,
    print_token_stats,
    render_chat_input,
    resolve_formats,
    summarize,
)


def _format_atoms(atoms: List[str]) -> str:
    """Join atoms one-per-line, stripping each and dropping empties."""
    cleaned = [a.strip() for a in atoms if isinstance(a, str) and a.strip()]
    return "\n".join(cleaned)


def _make_record(
    dp: Dict[str, Any],
    tokenizer,
    fmt: str,
    thinking: bool = False,
) -> Optional[Dict[str, Any]]:
    """Convert one raw record to an SFT example, or None if malformed/empty."""
    query = dp.get("query")
    response = dp.get("response")
    atoms = dp.get("atoms")
    if not isinstance(query, str) or not query.strip():
        return None
    if not isinstance(response, str) or not response.strip():
        return None
    if not isinstance(atoms, list):
        return None

    output_seq = _format_atoms(atoms)
    if not output_seq:
        return None

    messages = get_atomizer_messages(response, query)
    if fmt == "formatted":
        return {
            "input": render_chat_input(messages, tokenizer, thinking=thinking),
            "output": output_seq,
        }
    return {"input": messages, "output": output_seq}


def convert_split(
    input_path: str,
    output_path: str,
    fmt: str,
    tokenizer,
    max_output_length: int = 1024,
    thinking: bool = False,
) -> None:
    """Convert a single input JSON file to the target JSONL.

    Records whose output sequence tokenizes to ``max_output_length`` tokens or
    more are dropped (gate disabled when ``max_output_length <= 0``), matching
    the length gate in ``build_factuality_dataset.py``.

    For ``--format formatted`` a real tokenizer is available, so token-length
    stats are collected over the rendered input, the output, and their sum and
    printed after the file is written. ``--format chat`` keeps raw messages and
    renders no template, so no stats are produced.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"{input_path}: expected a top-level JSON list")

    collect_stats = fmt == "formatted"
    # The chat path may carry the null-object tokenizer, which cannot count tokens.
    gate_on = max_output_length > 0 and not isinstance(tokenizer, IdentityTokenizer)
    input_lengths: List[int] = []
    output_lengths: List[int] = []
    combined_lengths: List[int] = []

    written = 0
    skipped_malformed = 0
    skipped_length = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for dp in records:
            record = _make_record(dp, tokenizer, fmt, thinking) if isinstance(dp, dict) else None
            if record is None:
                skipped_malformed += 1
                continue

            # Count output tokens once; reused for both the gate and the stats.
            n_out = num_tokens(tokenizer, record["output"]) if (gate_on or collect_stats) else 0
            if gate_on and n_out >= max_output_length:
                skipped_length += 1
                continue

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if collect_stats:
                n_in = num_tokens(tokenizer, record["input"])
                input_lengths.append(n_in)
                output_lengths.append(n_out)
                combined_lengths.append(n_in + n_out)

    print(
        f"[atomizer/{fmt}] wrote {written} records to {output_path} "
        f"(skipped {skipped_malformed} malformed, {skipped_length} over-length)"
    )

    if collect_stats:
        print_token_stats(
            [
                summarize("input", input_lengths),
                summarize("output", output_lengths),
                summarize("input+output", combined_lengths),
            ],
            f"atomizer/{fmt}",
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_atomizer_dataset",
        description="Build an SFT dataset for decomposing text into atomic units.",
    )
    p.add_argument("--input", required=True, help="Input JSON file (top-level list of {response, atoms} records).")
    p.add_argument("--output-dir", required=True, help="Directory to write the JSONL output into.")
    p.add_argument(
        "--format",
        dest="fmt",
        default="both",
        choices=["formatted", "chat", "both"],
        help="'formatted' renders input via the chat template; 'chat' keeps the raw message list; "
        "'both' writes one file of each (default: both).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Model name or path for the tokenizer / chat template. Required for --format "
        "formatted/both or when --max-output-length > 0.",
    )
    p.add_argument(
        "--max-output-length",
        type=int,
        default=1024,
        help="Drop records whose output (atoms) tokenizes to this many tokens or more. "
        "Set to 0 to disable the gate (default: 1024).",
    )
    p.add_argument(
        "--thinking",
        action="store_true",
        help="Enable the template's thinking mode (Granite 4.2). Off by default so the "
        "atoms are not trained inside a <think> block.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not os.path.exists(args.input):
        print(f"[error] {args.input} not found", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    formats = resolve_formats(args.fmt)
    # A real tokenizer is needed to render the chat template and to count output
    # tokens for the length gate.
    needs_tokenizer = "formatted" in formats or args.max_output_length > 0
    tokenizer = load_tokenizer(
        args.model,
        needs_tokenizer,
        "for --format formatted/both or when --max-output-length > 0 "
        "(used for chat template rendering and/or output-length gating)",
    )

    for fmt in formats:
        convert_split(
            input_path=args.input,
            output_path=os.path.join(args.output_dir, output_name_for(args.input, fmt)),
            fmt=fmt,
            tokenizer=tokenizer,
            max_output_length=args.max_output_length,
            thinking=args.thinking,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
