#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""LoRA / QLoRA / aLoRA supervised fine-tuning with HF PEFT + TRL SFTTrainer.

Launched by hf_peft_sft_run.sh under `accelerate launch`. Step-owned flags parse
into StepArgs; every other flag goes to TRL's SFTConfig unchanged. Saves the
adapter only (adapter_config.json + adapter_model.safetensors + tokenizer).

This module imports only the standard library at top level so its helpers are
unit-testable without torch; main() imports the training stack.
"""

from __future__ import annotations

import dataclasses
import sys

METHODS = ("lora", "qlora", "alora")
DISTRIBUTED = ("ddp", "fsdp", "deepspeed")
PRECISIONS = ("bf16", "fp16", "no")
_DTYPE = {"bf16": "bfloat16", "fp16": "float16", "no": "float32"}
INVOCATION_CHECK_ROWS = 32


class StepError(Exception):
    """A user-facing configuration or data error; printed as hf-peft-sft: ERROR."""


@dataclasses.dataclass
class StepArgs:
    model_name_or_path: str = dataclasses.field(
        metadata={"help": "staged base model dir"}
    )
    method: str = dataclasses.field(default="lora", metadata={"choices": METHODS})
    distributed: str = dataclasses.field(
        default="ddp", metadata={"choices": DISTRIBUTED}
    )
    mixed_precision: str = dataclasses.field(
        default="bf16", metadata={"choices": PRECISIONS}
    )
    dataset_path: str = dataclasses.field(
        default="", metadata={"help": "staged dir or HF id"}
    )
    dataset_split: str = "train"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = dataclasses.field(
        default_factory=lambda: ["all-linear"]
    )
    lora_modules_to_save: list[str] = dataclasses.field(default_factory=list)
    alora_invocation_string: str = ""
    base_model_id: str = ""


STEP_ARG_FIELDS = tuple(f.name for f in dataclasses.fields(StepArgs))


def parse_target_modules(values: list[str]) -> str | list[str]:
    """`all-linear` stays a string (peft's sentinel); anything else is a list."""
    if not values:
        raise StepError("lora.target_modules is empty.")
    if values == ["all-linear"]:
        return "all-linear"
    return list(values)


def detect_format(columns: list[str]) -> str:
    """Which TRL SFT dataset format the columns are."""
    cols = set(columns)
    if "messages" in cols:
        return "messages"
    if {"prompt", "completion"} <= cols:
        return "prompt_completion"
    if "text" in cols:
        return "text"
    raise StepError(
        f"dataset columns {sorted(cols)} match no TRL SFT format; expected a "
        "'messages' column (conversational), 'prompt' + 'completion' columns, or a "
        "'text' column. Reshape the data in an upstream step."
    )


def contains_subsequence(seq: list[int], sub: list[int]) -> bool:
    n = len(sub)
    return n > 0 and any(seq[i : i + n] == sub for i in range(len(seq) - n + 1))


def check_invocation(rows: list[list[int]], tokens: list[int]) -> list[int]:
    """Indices of rows without the aLoRA invocation; error if none has it.

    peft activates an aLoRA adapter only after the invocation tokens; a row
    without them trains nothing (zero gradient), silently.
    """
    if not tokens:
        raise StepError(
            "alora_invocation_string tokenizes to nothing with this tokenizer."
        )
    missing = [
        i for i, row in enumerate(rows) if not contains_subsequence(list(row), tokens)
    ]
    if rows and len(missing) == len(rows):
        raise StepError(
            f"the aLoRA invocation tokens {tokens} never appears in the first "
            f"{len(rows)} tokenized training rows, so the adapter would never "
            "activate. Use the exact string the chat template emits before the "
            "assistant turn (e.g. '<|im_start|>assistant' for ChatML)."
        )
    return missing


def use_reentrant(method: str, distributed: str, world_size: int) -> bool:
    """peft's examples: reentrant only for QLoRA outside multi-process DDP."""
    return method == "qlora" and not (distributed == "ddp" and world_size > 1)


def lora_kwargs(
    r: int,
    alpha: int,
    dropout: float,
    target_modules: str | list[str],
    modules_to_save: list[str] | None,
    alora_tokens: list[int] | None,
) -> dict:
    kw = {
        "task_type": "CAUSAL_LM",
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "bias": "none",
        "target_modules": target_modules,
    }
    if modules_to_save:
        kw["modules_to_save"] = list(modules_to_save)
    if alora_tokens:
        kw["alora_invocation_tokens"] = list(alora_tokens)
    return kw


def bnb_kwargs(mixed_precision: str, sharded: bool) -> dict:
    """4-bit nf4 + double quant. Sharded storage must be a float dtype."""
    compute = "float32" if mixed_precision == "no" else "bfloat16"
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": compute,
        "bnb_4bit_quant_storage": "bfloat16" if sharded else "uint8",
    }


def model_kwargs(
    method: str, distributed: str, mixed_precision: str, local_rank: int
) -> dict:
    """from_pretrained kwargs with dtypes as strings (main() maps them to torch)."""
    kw: dict = {"dtype": _DTYPE[mixed_precision], "attn_implementation": "sdpa"}
    if method == "qlora":
        sharded = distributed != "ddp"
        kw["quantization"] = bnb_kwargs(mixed_precision, sharded)
        if sharded:
            kw["dtype"] = "bfloat16"
        else:
            kw["device_map"] = {"": local_rank}
    return kw


def step_line(step: int, max_steps: int, logs: dict) -> str:
    """The progress line the shared monitor's `Step:\\s\\d+` rule picks up."""
    head = (
        f"Step: {step}/{max_steps}" if max_steps and max_steps > 0 else f"Step: {step}"
    )
    loss = logs.get("loss")
    return f"{head} loss={loss:.4f}" if isinstance(loss, (int, float)) else head


def rewrite_base_model(adapter_config: dict, base_model_id: str) -> dict:
    """Record the portable base id instead of the node-local staging path."""
    if not base_model_id:
        return adapter_config
    return {**adapter_config, "base_model_name_or_path": base_model_id}


def fail(msg: str) -> None:
    print(f"hf-peft-sft: ERROR {msg}", file=sys.stderr, flush=True)
    sys.exit(1)
