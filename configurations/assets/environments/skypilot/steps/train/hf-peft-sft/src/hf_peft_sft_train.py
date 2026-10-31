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
    """Reentrant under DeepSpeed; otherwise only for QLoRA outside multi-process DDP.

    The non-DeepSpeed rule follows peft's examples. Under DeepSpeed, TRL's
    SFTTrainer forces use_reentrant=True for PEFT + ZeRO-3 (and warns), so the
    step picks True there rather than have its value silently overridden.
    """
    return distributed == "deepspeed" or (
        method == "qlora" and not (distributed == "ddp" and world_size > 1)
    )


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


def saved_tensor_count(path: str) -> int:
    """Number of tensors in a safetensors file, from its header (no torch needed)."""
    import json
    import struct

    try:
        with open(path, "rb") as f:
            (length,) = struct.unpack("<Q", f.read(8))
            # safetensors caps the header at 100 MB; a larger length is corruption.
            if length > 100_000_000:
                raise ValueError(f"header length {length} is out of range")
            header = json.loads(f.read(length))
    except (OSError, struct.error, ValueError) as e:
        raise StepError(f"{path} is not a readable safetensors file ({e}).") from e
    if not isinstance(header, dict):
        raise StepError(
            f"{path} is not a readable safetensors file (header is not a map)."
        )
    return sum(1 for k in header if k != "__metadata__")


def check_saved_adapter(path: str) -> None:
    """Refuse an adapter file with no weights, so it is never registered."""
    if saved_tensor_count(path) == 0:
        raise StepError(
            f"the saved adapter {path} has no weights (0 tensors); training ran but "
            "nothing was saved. Nothing registered."
        )


def fail(msg: str) -> None:
    print(f"hf-peft-sft: ERROR {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def _training_args(argv: list[str]):
    from transformers import HfArgumentParser
    from trl import SFTConfig

    parser = HfArgumentParser((StepArgs, SFTConfig))
    step, cfg = parser.parse_args_into_dataclasses(args=argv)
    return step, cfg


def _progress_callback():
    from transformers import TrainerCallback

    class StepLogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero and logs and "loss" in logs:
                print(step_line(state.global_step, state.max_steps, logs), flush=True)

    return StepLogCallback()


def main(argv: list[str] | None = None) -> None:
    import json
    import os

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTTrainer

    step, cfg = _training_args(sys.argv[1:] if argv is None else argv)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if cfg.gradient_checkpointing:
        cfg.gradient_checkpointing_kwargs = {
            "use_reentrant": use_reentrant(step.method, step.distributed, world_size)
        }

    tokenizer = AutoTokenizer.from_pretrained(step.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(step.dataset_path, split=step.dataset_split)
    fmt = detect_format(list(dataset.column_names))
    print(f"hf-peft-sft: {len(dataset)} training rows, format {fmt}", flush=True)

    alora_tokens = None
    if step.method == "alora":
        alora_tokens = tokenizer.encode(
            step.alora_invocation_string, add_special_tokens=False
        )
        if not alora_tokens:
            raise StepError(
                "alora_invocation_string tokenizes to nothing with this tokenizer."
            )

    kw = model_kwargs(step.method, step.distributed, step.mixed_precision, local_rank)
    load = {
        "dtype": getattr(torch, kw["dtype"]),
        "attn_implementation": kw["attn_implementation"],
    }
    if "quantization" in kw:
        q = dict(kw["quantization"])
        for k in ("bnb_4bit_compute_dtype", "bnb_4bit_quant_storage"):
            q[k] = getattr(torch, q[k])
        load["quantization_config"] = BitsAndBytesConfig(**q)
    if "device_map" in kw:
        load["device_map"] = kw["device_map"]
    model = AutoModelForCausalLM.from_pretrained(step.model_name_or_path, **load)

    peft_config = LoraConfig(
        **lora_kwargs(
            step.lora_r,
            step.lora_alpha,
            step.lora_dropout,
            parse_target_modules(step.lora_target_modules),
            step.lora_modules_to_save or None,
            alora_tokens,
        )
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[_progress_callback()],
    )
    if alora_tokens:
        n = min(INVOCATION_CHECK_ROWS, len(trainer.train_dataset))
        rows = [trainer.train_dataset[i]["input_ids"] for i in range(n)]
        missing = check_invocation(rows, alora_tokens)
        if missing:
            print(
                f"hf-peft-sft: WARNING the aLoRA invocation is missing from rows {missing} of "
                f"the first {n} (truncated at max_length?); those rows train nothing.",
                flush=True,
            )

    trainer.train()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(cfg.output_dir)

    if trainer.is_world_process_zero() and step.base_model_id:
        path = os.path.join(cfg.output_dir, "adapter_config.json")
        with open(path, encoding="utf-8") as f:
            adapter_config = json.load(f)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                rewrite_base_model(adapter_config, step.base_model_id), f, indent=2
            )
        print(f"hf-peft-sft: adapter base recorded as {step.base_model_id}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except StepError as e:
        fail(str(e))
