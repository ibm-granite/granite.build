#!/usr/bin/env python3

# Copyright Granite.Build Authors
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

# SPDX-License-Identifier: Apache-2.0
"""LoRA / QLoRA / aLoRA supervised fine-tuning with HF PEFT + TRL SFTTrainer.

Launched by hf_peft_sft_run.sh under `accelerate launch`. Step-owned flags parse
into StepArgs; every other flag goes to TRL's SFTConfig unchanged. Saves the
adapter only (adapter_config.json + adapter_model.safetensors + tokenizer).

This module imports only the standard library at top level so its helpers are
unit-testable without torch; the private helpers main() calls import the
training stack.
"""

from __future__ import annotations

import dataclasses
import json
import os
import struct
import sys
from typing import Any

METHODS = ("lora", "qlora", "alora")
DISTRIBUTED = ("ddp", "fsdp", "deepspeed")
PRECISIONS = ("bf16", "fp16", "no")
_DTYPE = {"bf16": "bfloat16", "fp16": "float16", "no": "float32"}
INVOCATION_CHECK_ROWS = 32


class StepError(Exception):
    """A user-facing configuration or data error; printed as hf-peft-sft: ERROR."""


@dataclasses.dataclass
class StepArgs:  # pylint: disable=too-many-instance-attributes
    """The step-owned trainer flags; HfArgumentParser parses them beside SFTConfig.

    Every field is a ``--<name>`` flag. No name may also be an SFTConfig field.
    """

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
    dataset_config: str = ""
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
    """`all-linear` stays a string (peft's sentinel); anything else is a list.

    :param values: The lora.target_modules values.
    :returns: ``"all-linear"``, or the module names as a list.
    :raises StepError: When the list is empty.
    """
    if not values:
        raise StepError("lora.target_modules is empty.")
    if values == ["all-linear"]:
        return "all-linear"
    return list(values)


def detect_format(columns: list[str]) -> str:
    """Which TRL SFT dataset format the columns are.

    :param columns: The dataset's column names.
    :returns: ``"messages"``, ``"prompt_completion"`` or ``"text"``.
    :raises StepError: When the columns match no TRL SFT format.
    """
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


def _saved_split_name(split: str) -> str:
    """The one split name a single saved Dataset is loaded under.

    :param split: The dataset_split spec, e.g. ``train[:1000]``.
    :returns: The split name, e.g. ``train``.
    :raises StepError: When the spec has whitespace (datasets' arrow builder
        refuses it with a raw ValueError), has no split name, or names more
        than one split (a saved Dataset has only one).
    """
    if any(c.isspace() for c in split):
        raise StepError(
            f"dataset_split {split!r} has whitespace; remove it, e.g. "
            "'train[:1000]' or 'train[:10]+train[-10:]'."
        )
    names = {sub.split("[", 1)[0] for sub in split.split("+")}
    if "" in names:
        raise StepError(
            f"dataset_split {split!r} has no split name; give one before any "
            "slice, e.g. 'train[:1000]'."
        )
    if len(names) > 1:
        raise StepError(
            f"dataset_split {split!r} names splits {sorted(names)}, but a "
            "save_to_disk Dataset has only one split; use one name, e.g. 'train'."
        )
    return names.pop()


def dataset_load_spec(path: str, split: str, config: str) -> tuple[str, dict]:
    """Return the (builder_or_path, kwargs) for datasets.load_dataset.

    A single-Dataset save_to_disk dir (state.json, no dataset_dict.json) is
    refused by load_dataset, so it loads through the `arrow` builder from the
    files state.json lists. Not load_from_disk: that cannot apply split slicing
    (train[:1000]), which dataset_split promises for either source. A saved
    Dataset has no configs, so `config` is ignored there. Everything else (HF
    ids, hub-style dirs, DatasetDict save_to_disk dirs) goes to load_dataset as is.

    :param path: The staged dataset dir or an HF dataset id.
    :param split: The split spec, e.g. ``train`` or ``train[:1000]``.
    :param config: The dataset config name, or ``""`` for none.
    :returns: The first positional argument and the keyword arguments for
        load_dataset.
    :raises StepError: When a save_to_disk dir has a malformed state.json, a
        listed data file is missing, or the split has no single split name.
    """
    state = os.path.join(path, "state.json")
    if not (
        os.path.isdir(path)
        and os.path.exists(state)
        and not os.path.exists(os.path.join(path, "dataset_dict.json"))
    ):
        return path, {"split": split, **({"name": config} if config else {})}
    try:
        with open(state, encoding="utf-8") as f:
            entries = json.load(f).get("_data_files")
    except (OSError, ValueError, AttributeError) as e:
        raise StepError(
            f"{state} is not a readable save_to_disk state file ({e})."
        ) from e
    if not entries or not isinstance(entries, list):
        raise StepError(
            f"{state} lists no _data_files; the save_to_disk dataset is incomplete."
        )
    files = []
    for entry in entries:
        name = entry.get("filename") if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name:
            raise StepError(
                f"{state} has a malformed _data_files entry {entry!r}; each entry "
                "must be a map with a 'filename' string."
            )
        full = os.path.join(os.path.abspath(path), name)
        if not os.path.isfile(full):
            raise StepError(
                f"{state} lists data file {name!r}, which is missing from {path}."
            )
        files.append(full)
    return "arrow", {"data_files": {_saved_split_name(split): files}, "split": split}


def contains_subsequence(seq: list[int], sub: list[int]) -> bool:
    """Whether `sub` occurs contiguously in `seq`; an empty `sub` never does.

    :param seq: The token ids to search.
    :param sub: The token ids to look for.
    :returns: True when `sub` is non-empty and occurs in `seq`.
    """
    n = len(sub)
    return n > 0 and any(seq[i : i + n] == sub for i in range(len(seq) - n + 1))


def check_invocation(rows: list[list[int]], tokens: list[int]) -> list[int]:
    """Indices of rows without the aLoRA invocation; error if none has it.

    peft activates an aLoRA adapter only after the invocation tokens; a row
    without them trains nothing (zero gradient), silently. An empty `tokens`
    is refused earlier, by _alora_tokens().

    :param rows: The first tokenized training rows (input_ids).
    :param tokens: The non-empty aLoRA invocation token ids.
    :returns: The indices of the rows that lack the invocation.
    :raises StepError: When there are rows and none has the invocation.
    """
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

    :param method: The method: lora, qlora or alora.
    :param distributed: The backend: ddp, fsdp or deepspeed.
    :param world_size: The total number of training processes.
    :returns: The gradient checkpointing ``use_reentrant`` value.
    """
    return distributed == "deepspeed" or (
        method == "qlora" and not (distributed == "ddp" and world_size > 1)
    )


def lora_kwargs(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    r: int,
    alpha: int,
    dropout: float,
    target_modules: str | list[str],
    modules_to_save: list[str] | None,
    alora_tokens: list[int] | None,
) -> dict:
    """The peft.LoraConfig kwargs for a causal-LM adapter with no bias training.

    :param r: The LoRA rank.
    :param alpha: The LoRA alpha.
    :param dropout: The LoRA dropout.
    :param target_modules: ``"all-linear"`` or the module names to adapt.
    :param modules_to_save: Modules to train fully and save, or None.
    :param alora_tokens: The aLoRA invocation tokens, or None for LoRA/QLoRA.
    :returns: The LoraConfig keyword arguments.
    """
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
    """4-bit nf4 + double quant. Sharded storage must be a float dtype.

    :param mixed_precision: The mixed precision: bf16, fp16 or no.
    :param sharded: Whether FSDP or DeepSpeed shards the quantized weights.
    :returns: The BitsAndBytesConfig kwargs, with dtypes as strings.
    """
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
    """from_pretrained kwargs with dtypes as strings (_load_model() maps them to torch).

    :param method: The method: lora, qlora or alora.
    :param distributed: The backend: ddp, fsdp or deepspeed.
    :param mixed_precision: The mixed precision: bf16, fp16 or no.
    :param local_rank: This process's rank on its node.
    :returns: ``dtype`` and ``attn_implementation``; for QLoRA also
        ``quantization``, plus ``device_map`` under DDP or a bf16 ``dtype``
        when sharded.
    """
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
    """The progress line the shared monitor's `Step:\\s\\d+` rule picks up.

    :param step: The global step.
    :param max_steps: The total steps; left out of the line when not positive.
    :param logs: The logged metrics; a numeric ``loss`` is appended.
    :returns: The progress line.
    """
    head = (
        f"Step: {step}/{max_steps}" if max_steps and max_steps > 0 else f"Step: {step}"
    )
    loss = logs.get("loss")
    return f"{head} loss={loss:.4f}" if isinstance(loss, (int, float)) else head


def rewrite_base_model(adapter_config: dict, base_model_id: str) -> dict:
    """Record the portable base id instead of the node-local staging path.

    :param adapter_config: The saved adapter_config.json contents.
    :param base_model_id: The portable base model id, or ``""`` to keep the path.
    :returns: The adapter config, with base_model_name_or_path replaced when
        `base_model_id` is set.
    """
    if not base_model_id:
        return adapter_config
    return {**adapter_config, "base_model_name_or_path": base_model_id}


def saved_tensor_count(path: str) -> int:
    """Number of tensors in a safetensors file, from its header (no torch needed).

    :param path: The safetensors file.
    :returns: The number of tensors, not counting ``__metadata__``.
    :raises StepError: When the file is unreadable or its header is malformed.
    """
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
    """Refuse an adapter file with no weights, so it is never registered.

    :param path: The saved adapter_model.safetensors.
    :raises StepError: When the file is unreadable or holds no tensors.
    """
    if saved_tensor_count(path) == 0:
        raise StepError(
            f"the saved adapter {path} has no weights (0 tensors); training ran but "
            "nothing was saved. Nothing registered."
        )


def fail(msg: str) -> None:
    """Print a user-facing error with the step's prefix and exit with status 1.

    :param msg: The error message, without the ``hf-peft-sft: ERROR`` prefix.
    """
    print(f"hf-peft-sft: ERROR {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def _training_args(argv: list[str]) -> tuple[StepArgs, Any]:
    """Parse the step's own flags into StepArgs and every other flag into SFTConfig.

    allow_abbrev=False: argparse's default takes a unique prefix as the full
    flag, so ``--output_d`` would silently set the step-owned output_dir. The
    words the parser does not use are what parse_args_into_dataclasses would
    raise its "not used by the HfArgumentParser" ValueError for; they are
    reported here as a StepError naming the flags (not their values).

    :param argv: The trainer's argv, without the program name.
    :returns: The parsed StepArgs and trl.SFTConfig.
    :raises StepError: When argv has flags (or words) the parser does not use,
        such as an abbreviation or an unknown name.
    """
    from transformers import HfArgumentParser
    from trl import SFTConfig

    parser = HfArgumentParser((StepArgs, SFTConfig), allow_abbrev=False)
    step, cfg, unused = parser.parse_args_into_dataclasses(
        args=argv, return_remaining_strings=True
    )
    if unused:
        # Flags only: a value (even a negative number) never reaches the message.
        flags = [
            w.split("=", 1)[0]
            for w in unused
            if w.startswith("--") or (w.startswith("-") and w[1:2].isalpha())
        ]
        what = (
            f"unknown or abbreviated trainer flag(s) {flags}"
            if flags
            else f"{len(unused)} stray trainer argument(s) with no flag"
        )
        raise StepError(
            f"{what}; every hf_peft_sft_config.args key must be an exact SFTConfig "
            "field name (no abbreviations)."
        )
    return step, cfg


def _progress_callback() -> Any:
    """Build the TrainerCallback that prints the progress line on world process zero.

    :returns: A transformers.TrainerCallback instance.
    """
    from transformers import TrainerCallback

    class StepLogCallback(TrainerCallback):  # pylint: disable=too-few-public-methods
        """Print step_line() for every log event that carries a loss."""

        # pylint: disable-next=unused-argument
        def on_log(self, args, state, control, logs=None, **kwargs):
            """Print the progress line on world process zero.

            :param args: The training arguments (unused).
            :param state: The trainer state; gives the step, max steps and rank.
            :param control: The trainer control (unused).
            :param logs: The logged metrics, if any.
            """
            if state.is_world_process_zero and logs and "loss" in logs:
                print(step_line(state.global_step, state.max_steps, logs), flush=True)

    return StepLogCallback()


def _load_tokenizer(step: StepArgs) -> Any:
    """Load the base model's tokenizer; pad with EOS when it has no pad token.

    :param step: The step arguments; gives the staged base model dir.
    :returns: The transformers tokenizer.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(step.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_dataset(step: StepArgs) -> Any:
    """Load the training split, check its format, and print the row count.

    :param step: The step arguments; gives the dataset path, split and config.
    :returns: The datasets.Dataset to train on.
    :raises StepError: When the dataset dir is malformed or the columns match
        no TRL SFT format.
    """
    from datasets import load_dataset

    builder, load_kw = dataset_load_spec(
        step.dataset_path, step.dataset_split, step.dataset_config
    )
    dataset = load_dataset(builder, **load_kw)
    fmt = detect_format(list(dataset.column_names))
    print(f"hf-peft-sft: {len(dataset)} training rows, format {fmt}", flush=True)
    return dataset


def _alora_tokens(step: StepArgs, tokenizer: Any) -> list[int] | None:
    """Encode the aLoRA invocation string, for the alora method only.

    :param step: The step arguments; gives the method and invocation string.
    :param tokenizer: The tokenizer; only its ``encode`` method is used.
    :returns: The invocation token ids, or None when the method is not alora.
    :raises StepError: When the invocation string encodes to no tokens.
    """
    if step.method != "alora":
        return None
    tokens = tokenizer.encode(step.alora_invocation_string, add_special_tokens=False)
    if not tokens:
        raise StepError(
            "alora_invocation_string tokenizes to nothing with this tokenizer."
        )
    return tokens


def _load_model(step: StepArgs, local_rank: int) -> Any:
    """Load the base model with the dtype, attention and quantization the method needs.

    :param step: The step arguments; gives the model dir, method, distributed
        backend and mixed precision.
    :param local_rank: This process's rank on its node; QLoRA under DDP pins
        the model to that device.
    :returns: The transformers causal LM.
    """
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

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
    return AutoModelForCausalLM.from_pretrained(step.model_name_or_path, **load)


def _build_trainer(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    step: StepArgs,
    cfg: Any,
    model: Any,
    tokenizer: Any,
    dataset: Any,
    alora_tokens: list[int] | None,
) -> Any:
    """Build the LoRA config and the TRL SFTTrainer that applies it.

    :param step: The step arguments; gives the LoRA hyperparameters.
    :param cfg: The trl.SFTConfig.
    :param model: The base model.
    :param tokenizer: The tokenizer, passed to the trainer as processing class.
    :param dataset: The training dataset.
    :param alora_tokens: The aLoRA invocation tokens, or None for LoRA/QLoRA.
    :returns: The trl.SFTTrainer.
    :raises StepError: When lora.target_modules is empty.
    """
    from peft import LoraConfig
    from trl import SFTTrainer

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
    return SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[_progress_callback()],
    )


def _check_alora(trainer: Any, alora_tokens: list[int] | None) -> None:
    """Warn about the first tokenized rows that lack the aLoRA invocation.

    Does nothing when there are no invocation tokens (LoRA/QLoRA).

    :param trainer: The SFTTrainer; its train_dataset is already tokenized.
    :param alora_tokens: The aLoRA invocation tokens, or None.
    :raises StepError: When none of the checked rows has the invocation.
    """
    if not alora_tokens:
        return
    n = min(INVOCATION_CHECK_ROWS, len(trainer.train_dataset))
    rows = [trainer.train_dataset[i]["input_ids"] for i in range(n)]
    missing = check_invocation(rows, alora_tokens)
    if missing:
        print(
            f"hf-peft-sft: WARNING the aLoRA invocation is missing from rows {missing} of "
            f"the first {n} (truncated at max_length?); those rows train nothing.",
            flush=True,
        )


def _save_adapter(trainer: Any, step: StepArgs, cfg: Any) -> None:
    """Save the adapter, then record the portable base model id in its config.

    Under FSDP the state dict is gathered in full first. Only world process
    zero rewrites adapter_config.json, and only when base_model_id is set.

    :param trainer: The trained SFTTrainer.
    :param step: The step arguments; gives base_model_id.
    :param cfg: The trl.SFTConfig; gives output_dir.
    """
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


def main(argv: list[str] | None = None) -> None:
    """Train and save a LoRA, QLoRA or aLoRA adapter.

    :param argv: The trainer's argv without the program name; defaults to
        ``sys.argv[1:]``.
    :raises StepError: On a configuration or data error.
    """
    step, cfg = _training_args(sys.argv[1:] if argv is None else argv)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if cfg.gradient_checkpointing:
        cfg.gradient_checkpointing_kwargs = {
            "use_reentrant": use_reentrant(step.method, step.distributed, world_size)
        }

    tokenizer = _load_tokenizer(step)
    dataset = _load_dataset(step)
    alora_tokens = _alora_tokens(step, tokenizer)
    model = _load_model(step, local_rank)
    trainer = _build_trainer(step, cfg, model, tokenizer, dataset, alora_tokens)
    _check_alora(trainer, alora_tokens)
    trainer.train()
    _save_adapter(trainer, step, cfg)


if __name__ == "__main__":
    try:
        main()
    except StepError as e:
        fail(str(e))
