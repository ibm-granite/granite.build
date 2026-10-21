# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

import dataclasses
import importlib.resources as pkg_resources
import json
import os
import random
import socket
import warnings
from collections.abc import Mapping, Sequence, Sized
from dataclasses import dataclass, field
from importlib.metadata import version
from itertools import accumulate
from typing import Any, Literal, TypeVar

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
import transformers
from accelerate import Accelerator, PartialState, logging
from accelerate.state import AcceleratorState
from huggingface_hub import ModelCard, ModelCardData
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler
from transformers import (
    AutoConfig,
    BitsAndBytesConfig,
    EvalPrediction,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerState,
    TrainingArguments,
    is_comet_available,
)
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from transformers.utils import (
    ModelOutput,
    is_peft_available,
    is_rich_available,
    is_torch_mlu_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)

if is_rich_available():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

if is_comet_available():
    import comet_ml

if is_peft_available():
    from peft import LoraConfig, PeftConfig


logger = logging.get_logger(__name__)


# Mode-1: pre_tokenizer / BPE-override sensitive strings.
_TOKENIZER_MODE1_PROBES = ("a\n\nb", "public.roads2", "1200", "schema\n\nfollow")

# Mode-2: named special-token literals emitted by the granite chat template
# (see docs/tokenizer_mismatch.md, "Secondary divergence"). If the added_tokens
# id table on the two sides disagrees, these literals encode to different id
# sequences on each side.
_TOKENIZER_MODE2_PROBES = (
    "<tool_call>", "</tool_call>",
    "<tool_response>", "</tool_response>",
    "<think>", "</think>",
    "<think_on>", "<think_off>",
    "<|start_of_role|>", "<|end_of_role|>",
    "<schema>", "</schema>",
    "<tools>", "</tools>",
    "<documents>", "</documents>",
    "<|end_of_text|>", "<|pad|>", "<|unk|>",
)

# Compound chat-template-shaped probe — catches interactions between mode-1
# BPE differences on regular text and mode-2 special-token boundaries.
_TOKENIZER_COMPOUND_PROBES = (
    "<|start_of_role|>system<|end_of_role|>you are helpful<|end_of_text|>",
)


def _tokenizer_backend(tok):
    """Return the tokenizers-lib backend object for either fast or slow
    HF tokenizers, or None if neither exposes one. PreTrainedTokenizerFast
    exposes it as `backend_tokenizer`; slow tokenizers that internally use
    a rust backend (e.g. GPT2Tokenizer via the __init__ override described
    in docs/tokenizer_mismatch.md) expose it as `_tokenizer`.
    """
    return getattr(tok, "backend_tokenizer", None) or getattr(tok, "_tokenizer", None)


def verify_fast_tokenizer(tokenizer, model_dir: str, *, source_label: str) -> None:
    """Raise RuntimeError if `tokenizer` is not backed by the fast tokenizer
    declared in `<model_dir>/tokenizer.json`.

    The failure mode this catches is documented in docs/tokenizer_mismatch.md
    → "Which tokenizer was the teacher trained with? (empirical, 2026-07-03)":
    when `tokenizer_config.json` declares `tokenizer_class: "GPT2Tokenizer"`,
    `AutoTokenizer.from_pretrained(model_dir)` constructs that class, whose
    `__init__` unconditionally rebuilds the backend BPE and overrides the
    pre_tokenizer to `ByteLevel(use_regex=True)` — NOT what the granite model
    was trained on (empirically confirmed: PPL/token 26.1 vs 3.29 with the
    correct fast tokenizer). Both training and vLLM on-policy rollouts silently
    corrupt when this fires — text is tokenized under a distribution the model
    never saw.

    Corrected 2026-08-25: this docstring previously named the legacy
    `vocab.json` + `merges.txt` sidecars as the trigger. Measured under this
    tree's transformers 5.8.0, confirmed by direct measurement, the sidecars are inert — with only
    them alongside tokenizer.json the resolved pre_tokenizer is still the
    trained one — and the single load-bearing input is the `tokenizer_class`
    key. The checks below never keyed on file presence, so their behaviour is
    unchanged; only this explanation was wrong. See the variant table at the top
    of docs/tokenizer_mismatch.md.

    Checks in order (first failure raises):
      1. `tokenizer.is_fast is True`.
      2. `<model_dir>/tokenizer.json` exists and can be loaded as
         `PreTrainedTokenizerFast(tokenizer_file=...)`.
      3. `repr` equality of {pre_tokenizer, post_processor, normalizer, decoder}
         between the loaded tokenizer's live backend and the reference built
         from `tokenizer.json`. Detects the slow-path pre_tokenizer override.
      4. Mode-1 encoding probes agree between the two. Guards against subtle
         load-order bugs where reprs match but behavior diverges.
    """
    from transformers import PreTrainedTokenizerFast

    footer = (
        "See docs/tokenizer_mismatch.md → "
        "'Which tokenizer was the teacher trained with?' for root cause. "
        "Remediation: set tokenizer_class=\"PreTrainedTokenizerFast\" in "
        "tokenizer_config.json (what retag_student.py does), or drop the key "
        "entirely for a tokenizer-only dir with no config.json (what "
        "build_overlay.py does). Stripping legacy vocab.json/merges.txt is NOT "
        "a remedy — those sidecars are inert under transformers 5.8.0 "
        "confirmed by direct measurement; an earlier version of this message advised it."
    )
    header = (
        f"[{source_label}] loaded tokenizer does not match the fast tokenizer "
        f"declared in tokenizer.json.\n"
        f"  model_dir: {model_dir}\n"
    )

    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            header
            + f"  loaded tokenizer class is {type(tokenizer).__name__!r} "
              "(is_fast=False). tokenizer.json was not honored on load.\n"
            + footer
        )

    tokenizer_json = os.path.join(model_dir, "tokenizer.json")
    if not os.path.isfile(tokenizer_json):
        raise RuntimeError(
            header
            + f"  {tokenizer_json} does not exist; cannot verify against a "
              "declared fast tokenizer.\n"
            + footer
        )

    ref = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json)

    loaded_backend = _tokenizer_backend(tokenizer)
    ref_backend = _tokenizer_backend(ref)
    if loaded_backend is None or ref_backend is None:
        raise RuntimeError(
            header
            + "  could not access the tokenizers-lib backend on one of the "
              "tokenizers (mixed fast/slow load path).\n"
            + footer
        )
    for attr in ("pre_tokenizer", "post_processor", "normalizer", "decoder"):
        loaded_val = repr(getattr(loaded_backend, attr, None))
        ref_val = repr(getattr(ref_backend, attr, None))
        if loaded_val != ref_val:
            raise RuntimeError(
                header
                + f"  differing {attr} vs tokenizer.json reference:\n"
                + f"    loaded:    {loaded_val}\n"
                + f"    reference: {ref_val}\n"
                + footer
            )

    for probe in _TOKENIZER_MODE1_PROBES:
        loaded_ids = tokenizer.encode(probe, add_special_tokens=False)
        ref_ids = ref.encode(probe, add_special_tokens=False)
        if loaded_ids != ref_ids:
            raise RuntimeError(
                header
                + f"  Mode-1 encoding probe {probe!r} differs vs "
                  "tokenizer.json reference:\n"
                + f"    loaded ids:    {loaded_ids}\n"
                + f"    reference ids: {ref_ids}\n"
                + footer
            )


def verify_tokenizer_consistency(
    train_tokenizer,
    ref_tokenizer,
    *,
    train_source: str,
    ref_source: str,
    context: str = "KD",
) -> None:
    """Raise RuntimeError if `train_tokenizer` and `ref_tokenizer` disagree on
    class, pre_tokenizer, post_processor, normalizer, decoder,
    added_tokens_decoder, or a battery of encoding probes covering both
    Mode-1 (pre_tokenizer / BPE override) and Mode-2 (special-token id
    table) failure modes documented in docs/tokenizer_mismatch.md.

    Call sites: between the train_tokenizer and whatever "reference"
    tokenizer it must agree with — the precompute-time teacher tokenizer
    for off-policy KD, or the live teacher tokenizer for on-policy GOLD.

    All checks run in order, first failure raises with `context`,
    `train_source`, `ref_source`, and a pointer to
    docs/tokenizer_mismatch.md.

    Note: class name equality is intentionally not checked, and this is
    the right call for a sharper reason than originally recorded. Under a
    transformers>=5.8 venv a teacher dir loads as
    GPT2Tokenizer while a student dir shipping only tokenizer.json loads
    as TokenizersBackend — but the asymmetry tracks the
    `tokenizer_class` key in tokenizer_config.json, not the presence of
    vocab.json+merges.txt as this note used to say (confirmed by direct measurement: the
    sidecars are inert at this version). The class name is genuinely
    uninformative either way: it reads GPT2Tokenizer both when
    segmentation is correct (the 4.2 teacher, whose tokenizer.json
    pre_tokenizer is already a plain ByteLevel, so the override is a
    no-op) and when it is broken (a 4.1 base student, whose
    Sequence[Split, ByteLevel] the override discards). Only the
    backend-repr and encoding-probe checks below can tell those apart,
    and they fire with actionable diagnostics.
    """
    header = (
        f"{context}: training tokenizer and reference tokenizer disagree. "
        "This will silently corrupt training by mis-aligning token spans.\n"
        f"  training tokenizer dir:  {train_source}\n"
        f"  reference tokenizer dir: {ref_source}\n"
    )
    footer = "See docs/tokenizer_mismatch.md for root cause and workarounds."

    # 1-4. Backend-level repr equality: pre_tokenizer, post_processor,
    # normalizer, decoder. Catches e.g. GPT2Tokenizer slow-path injecting a
    # TemplateProcessing post_processor even when tokenizer.json says none.
    train_backend = _tokenizer_backend(train_tokenizer)
    ref_backend = _tokenizer_backend(ref_tokenizer)
    if (train_backend is None) != (ref_backend is None):
        raise RuntimeError(
            header
            + "  one tokenizer exposes a rust backend and the other does "
              "not (mixed fast/slow load path).\n"
            + footer
        )
    if train_backend is not None and ref_backend is not None:
        for attr in ("pre_tokenizer", "post_processor", "normalizer", "decoder"):
            train_val = repr(getattr(train_backend, attr, None))
            ref_val = repr(getattr(ref_backend, attr, None))
            if train_val != ref_val:
                raise RuntimeError(
                    header
                    + f"  differing {attr}:\n"
                    + f"    training:  {train_val}\n"
                    + f"    reference: {ref_val}\n"
                    + footer
                )

    # 5. added_tokens_decoder entry-by-entry equality on the `.content`
    # string. Primary Mode-2 fast-fail: catches the "id 100270 = <tool_call>
    # on teacher, <|unused_1|> on student" family without waiting for the
    # chat template to emit the literal.
    train_added = getattr(train_tokenizer, "added_tokens_decoder", {}) or {}
    ref_added = getattr(ref_tokenizer, "added_tokens_decoder", {}) or {}
    for tid in sorted(set(train_added) | set(ref_added)):
        tv = train_added.get(tid)
        rv = ref_added.get(tid)
        tc = getattr(tv, "content", None) if tv is not None else None
        rc = getattr(rv, "content", None) if rv is not None else None
        if tc != rc:
            raise RuntimeError(
                header
                + f"  added_tokens_decoder entry differs at id={tid}:\n"
                + f"    training:  {tc!r}\n"
                + f"    reference: {rc!r}\n"
                + footer
            )

    # 6. Extended encoding probe battery — Mode-1 + Mode-2 + one compound.
    for probe in (*_TOKENIZER_MODE1_PROBES, *_TOKENIZER_MODE2_PROBES, *_TOKENIZER_COMPOUND_PROBES):
        train_ids = train_tokenizer.encode(probe, add_special_tokens=False)
        ref_ids = ref_tokenizer.encode(probe, add_special_tokens=False)
        if train_ids != ref_ids:
            raise RuntimeError(
                header
                + f"  encoding probe {probe!r} differs:\n"
                + f"    training ids:  {train_ids}\n"
                + f"    reference ids: {ref_ids}\n"
                + footer
            )


@dataclass
class CustomDataCollatorForChatML:
    """
    Data collator for ChatML format datasets.
    """

    tokenizer: PreTrainedTokenizerBase
    ignore_index: int = -100
    max_length: int = None
    prompt_key: str = "prompt"
    messages_key: str = "messages"
    instruction_template: str = None
    response_template: str = None
    last_message_only: bool = False

    def __post_init__(self):
        if self.tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer does not have a pad token. Please set `pad_token_id` in the tokenizer.")
        if self.max_length is None:
            # set a sensible default
            self.max_length = min(self.tokenizer.model_max_length, 1024)

        # Tokenize templates once during initialization
        if self.response_template is not None:
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            self.response_token_ids = None

        if self.instruction_template is not None:
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            self.instruction_token_ids = None

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = []
        attention_mask = []
        prompts_input_ids = []
        prompt_attention_mask = []
        labels = []

        # Collect messages for cross-tokenizer distillation
        messages_list = [example.get("messages", None) for example in examples]
        tools_list = [example.get("tools", None) for example in examples]

        for example in examples:
            formatted_prompt = example.get(self.prompt_key, None)
            if formatted_prompt is None:
                formatted_prompt = example['original_prompt_text']
            #     prompt = example[self.messages_key][:-1]
            #     formatted_prompt = self.tokenizer.apply_chat_template(
            #         prompt,
            #         tokenize=False,
            #         add_generation_prompt=True,
            #         tools=example["tools"],
            #         documents=example["documents"],
            #     )
            # if formatted_prompt != example['original_prompt_text']:
            #     raise ValueError("formatted_prompt does not match original_prompt_text")

            if "input_ids" not in example:
                raise ValueError("input_ids not found in example")
                # message = example[self.messages_key]
                # formatted_message = self.tokenizer.apply_chat_template(
                #     message,
                #     tokenize=False,
                #     add_generation_prompt=False,
                #     tools=example["tools"],
                #     documents=example["documents"],
                # )

                # tokenized_message = self.tokenizer(
                #     formatted_message,
                #     truncation=False,
                #     padding=False,
                #     return_tensors=None,
                #     add_special_tokens=False,
                #     return_offsets_mapping=True,
                # )
                # message_input_ids_full = tokenized_message["input_ids"]
                # offsets = tokenized_message.get("offset_mapping")

                # if offsets is not None:
                #     prompt_char_len = len(formatted_prompt)
                #     completion_start_idx_full = next(
                #         (idx for idx, (start, _) in enumerate(offsets) if start >= prompt_char_len),
                #         len(message_input_ids_full),
                #     )
                # else:
                #     tokenized_prompt_full = self.tokenizer(
                #         formatted_prompt,
                #         truncation=False,
                #         padding=False,
                #         return_tensors=None,
                #         add_special_tokens=False,
                #     )
                #     completion_start_idx_full = len(tokenized_prompt_full["input_ids"])
                
                # prompt_tokens_full = message_input_ids_full[:completion_start_idx_full]
                # completion_input_ids_full = message_input_ids_full[completion_start_idx_full:]

                # if self.max_length is not None and len(message_input_ids_full) > self.max_length:
                #     completion_ids = completion_input_ids_full
                #     if len(completion_ids) >= self.max_length:
                #         completion_ids = completion_ids[-self.max_length :]
                #         prompt_ids = []
                #     else:
                #         max_prompt_tokens = self.max_length - len(completion_ids)
                #         prompt_ids = prompt_tokens_full[-max_prompt_tokens:] if max_prompt_tokens > 0 else []
                #     message_input_ids = prompt_ids + completion_ids
                # else:
                #     message_input_ids = message_input_ids_full
                #     prompt_ids = prompt_tokens_full

                # input_ids.append(message_input_ids)
                # attention_mask.append([1] * len(message_input_ids))
                # current_prompt_ids = prompt_ids
            else:
                message_input_ids = example["input_ids"]
                input_ids.append(message_input_ids)
                if "attention_mask" in example:
                    attention_mask.append(example["attention_mask"])
                else:
                    attention_mask.append([1] * len(message_input_ids))

                tokenized_prompt = self.tokenizer(
                    formatted_prompt,
                    truncation=True,
                    max_length=len(message_input_ids),
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=False,
                )
                current_prompt_ids = tokenized_prompt["input_ids"]

            # ### sanity check
            # assert " ".join([str(x) for x in message_input_ids]).startswith(" ".join([str(x) for x in current_prompt_ids]))
            # import pdb; pdb.set_trace()

            prompts_input_ids.append(current_prompt_ids)
            prompt_attention_mask.append([1] * len(current_prompt_ids))

            # Create labels to train on all assistant messages
            sequence = input_ids[-1]
            label = [self.ignore_index] * len(sequence)

            if self.last_message_only or self.response_token_ids is None:
                # Fallback: train only on last completion
                completion_start_idx = len(current_prompt_ids)
                label[completion_start_idx:] = sequence[completion_start_idx:]
            else:
                # Use pre-tokenized response template
                response_template_len = len(self.response_token_ids)

                # Find all occurrences of response_template in input_ids
                i = 0
                while i <= len(sequence) - response_template_len:
                    # Check if response_template matches at position i
                    if sequence[i:i + response_template_len] == self.response_token_ids:
                        # Found response template, find where this assistant message ends
                        start_idx = i + response_template_len

                        # Look for EOS token or end of sequence
                        end_idx = len(sequence)
                        if self.tokenizer.eos_token_id is not None:
                            for j in range(start_idx, len(sequence)):
                                if sequence[j] == self.tokenizer.eos_token_id:
                                    end_idx = j + 1  # Include the EOS token in training
                                    break

                        # Set labels for this assistant span
                        label[start_idx:end_idx] = sequence[start_idx:end_idx]

                        i = end_idx
                    else:
                        i += 1

            labels.append(label)

        # convert to list of tensors and pad
        input_ids = [torch.tensor(ids, dtype=torch.long) for ids in input_ids]
        attention_mask = [torch.tensor(mask, dtype=torch.long) for mask in attention_mask]
        labels = [torch.tensor(label, dtype=torch.long) for label in labels]
        input_ids = pad(input_ids, padding_side="left", padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad(attention_mask, padding_side="left", padding_value=0)
        labels = pad(labels, padding_side="left", padding_value=self.ignore_index)

        prompts_input_ids = [torch.tensor(ids, dtype=torch.long) for ids in prompts_input_ids]
        prompt_attention_mask = [torch.tensor(mask, dtype=torch.long) for mask in prompt_attention_mask]
        prompts_input_ids = pad(prompts_input_ids, padding_side="left", padding_value=self.tokenizer.pad_token_id)
        prompt_attention_mask = pad(prompt_attention_mask, padding_side="left", padding_value=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompts": prompts_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "on_policy": [False] * input_ids.shape[0],
            "messages": messages_list,
            "tools": tools_list,
        }

def pad(
    tensors: list[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "right",
    pad_to_multiple_of: int | None = None,
) -> torch.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension.

    Args:
        tensors (`list[torch.Tensor]`):
            List of input tensors to pad.
        padding_value (`int`):
            Value to use for padding. Default is 0.
        padding_side (`str`):
            Side on which to add padding. Must be 'left' or 'right'. Default is 'right'.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

    Returns:
        `torch.Tensor`:
            A single tensor containing the padded tensors.

    Examples:
    ```python
    >>> import torch

    >>> pad([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
    tensor([[1, 2, 3],
            [4, 5, 0]])

    >>> pad([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])])
    tensor([[[1, 2],
            [3, 4]],
            [[5, 6],
            [0, 0]]])
    ```
    """
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Apply pad_to_multiple_of to the first (sequence) dimension
    if pad_to_multiple_of is not None:
        remainder = output_shape[0] % pad_to_multiple_of
        if remainder != 0:
            output_shape[0] += pad_to_multiple_of - remainder

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        if padding_side == "left":
            seq_start = output_shape[0] - t.shape[0]
        elif padding_side == "right":
            seq_start = 0
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        # Define the slices
        seq_slice = slice(seq_start, seq_start + t.shape[0])
        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output