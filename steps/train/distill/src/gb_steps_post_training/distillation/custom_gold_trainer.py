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
#
# PORTED, not authored here: imported from
# github.com/laminair/gb-steps-distillation at commit
# a5d59bc45524a8d75706e20d44ae1a254f273f23 (2026-10-21), path
# gb_steps_post_training/distillation/custom_gold_trainer.py, which carries the HuggingFace
# notice above. Local patches are listed in steps/train/distill/src/VENDORED.md.

import html
import json
import math
import os
import random
import re
import textwrap
import time
import warnings
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import (
    DistributedType,
    broadcast_object_list,
    gather_object,
    is_peft_model,
)
from datasets import Dataset, IterableDataset
from gb_steps_post_training.distillation.on_policy_draw import draw_on_policy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoTokenizer, is_bitsandbytes_available
from transformers.data.data_collator import DataCollator
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.generation.configuration_utils import GenerationConfig
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.integration_utils import is_wandb_available
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import EvalPrediction
from transformers.utils import (
    is_flash_attn_2_available,
    is_peft_available,
    is_rich_available,
)
from trl.data_utils import (
    is_conversational,
    maybe_convert_to_chatml,
    pack_dataset,
)
from trl.experimental.utils import empty_cache, truncate_dataset
from trl.extras.profiling import profiling_decorator
from trl.generation.vllm_client import VLLMClient


class MultiVLLMClient:
    """N trl VLLMClients behind the one-client interface trl's VLLMGeneration uses.

    generate() splits the prompts evenly across the servers and merges the results in prompt
    order; update_named_params() streams every tensor to every server's weight-transfer group,
    one weight update per server (trl 1.x's VLLMClient.weight_update), so each server prepares
    and finalizes its model once per sync.
    """

    def __init__(self, clients: list[VLLMClient]):
        if not clients:
            raise ValueError("MultiVLLMClient requires at least one VLLMClient.")
        self.clients = clients
        self._streams = [torch.cuda.Stream() for _ in clients]

    @property
    def base_url(self) -> str:
        return self.clients[0].base_url

    def init_communicator(self, **kwargs):
        for client in self.clients:
            client.init_communicator(**kwargs)

    def generate(self, prompts, **kwargs) -> dict[str, list]:
        n_servers = len(self.clients)
        base, remainder = divmod(len(prompts), n_servers)
        chunks, offset = [], 0
        for i in range(n_servers):
            size = base + (1 if i < remainder else 0)
            chunks.append(prompts[offset : offset + size])
            offset += size

        from concurrent.futures import ThreadPoolExecutor

        def _gen(client, chunk):
            return client.generate(prompts=chunk, **kwargs) if chunk else None

        with ThreadPoolExecutor(max_workers=n_servers) as pool:
            results = list(pool.map(_gen, self.clients, chunks))

        # Chunks are contiguous slices in prompt order, so extending in result order keeps every
        # list aligned with the prompts. prompt_ids is merged too: callers read it.
        keys = ("prompt_ids", "completion_ids", "logprobs", "logprob_token_ids")
        merged = {k: [] for k in keys}
        for r in results:
            if r is None:
                continue
            for k in keys:
                if r.get(k) is None:
                    merged[k] = None
                elif merged[k] is not None:
                    merged[k].extend(r[k])
        return merged

    def update_named_params(self, metadata, named_params):
        from concurrent.futures import ThreadPoolExecutor
        from contextlib import ExitStack

        def _send(client, stream, meta, item):
            torch.cuda.set_stream(stream)
            client.update_named_params([meta], iter([item]))

        with (
            ExitStack() as stack,
            ThreadPoolExecutor(max_workers=len(self.clients)) as pool,
        ):
            for client in self.clients:
                stack.enter_context(client.weight_update())
            for meta, item in zip(metadata, named_params, strict=True):
                futures = [
                    pool.submit(_send, c, st, meta, item)
                    for c, st in zip(self.clients, self._streams)
                ]
                for f in futures:
                    f.result()
                # The broadcasts run on side streams: finish them before the caller's gather
                # context releases (and DeepSpeed frees) the tensor.
                for st in self._streams:
                    st.synchronize()

    def reset_prefix_cache(self):
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(self.clients)) as pool:
            for f in [pool.submit(c.reset_prefix_cache) for c in self.clients]:
                f.result()

    def close_communicator(self):
        for client in self.clients:
            client.close_communicator()


from trl.models import prepare_deepspeed
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.utils import (
    create_model_from_path,
    disable_dropout_in_model,
    pad,
)

if is_peft_available():
    from peft import PeftConfig

if is_wandb_available():
    import wandb

if is_rich_available():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

if is_bitsandbytes_available():
    import bitsandbytes as bnb

from trl.experimental.gold.gold_trainer import GOLDTrainer, ULDLoss


def mask_pad_labels_by_value(
    labels: torch.Tensor, pad_token_id: int | None, eos_token_id: int | None
) -> torch.Tensor:
    """Set label positions holding the pad token to -100, unless pad == eos.

    gold.py sets ``pad_token = eos_token`` for models without a dedicated pad token, so
    pad == eos is the default, not an edge case. Masking by value then also masks every
    real EOS, and the student never gets a signal for stopping. Every caller masks the
    prompt and padding by position as well, so skipping the value mask loses nothing.
    Edits ``labels`` in place and returns it.
    """
    if pad_token_id is not None and pad_token_id != eos_token_id:
        labels[labels == pad_token_id] = -100
    return labels


def build_teacher_inputs_from_texts_with_eos_control(
    tokenizer: PreTrainedTokenizerBase,
    prompt_texts: list[str],
    completion_texts: list[str],
    append_eos: list[bool] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Like build_teacher_inputs_from_texts but with per-sample EOS control.

    Prompts are left-padded to max_prompt_length so that completions start at
    the same position across the batch, matching the student's left-padding
    behavior from CustomDataCollatorForChatML.
    """
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_value = pad_token_id if pad_token_id is not None else 0

    prompt_token_ids = tokenizer(prompt_texts, add_special_tokens=True)["input_ids"]
    completion_token_ids = tokenizer(completion_texts, add_special_tokens=False)[
        "input_ids"
    ]

    # First pass: strip trailing EOS from prompts and compute max_prompt_length
    cleaned_prompt_ids: list[list[int]] = []
    for prompt_ids in prompt_token_ids:
        if eos_token_id is not None and prompt_ids and prompt_ids[-1] == eos_token_id:
            prompt_ids = prompt_ids[:-1]
        cleaned_prompt_ids.append(prompt_ids)

    max_prompt_length = (
        max(len(p) for p in cleaned_prompt_ids) if cleaned_prompt_ids else 0
    )

    # Second pass: build left-padded sequences
    sequences: list[torch.Tensor] = []
    attention_masks: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    for i, (prompt_ids, completion_ids) in enumerate(
        zip(cleaned_prompt_ids, completion_token_ids, strict=True)
    ):
        left_pad_count = max_prompt_length - len(prompt_ids)

        sequence = (
            [pad_value] * left_pad_count + list(prompt_ids) + list(completion_ids)
        )
        should_append = append_eos is None or append_eos[i]
        if eos_token_id is not None and should_append:
            sequence.append(eos_token_id)

        seq_tensor = torch.tensor(sequence, dtype=torch.long)
        sequences.append(seq_tensor)

        attn_mask = [0] * left_pad_count + [1] * (len(sequence) - left_pad_count)
        attention_masks.append(torch.tensor(attn_mask, dtype=torch.long))

        labels = list(sequence)
        # Mask left-padding and prompt positions
        for j in range(left_pad_count + len(prompt_ids)):
            labels[j] = -100
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        # Positional masking above covers the prompt and left-padding, and `pad()`
        # below right-pads labels with -100; the value mask must not eat the
        # appended EOS when pad == eos (see mask_pad_labels_by_value).
        mask_pad_labels_by_value(labels_tensor, pad_token_id, eos_token_id)
        labels_list.append(labels_tensor)

    teacher_input_ids = pad(sequences, padding_side="right", padding_value=pad_value)
    teacher_attention_mask = pad(
        attention_masks, padding_side="right", padding_value=0
    ).bool()
    teacher_labels = pad(labels_list, padding_side="right", padding_value=-100)

    teacher_prompt_length = max_prompt_length
    return (
        teacher_input_ids,
        teacher_labels,
        teacher_attention_mask,
        teacher_prompt_length,
    )


def earliest_completion_column(labels: torch.Tensor, ignore_index: int = -100) -> int:
    """`teacher_prompt_length` for a teacher that reuses the student's input
    (uld_reuse_student_input): the first column any row labels, and at least 1.

    The crop keeps `teacher_prompt_length - 1` onward, the logit before each row's first
    completion token included, so no row's completion is cut into and none starts at
    position 0 unless its own input has no prompt. The batch's longest prompt is the
    wrong column here: the collator left-pads input_ids, so the rows END together and
    each completion starts at the width less its own length, before that column for any
    row whose completion is longer than the width less the longest prompt. Cropping every
    row at the earliest completion is also no wider than a per-row crop padded to one
    width: for rows that end together, both are the longest completion plus one; for the
    on-policy batch, whose completions start at one column, the two are the same crop. A
    batch with no labelled position gets its last column, a crop of one unlabelled column.
    """
    labelled = (labels != ignore_index).any(dim=0).nonzero()
    if labelled.numel() == 0:
        return labels.shape[1]
    return max(int(labelled[0]), 1)


def contiguous_label_spans(
    labels_row, ignore_index: int = -100
) -> list[tuple[int, int]]:
    """Every maximal run of supervised positions in one label row, as (start, size).

    WHY THIS IS NOT `_get_start_and_size_answers`. That helper (trl gold_trainer.py:687)
    returns the FIRST supervised index and the TOTAL COUNT of supervised positions, and
    every consumer then slices `[start : start + size]` as if the region were contiguous.
    On a single-region row it is. On a row where the collator labelled every assistant
    turn (utils.py:478-499 scans for `response_template` and labels each occurrence
    through its EOS) it is not: the slice then starts at the first assistant turn and
    runs for the total supervised length, straddling the user and tool turns in between
    and stopping before the last assistant turn ends. That region is wrong on its own
    terms, not merely misaligned with the teacher's.
    """
    spans: list[tuple[int, int]] = []
    start = None
    row = labels_row.tolist() if hasattr(labels_row, "tolist") else list(labels_row)
    for i, v in enumerate(row):
        if v != ignore_index:
            if start is None:
                start = i
        elif start is not None:
            spans.append((start, i - start))
            start = None
    if start is not None:
        spans.append((start, len(row) - start))
    return spans


def build_teacher_inputs_from_turn_segments(
    tokenizer: PreTrainedTokenizerBase,
    rows: list[list[tuple[str, bool]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[tuple[int, int]]]]:
    """Build teacher sequences with ONE SUPERVISED REGION PER ASSISTANT TURN.

    `rows[i]` is that row's render split into alternating segments as
    `(text, supervised)` -- unsupervised context, then a supervised assistant turn, then
    the next stretch of context, and so on. Concatenating the segments reproduces the
    row's full render; labelling only the supervised ones reproduces, on the teacher
    side, exactly the structure `CustomDataCollatorForChatML` produces on the student
    side.

    THE RENDER IS TOKENISED ONCE, AND THE REGIONS ARE FOUND BY CHARACTER OFFSET. An earlier
    version of this function tokenised each segment separately and concatenated, on the
    argument that "every boundary here is a chat-template control token, which no merge
    crosses". That argument is FALSE, and the check written to defend it is what disproved
    it, confirmed directly and then attributed by a companion audit tool:
    on the tools corpus with a thinking teacher, the generation prompt ends
    `<|im_start|>assistant\\n<think>\\n` and the turn's content begins `\\n</think>...`, so
    the boundary falls BETWEEN the two newlines -- inside a single canonical `\\n\\n` token.
    Segment-wise tokenisation therefore emitted two `\\n` tokens where the teacher model has
    only ever seen one `\\n\\n`, on 360 of 438 supervised regions (82%), and the teacher was
    asked to score ids it would never be given at inference.

    So: tokenise the concatenated render exactly once -- the ids are then by construction
    the ids the teacher would see if the row were fed to it normally -- and take each
    supervised region to be the tokens whose characters lie WHOLLY inside that turn's
    character interval. A token straddling a boundary is thus assigned to the context side,
    which is the correct side: the straddling token is template scaffolding after `<think>`,
    not the turn's content. Measured on the same rows, region ENDS already land exactly on
    token boundaries (438/438), so the EOS-inclusive end the student's spans require is
    preserved untouched; only the start moves, by at most that one whitespace token.

    This also retires the whole failure class rather than fixing one instance of it. There
    is only one tokenisation now, so no future template change can put a boundary inside a
    mergeable span -- there is no second tokenisation for it to disagree with.

    REQUIRES A FAST TOKENIZER, and raises rather than falling back. Offset mapping is how
    the character intervals become token indices; without it the only alternative is the
    segment-wise pass this replaced, and silently reverting to a known-wrong render is worse
    than refusing to start. All three teachers in configs/distillation/gold ship a
    tokenizer.json, so this is a guard, not a limitation.

    NO LEFT-PADDING, deliberately, and this is the substantive departure. The single-
    region builder left-pads every prompt to a common length so that all completions
    begin at the same column, which is what lets compute_loss crop the batch at
    `teacher_prompt_length - 1` and pay for logits on completion positions only. With one
    region per turn there IS no common column -- the regions are scattered, and differently
    per row -- so the crop is replaced by a GATHER of the positions the spans actually
    name (see `CustomGOLDTrainer._pack_teacher_regions`). Right-padding then costs nothing,
    because nothing downstream reads a shared offset any more.

    Returns (input_ids, labels, attention_mask, spans) where `spans[i]` is the list of
    (start, size) supervised regions in row i, in order -- the same order as the student's
    turns, which is what makes the pairing in the loss meaningful.
    """
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_value = pad_token_id if pad_token_id is not None else 0
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "uld_per_turn_regions needs a FAST teacher tokenizer: the supervised regions are "
            "located by character offset over a single tokenisation of the render, and "
            "`return_offsets_mapping` is available only on the fast tokenizers. The "
            f"tokenizer for this teacher ({type(tokenizer).__name__}) is not fast. Convert it "
            "(a tokenizer.json next to the weights is enough) rather than removing this "
            "check: the alternative is the segment-wise render this replaced, which feeds "
            "the teacher ids it never sees at inference."
        )

    sequences: list[torch.Tensor] = []
    attention_masks: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    spans_out: list[list[tuple[int, int]]] = []

    for segments in rows:
        full = "".join(text for text, _ in segments)
        # Character interval of every segment, in render order. These are offsets into
        # `full`, which IS the render (teacher_turn_segments partitions it and never drops
        # or duplicates a character; uld-per-turn-render.py asserts that separately).
        bounds: list[tuple[int, int, bool]] = []
        cursor = 0
        for text, supervised in segments:
            bounds.append((cursor, cursor + len(text), supervised))
            cursor += len(text)

        enc = tokenizer(full, add_special_tokens=True, return_offsets_mapping=True)
        ids: list[int] = list(enc["input_ids"])
        offsets: list[tuple[int, int]] = [tuple(o) for o in enc["offset_mapping"]]
        # A tokenizer whose `add_special_tokens` APPENDS an EOS would put a second one after
        # the render's own end-of-turn marker, at a position the student has no counterpart
        # for. Such a token is backed by no characters, which is how it is told apart from
        # the render's own `<|im_end|>` (that one carries a real offset span). A prepended
        # BOS is kept -- the old builder kept it too, and its empty span means it can never
        # fall inside a region.
        while (
            ids
            and offsets
            and offsets[-1][1] <= offsets[-1][0]
            and eos_token_id is not None
            and ids[-1] == eos_token_id
        ):
            ids.pop()
            offsets.pop()

        label: list[int] = [-100] * len(ids)
        spans: list[tuple[int, int]] = []
        for lo, hi, supervised in bounds:
            if not supervised:
                continue
            # WHOLLY inside, not merely overlapping: a token that straddles the boundary
            # belongs to the context side (see the docstring). `b > a` drops the specials
            # that carry no characters.
            idx = [
                k for k, (a, b) in enumerate(offsets) if b > a and a >= lo and b <= hi
            ]
            if not idx:
                # The turn's characters do not contain a single whole token. Nothing to
                # supervise, so no region -- the same outcome the old builder reached for an
                # empty segment. The row then has fewer teacher regions than student spans,
                # which `_compute_distillation_loss` counts as a mismatch row and
                # a companion check refuses before any GPU is booked.
                continue
            if idx[-1] - idx[0] + 1 != len(idx):
                raise ValueError(
                    f"supervised region [{lo},{hi}) is not contiguous in the render's "
                    f"tokenisation: token indices {idx[0]}..{idx[-1]} span "
                    f"{idx[-1] - idx[0] + 1} positions but only {len(idx)} lie inside it. "
                    "That means a token with no character span sits in the middle of a turn, "
                    "which contiguous_label_spans cannot represent."
                )
            spans.append((idx[0], len(idx)))
            for k in idx:
                # The EOS is the region's last token, not an appended one: the region's end
                # lands on the render's own `<|im_end|>` because teacher_turn_segments cut
                # the turn's text there, and the student's span likewise ends at EOS
                # inclusive (utils.py:487, `end_idx = j + 1`).
                label[k] = ids[k]

        seq_tensor = torch.tensor(ids, dtype=torch.long)
        sequences.append(seq_tensor)
        attention_masks.append(torch.ones(len(ids), dtype=torch.long))
        labels_tensor = torch.tensor(label, dtype=torch.long)
        # NOTE: no `labels[labels == pad_token_id] = -100` here. The single-region builder
        # does that to unmask left-padding it inserted; this builder inserts none, so the
        # same line would silently drop a real pad token that a turn legitimately contains
        # and split one span into two.
        labels_list.append(labels_tensor)
        spans_out.append(spans)

    teacher_input_ids = pad(sequences, padding_side="right", padding_value=pad_value)
    teacher_attention_mask = pad(
        attention_masks, padding_side="right", padding_value=0
    ).bool()
    teacher_labels = pad(labels_list, padding_side="right", padding_value=-100)
    return teacher_input_ids, teacher_labels, teacher_attention_mask, spans_out


def parse_template_kwargs(entries) -> dict:
    """`KEY=VALUE` strings -> chat-template variables; `true`/`false` become bools.

    Shared with a companion analysis tool's --template-kwarg so the
    spelling that MEASURED a template variable is the spelling that configures it.
    """
    out: dict = {}
    for kv in entries or []:
        k, _, v = kv.partition("=")
        low = v.strip().lower()
        out[k.strip()] = True if low == "true" else False if low == "false" else v
    return out


def _turn_boundary(full: str, prefix: str, through: str, j: int) -> tuple[int, bool]:
    """Character offset in `full` where assistant turn `j`'s own rendered text begins.

    Normally `len(prefix)`: the generation-prompt render is a prefix of the full render, so
    the two agree on where the turn starts. Against granite's template that is false on 36% of
    real corpus rows, and the disagreement is always one shape -- 1068 of 1081 refusals, job
    1669148: the generation prompt speculatively opens `<think>\\n` while a turn carrying no
    reasoning renders `<think></think>`, so the prefix ends in ONE newline the full render does
    not have.

    When the prefix's ENTIRE excess is whitespace and `through` is still a prefix of `full`,
    the boundary is the divergence point and nothing about it is ambiguous. `full` is the
    string the teacher actually scores; the generation prompt is a string this row never
    produces, and its only job here is to locate an offset -- so when the two disagree by
    whitespace alone, the render wins. Resolving to the divergence point supervises `</think>`
    and everything after it, which is precisely the teacher's own continuation after `<think>`
    in the sequence being scored. Same policy as the straddling-token rule in the tokenisation
    below: the render is authoritative and the boundary resolves toward the context side.

    Anything else raises. A `through` that is not a prefix of `full` means the full render does
    not CONTAIN turn j's text as rendered on its own -- granite's template does that to
    historical turns when `truncate_history_thinking` is left at its default True (110 rows in
    3000, confirmed directly; zero once it is passed False) -- and no offset arithmetic
    can recover text that is not in the string.

    Returns (boundary, relaxed) so the caller can COUNT the relaxation. A policy that fires on
    a third of the rows has to be observable in the run log, not just correct here.
    """
    if not full.startswith(through):
        raise ValueError(
            f"teacher renders are not prefix-monotone at turn {j}: render(msgs[:j+1]) is not a "
            f"prefix of render(msgs), so the full render does not contain this turn's own "
            f"rendered text and no offset can recover it. A template that strips reasoning from "
            f"historical turns does exactly this; if it takes a flag for that (granite: "
            f"truncate_history_thinking), pass it through uld_teacher_template_kwargs"
        )
    if full.startswith(prefix):
        if len(through) < len(prefix):
            raise ValueError(
                f"teacher renders are not prefix-monotone at turn {j}: render(msgs[:j+1]) is "
                f"shorter than render(msgs[:j], add_generation_prompt=True), so the turn would "
                f"have negative length"
            )
        return len(prefix), False
    k = next(
        (c for c in range(min(len(prefix), len(full))) if prefix[c] != full[c]),
        min(len(prefix), len(full)),
    )
    if prefix[k:].strip() or k > len(through):
        raise ValueError(
            f"teacher renders are not prefix-monotone at turn {j}: the generation-prompt render "
            f"diverges from the full render at char {k} by more than trailing whitespace "
            f"({prefix[k:k + 40]!r} vs {full[k:k + 40]!r}), so where this turn begins is a guess"
        )
    return k, True


def teacher_turn_segments(
    tokenizer: PreTrainedTokenizerBase,
    msgs: list[dict],
    tools_kwargs: dict,
    enable_thinking: bool | None,
    turn_suffix: str | None,
    eos_str: str | None = None,
    append_eos: bool = True,
    stats: dict | None = None,
) -> list[tuple[str, bool]]:
    """Split one row's teacher render into (text, supervised) segments, one per turn.

    Built from PREFIX RENDERS rather than by searching the full render for a header:
    `apply_chat_template(msgs[:j], add_generation_prompt=True)` is where turn j's content
    begins and `apply_chat_template(msgs[:j+1])` is where it ends, both by the template's
    own account. A search for the assistant header would instead be a second, independent
    implementation of the template's turn structure -- which is the bug class this whole
    change exists to remove, not one to reintroduce on the teacher side.

    Raises ValueError if the renders are not prefix-monotone, because every offset below
    is computed from the assumption that they are, and a template that violates it would
    otherwise yield silently mislabelled regions.
    """

    def _render(m, gen_prompt):
        kw = dict(tokenize=False, add_generation_prompt=gen_prompt, **tools_kwargs)
        if enable_thinking is not None:
            kw["enable_thinking"] = enable_thinking
        return tokenizer.apply_chat_template(m, **kw)

    segments: list[tuple[str, bool]] = []
    cursor = 0
    full = _render(msgs, False)
    last_assistant = max(
        (j for j, m in enumerate(msgs) if m.get("role") == "assistant"), default=-1
    )
    for j, msg in enumerate(msgs):
        if msg.get("role") != "assistant":
            continue
        prefix = _render(msgs[:j], True)
        through = _render(msgs[: j + 1], False)
        boundary, relaxed = _turn_boundary(full, prefix, through, j)
        if relaxed and stats is not None:
            stats["relaxed_ws_boundaries"] = stats.get("relaxed_ws_boundaries", 0) + 1
        content = through[boundary:]
        # WHERE THE SUPERVISED REGION ENDS. The student's span runs through its EOS token
        # inclusive and stops there, so the end-of-turn marker is supervised and whatever
        # the template puts AFTER it (granite and qwen both emit a newline) is not. Cut at
        # the EOS string for that reason, and fall back to the turn suffix only when the
        # tokenizer has no eos string to cut at.
        if eos_str and eos_str in content:
            content = content[: content.index(eos_str) + len(eos_str)]
        elif turn_suffix and content.endswith(turn_suffix):
            content = content[: -len(turn_suffix)]
        # A TRUNCATED student row has no final EOS to supervise (append_eos is False for
        # exactly that case, computed from the student's own last token in compute_loss), so
        # the teacher's last turn must not carry one either. Intermediate turns are
        # unaffected: their EOS is in the middle of the sequence and cannot be the
        # truncation point.
        if (
            j == last_assistant
            and not append_eos
            and eos_str
            and content.endswith(eos_str)
        ):
            content = content[: -len(eos_str)]
        if not content:
            continue
        if boundary > cursor:
            segments.append((full[cursor:boundary], False))
        segments.append((content, True))
        cursor = boundary + len(content)
    if cursor < len(full):
        segments.append((full[cursor:], False))
    return segments


def _row_teacher_enable_thinking(msgs) -> bool:
    """Does the assistant turn being learned open a reasoning block?

    The mixture is bimodal (58% thinking / 42% not) and the teacher template's generation prompt
    commits to `<think>\\n` or to a closed empty `<think>\\n\\n</think>\\n\\n` from this one
    value, so a global is wrong for one half of the corpus whichever way it is set. Derived from
    the row rather than read from the corpus's `render_thinking` column because that column does
    not reach the batch (`columns_to_keep`, :2031, admits only `messages` under ULD) and because
    the on-policy path synthesises messages that have no corpus row behind them.

    `reasoning_content` is checked first: a row that carries reasoning in the proper field is a
    thinking row even though its `content` does not start with the tag.
    """
    if not msgs:
        return False
    last = msgs[-1]
    if not isinstance(last, dict):
        return False
    reasoning = last.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return True
    content = last.get("content") or ""
    return isinstance(content, str) and content.lstrip().startswith("<think>")


def _is_weight_stats(weights: torch.Tensor) -> dict:
    """mean/std/min/max of the importance weights that exist (NaN tokens already dropped)."""
    if weights.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": weights.mean().item(),
        "std": weights.std().item() if weights.numel() > 1 else 0.0,
        "min": weights.min().item(),
        "max": weights.max().item(),
    }


class CustomULDLoss(ULDLoss):
    """trl's ULDLoss with this trainer's fixes. Each is described where it is made:

    - EOS for empty completions. When the teacher generates only an EOS token, the parent
      ULDLoss produces zero loss because skip_student_eos=True decrements answer_size from 1
      to 0. This subclass forces EOS inclusion and ensures EOS tokens are properly aligned
      between student and teacher vocabularies.
    - The base (non-hybrid) path, `_compute_base_distillation_loss`: trl 0.26.2's, owned
      here, reading the logits before each token rather than at it, and skipping a row whose
      alignment yields no groups instead of scoring it unmerged.
    - Every row (region, on the hybrid path) that reaches alignment is counted in
      `_alignment_rows`, and every one that yields no groups in `_alignment_unaligned`. The
      trainer's ULDAlignmentGuard drains both at each optimizer step, reduces them across
      ranks, and raises when a whole step, or too much of a window, trained on nothing.
    - A completion that starts at position 0 has no logit predicting its first token; both
      paths skip it and count it in `_alignment_unpredictable` (`_unpredictable_start`),
      and ULDAlignmentGuard refuses the step on every rank.
    """

    def __init__(
        self, config, student_tokenizer=None, teacher_tokenizer=None, trainer=None
    ):
        self._trainer = trainer

        # Resolve EOS token IDs BEFORE super().__init__ because it calls
        # _initialize_vocabulary_mapping() which needs these attributes
        self._student_eos_id = getattr(config, "student_eos_token_id", None)
        if self._student_eos_id is None and student_tokenizer is not None:
            self._student_eos_id = student_tokenizer.eos_token_id
        self._teacher_eos_id = getattr(config, "teacher_eos_token_id", None)
        if self._teacher_eos_id is None and teacher_tokenizer is not None:
            self._teacher_eos_id = teacher_tokenizer.eos_token_id

        super().__init__(config, student_tokenizer, teacher_tokenizer)

        # Force EOS inclusion so single-EOS completions produce non-zero loss
        self.skip_student_eos = False
        self.skip_teacher_eos = False

        self._max_group_size = getattr(config, "max_alignment_group_size", 10)
        self._hybrid_sampled_opd_weight = getattr(
            config, "uld_hybrid_sampled_opd_weight", 0.0
        )
        self._opd_normalize_by_group = getattr(config, "opd_normalize_by_group", False)
        self._opd_ignore_multi_loss = getattr(config, "opd_ignore_multi_loss", False)
        self._opd_ignore_unbalanced_loss = getattr(
            config, "opd_ignore_unbalanced_loss", False
        )
        self._uld_ignore_multi_loss = getattr(config, "uld_ignore_multi_loss", False)
        self._matched_top_k = getattr(config, "uld_matched_top_k", 0)
        self._renormalize_probs = getattr(config, "uld_renormalize_probs", False)
        # Number of ALIGNMENT GROUPS whose loss terms are computed at a time, each chunk wrapped
        # in torch.utils.checkpoint so its full-vocabulary probability tensors are freed and
        # recomputed in backward instead of being held. 0 keeps the historical whole-region
        # behaviour. See _region_terms_chunked for why this exists and what it costs.
        self._region_chunk_groups = int(
            getattr(config, "uld_region_chunk_groups", 0) or 0
        )
        self._opd_normalize_reward_with_baseline = getattr(
            config, "uld_opd_normalize_reward_with_baseline", False
        )
        self._current_vllm_logprobs = None

        self._align_at_bytes = getattr(config, "uld_align_at_bytes", False)
        if self._align_at_bytes:
            from transformers.convert_slow_tokenizer import bytes_to_unicode

            self._unicode_to_byte = {v: k for k, v in bytes_to_unicode().items()}

        self._opd_is_stats_accum = {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "masked_fraction": 0.0,
        }
        self._opd_is_stats_count = 0

        self._opd_reward_stats_accum = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        self._opd_reward_stats_count = 0
        self._opd_reward_1to1_stats_accum = {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
        self._opd_reward_1to1_stats_count = 0
        self._opd_reward_multi_stats_accum = {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
        self._opd_reward_multi_stats_count = 0
        self._opd_reward_1toN_stats_accum = {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
        self._opd_reward_1toN_stats_count = 0
        self._opd_reward_Nto1_stats_accum = {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
        self._opd_reward_Nto1_stats_count = 0
        self._opd_reward_NtoN_stats_accum = {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
        self._opd_reward_NtoN_stats_count = 0

        self._opd_group_type_accum = {
            "total": 0,
            "1to1": 0,
            "1toN": 0,
            "Nto1": 0,
            "NtoN": 0,
        }

        self._alignment_total = 0
        self._alignment_truncated = 0
        # Rows (regions, on the hybrid path) that reached alignment, and those of them whose
        # alignment produced no groups at all and so contributed no loss. RANK-LOCAL and
        # drained at every optimizer step by ULDAlignmentGuard, which reduces them across
        # ranks before it checks or logs them (`train/alignment_unaligned_rate`).
        self._alignment_rows = 0
        self._alignment_unaligned = 0
        # Rows (regions) skipped because a completion starts at position 0. Not a data
        # condition like the two above: the guard refuses any, on every rank alike, which
        # a raise here, on the one rank whose row it is, could not do.
        self._alignment_unpredictable = 0

        self._opd_batch_norm_stats_accum = {"mean": 0.0, "std": 0.0}
        self._opd_batch_norm_stats_count = 0

        self._reward_html_buffer = []

    def _initialize_vocabulary_mapping(self):
        super()._initialize_vocabulary_mapping()

        student_vocab = self.student_tokenizer.get_vocab()
        teacher_vocab = self.teacher_tokenizer.get_vocab()
        matched = len(self._vocab_mapping)
        self.jaccard = (
            matched / (len(student_vocab) + len(teacher_vocab) - matched)
            if (len(student_vocab) + len(teacher_vocab) - matched) > 0
            else 0.0
        )
        print(f"Student Vocab: {len(student_vocab)}")
        print(f"Teacher Vocab: {len(teacher_vocab)}")
        print(f"Matched Vocab: {matched}")
        print(f"Jaccard Index: {self.jaccard:.4f}")

        # Force EOS mapping so EOS-only completions produce aligned groups
        if self._teacher_eos_id is not None and self._student_eos_id is not None:
            already_mapped = (
                self._teacher_eos_id in self._vocab_mapping
                and self._vocab_mapping[self._teacher_eos_id] == self._student_eos_id
            )
            if not already_mapped:
                self._vocab_mapping[self._teacher_eos_id] = self._student_eos_id
                self._teacher_matched_ids.add(self._teacher_eos_id)
                self._student_matched_ids.add(self._student_eos_id)
                # trl 1.x also keeps the mapping as a tensor, built in super() before this
                # addition. Nothing here reads it, but keep it equal to the dict.
                mt = getattr(self, "mapping_tensor", None)
                if mt is not None:
                    if self._teacher_eos_id >= mt.numel():
                        mt = torch.cat(
                            [
                                mt,
                                mt.new_full(
                                    (self._teacher_eos_id + 1 - mt.numel(),), -1
                                ),
                            ]
                        )
                    mt[self._teacher_eos_id] = self._student_eos_id
                    self.mapping_tensor = mt
                print(
                    f"Forced EOS mapping: teacher {self._teacher_eos_id} -> student {self._student_eos_id}"
                )
            else:
                print(
                    f"EOS mapping already present: teacher {self._teacher_eos_id} -> student {self._student_eos_id}"
                )

        # self._inverse_vocab_mapping = {v:k for k,v in self._vocab_mapping.items()} # for degub; remove later

    def __call__(
        self,
        student_logits,
        teacher_logits,
        student_labels,
        teacher_labels,
        student_input_ids,
        teacher_input_ids,
        vllm_logprobs=None,
    ):
        self._current_vllm_logprobs = vllm_logprobs
        result = super().__call__(
            student_logits,
            teacher_logits,
            student_labels,
            teacher_labels,
            student_input_ids,
            teacher_input_ids,
        )
        self._current_vllm_logprobs = None
        return result

    def _build_alignment_groups_from_ids(self, student_token_ids, teacher_token_ids):
        """Build alignment groups with EOS handling and inline truncation for oversized groups.

        Inlines the parent's greedy text-matching loop so we can stop early when a
        group exceeds ``_max_group_size`` on either side (Unicode normalization
        mismatches cause the greedy matcher to lump all remaining tokens into one
        mega-group).  EOS is only appended when the full body aligned successfully.
        """
        s_has_eos = (
            len(student_token_ids) > 0 and student_token_ids[-1] == self._student_eos_id
        )
        t_has_eos = (
            len(teacher_token_ids) > 0 and teacher_token_ids[-1] == self._teacher_eos_id
        )

        s_ids_core = student_token_ids[:-1] if s_has_eos else student_token_ids
        t_ids_core = teacher_token_ids[:-1] if t_has_eos else teacher_token_ids

        s_groups: list[list[int]] = []
        t_groups: list[list[int]] = []
        truncated = False

        if s_ids_core and t_ids_core:

            def to_canonical_pieces(tok, ids):
                pieces = []
                prev = ""
                for k in range(len(ids)):
                    cur = tok.decode(
                        ids[: k + 1],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    pieces.append(cur[len(prev) :])
                    prev = cur
                return pieces

            s_pieces = to_canonical_pieces(self.student_tokenizer, s_ids_core)
            t_pieces = to_canonical_pieces(self.teacher_tokenizer, t_ids_core)

            i = j = 0
            s_buf = t_buf = ""
            s_group: list[int] = []
            t_group: list[int] = []

            def flush():
                if s_group and t_group:
                    s_groups.append(s_group.copy())
                    t_groups.append(t_group.copy())

            while i < len(s_pieces) or j < len(t_pieces):
                if s_buf == t_buf and s_buf != "":
                    flush()
                    s_buf = t_buf = ""
                    s_group = []
                    t_group = []
                    continue

                # Check size limit before accumulating more tokens
                if self._max_group_size > 0 and (
                    len(s_group) >= self._max_group_size
                    or len(t_group) >= self._max_group_size
                ):
                    truncated = True
                    break

                if s_buf == "" and i < len(s_pieces):
                    s_buf += s_pieces[i]
                    s_group.append(i)
                    i += 1
                    continue
                if t_buf == "" and j < len(t_pieces):
                    t_buf += t_pieces[j]
                    t_group.append(j)
                    j += 1
                    continue

                if len(s_buf) <= len(t_buf):
                    if i < len(s_pieces):
                        s_buf += s_pieces[i]
                        s_group.append(i)
                        i += 1
                    elif j < len(t_pieces):
                        t_buf += t_pieces[j]
                        t_group.append(j)
                        j += 1
                else:
                    if j < len(t_pieces):
                        t_buf += t_pieces[j]
                        t_group.append(j)
                        j += 1
                    elif i < len(s_pieces):
                        s_buf += s_pieces[i]
                        s_group.append(i)
                        i += 1

            if not truncated:
                if s_buf == t_buf and s_group and t_group:
                    flush()
                elif s_group or t_group:
                    # Remainder buffers don't match — treat as truncation
                    truncated = True

            # Drop trailing empty student groups (truncated student sequence)
            while s_groups and not s_groups[-1]:
                s_groups.pop()
                t_groups.pop()

        # Only append EOS when the body was fully aligned (no truncation)
        if not truncated and s_has_eos and t_has_eos:
            s_groups.append([len(student_token_ids) - 1])
            t_groups.append([len(teacher_token_ids) - 1])

        self._alignment_total += 1
        if truncated:
            self._alignment_truncated += 1

        return s_groups, t_groups

    # A think block's DELIMITERS are template control text, and the two templates spell them
    # differently: granite renders `<think></think>` at the head of a non-thinking turn and
    # `<think>\n` before reasoning, while Qwen's per-turn region opens either at the content
    # itself or at `</think>\n\n` (its generation prompt having already emitted the opener).
    # Those bytes exist on one side and not the other, so the greedy walk below -- which can
    # buffer past a different SPLIT of the same bytes but never past DIFFERENT bytes -- can
    # never close a group, at any `max_alignment_group_size`. Measured on this round's corpus,
    # confirmed directly: 373 of 373 paired regions
    # diverged at byte 0 of token 0, so every ULD arm trained at exactly zero gradient via the
    # `if not s_groups` fallback below. Trimming this prefix from each side independently took
    # that to 371 of 373 aligning, confirmed by a separate direct measurement.
    _THINK_SCAFFOLD_RE = re.compile(rb"^(?:<think>)?\s*(?:</think>)?\s*$")

    def _scaffold_prefix_len(self, tokenizer, token_ids) -> int:
        """How many LEADING tokens of a region are think-block scaffolding and nothing else.

        Token-granular because the region is a token span: a byte-granular trim could land
        mid-token and there would be no span to express it as. Matched with ^...$ against the
        ACCUMULATED prefix, so it stops at the first token that carries content rather than
        eating whitespace that belongs to one (` Okay` is content, `\\n` alone is not).
        Excluding these positions costs no supervision: the student emits them deterministically
        from its own template, and the teacher has no distribution over the student's spelling
        of them to imitate.
        """
        n = 0
        acc = b""
        for idx, tid in enumerate(token_ids):
            token_str = tokenizer.convert_ids_to_tokens(int(tid))
            try:
                piece = bytes(self._unicode_to_byte[c] for c in token_str)
            except KeyError:
                break
            cand = acc + piece
            if not self._THINK_SCAFFOLD_RE.match(cand):
                break
            acc = cand
            n = idx + 1
        return n

    def _build_alignment_groups_at_bytes_from_ids(
        self, student_token_ids, teacher_token_ids
    ):
        """Build alignment groups using byte-level matching instead of string-level.

        Same greedy matching structure as _build_alignment_groups_from_ids, but
        compares raw UTF-8 bytes instead of decoded Unicode strings. This avoids
        corruption from partial multi-byte sequences (e.g. emojis) that cause
        replacement characters in incremental string decoding.
        """
        s_has_eos = (
            len(student_token_ids) > 0 and student_token_ids[-1] == self._student_eos_id
        )
        t_has_eos = (
            len(teacher_token_ids) > 0 and teacher_token_ids[-1] == self._teacher_eos_id
        )

        s_ids_core = student_token_ids[:-1] if s_has_eos else student_token_ids
        t_ids_core = teacher_token_ids[:-1] if t_has_eos else teacher_token_ids

        s_groups: list[list[int]] = []
        t_groups: list[list[int]] = []
        truncated = False

        if s_ids_core and t_ids_core:

            def to_byte_pieces(tok, ids):
                pieces = []
                for tid in ids:
                    token_str = tok.convert_ids_to_tokens(tid)
                    pieces.append(bytes(self._unicode_to_byte[c] for c in token_str))
                return pieces

            s_pieces = to_byte_pieces(self.student_tokenizer, s_ids_core)
            t_pieces = to_byte_pieces(self.teacher_tokenizer, t_ids_core)

            i = j = 0
            s_buf = b""
            t_buf = b""
            s_group: list[int] = []
            t_group: list[int] = []

            def flush():
                if s_group and t_group:
                    s_groups.append(s_group.copy())
                    t_groups.append(t_group.copy())

            while i < len(s_pieces) or j < len(t_pieces):
                if s_buf == t_buf and s_buf != b"":
                    flush()
                    s_buf = b""
                    t_buf = b""
                    s_group = []
                    t_group = []
                    continue

                if self._max_group_size > 0 and (
                    len(s_group) >= self._max_group_size
                    or len(t_group) >= self._max_group_size
                ):
                    truncated = True
                    break

                if s_buf == b"" and i < len(s_pieces):
                    s_buf += s_pieces[i]
                    s_group.append(i)
                    i += 1
                    continue
                if t_buf == b"" and j < len(t_pieces):
                    t_buf += t_pieces[j]
                    t_group.append(j)
                    j += 1
                    continue

                if len(s_buf) <= len(t_buf):
                    if i < len(s_pieces):
                        s_buf += s_pieces[i]
                        s_group.append(i)
                        i += 1
                    elif j < len(t_pieces):
                        t_buf += t_pieces[j]
                        t_group.append(j)
                        j += 1
                else:
                    if j < len(t_pieces):
                        t_buf += t_pieces[j]
                        t_group.append(j)
                        j += 1
                    elif i < len(s_pieces):
                        s_buf += s_pieces[i]
                        s_group.append(i)
                        i += 1

            if not truncated:
                if s_buf == t_buf and s_group and t_group:
                    flush()
                elif s_group or t_group:
                    truncated = True

            while s_groups and not s_groups[-1]:
                s_groups.pop()
                t_groups.pop()

        if not truncated and s_has_eos and t_has_eos:
            s_groups.append([len(student_token_ids) - 1])
            t_groups.append([len(teacher_token_ids) - 1])

        self._alignment_total += 1
        if truncated:
            self._alignment_truncated += 1
            # student_text = self.student_tokenizer.decode(student_token_ids, skip_special_tokens=False)
            # teacher_text = self.teacher_tokenizer.decode(teacher_token_ids, skip_special_tokens=False)
            # print(
            #     f"[ALIGNMENT TRUNCATED] total={self._alignment_total} truncated={self._alignment_truncated}\n"
            #     f"  student ({len(student_token_ids)} tokens): {student_text}\n"
            #     f"  teacher ({len(teacher_token_ids)} tokens): {teacher_text}"
            # )

        return s_groups, t_groups

    def _compute_sampled_opd_for_groups(
        self,
        student_probs,
        teacher_probs,
        s_token_ids,
        t_token_ids,
        s_groups,
        t_groups,
        vllm_lp_per_pos=None,
        student_logits_slice=None,
        teacher_logits_slice=None,
    ):
        """Sampled OPD loss at alignment group level.

        For each group g:
          s_lp[g] = sum of log p_student(sampled_token) for positions in s_groups[g]
          t_lp[g] = sum of log p_teacher(sampled_token) for positions in t_groups[g]
          reward = (t_lp - s_lp).detach(), with clip_alpha clipping
          loss = -(s_lp * reward).mean()

        When vllm_lp_per_pos is provided, applies truncated IS correction.
        """
        _zero_grad_source = (
            student_probs if student_probs is not None else student_logits_slice
        )

        if not s_groups:
            return (_zero_grad_source.sum() * 0.0, 0)

        s_ids_t = torch.tensor(s_token_ids, device=_zero_grad_source.device)
        t_ids_t = torch.tensor(t_token_ids, device=_zero_grad_source.device)
        if student_logits_slice is not None and teacher_logits_slice is not None:
            s_lp_all = F.log_softmax(
                student_logits_slice / self.student_temperature, dim=-1
            )
            t_lp_all = F.log_softmax(
                teacher_logits_slice / self.teacher_temperature, dim=-1
            )
        else:
            assert (
                student_probs is not None and teacher_probs is not None
            ), "Either logits slices or probability tensors must be provided"
            s_lp_all = torch.log(student_probs.clamp_min(1e-8))
            t_lp_all = torch.log(teacher_probs.clamp_min(1e-8))

        s_lp_per_pos = s_lp_all[
            torch.arange(len(s_token_ids), device=s_ids_t.device), s_ids_t
        ]
        t_lp_per_pos = t_lp_all[
            torch.arange(len(t_token_ids), device=t_ids_t.device), t_ids_t
        ]

        group_s_lps = []
        group_t_lps = []
        group_vllm_lps = []
        for s_group, t_group in zip(s_groups, t_groups):
            if self._opd_normalize_by_group:
                group_s_lps.append(
                    s_lp_per_pos[s_group].mean()
                    if s_group
                    else s_lp_per_pos.new_tensor(0.0)
                )
                group_t_lps.append(
                    t_lp_per_pos[t_group].mean()
                    if t_group
                    else t_lp_per_pos.new_tensor(0.0)
                )
                if vllm_lp_per_pos is not None:
                    group_vllm_lps.append(
                        vllm_lp_per_pos[s_group].mean()
                        if s_group
                        else vllm_lp_per_pos.new_tensor(0.0)
                    )
            else:
                group_s_lps.append(
                    s_lp_per_pos[s_group].sum()
                    if s_group
                    else s_lp_per_pos.new_tensor(0.0)
                )
                group_t_lps.append(
                    t_lp_per_pos[t_group].sum()
                    if t_group
                    else t_lp_per_pos.new_tensor(0.0)
                )
                if vllm_lp_per_pos is not None:
                    group_vllm_lps.append(
                        vllm_lp_per_pos[s_group].sum()
                        if s_group
                        else vllm_lp_per_pos.new_tensor(0.0)
                    )

        s_lps = torch.stack(group_s_lps)
        t_lps = torch.stack(group_t_lps)

        # Skew teacher group log-probs: log(alpha * exp(s_lps) + (1-alpha) * exp(t_lps))
        current_alpha = (
            self._trainer._get_current_alpha() if self._trainer is not None else 0.0
        )
        if current_alpha > 0:
            log_alpha = math.log(current_alpha)
            log_1_minus_alpha = math.log(1 - current_alpha)
            t_lps = torch.logaddexp(log_alpha + s_lps, log_1_minus_alpha + t_lps)

        reward = (t_lps - s_lps).detach()

        ### debug extremely long numbers; remove later
        # N = 10
        # if re.search(rf"\d{{{N},}}", self.student_tokenizer.decode(s_token_ids)):
        #     print(self.student_tokenizer.decode(s_token_ids))
        #     import pdb; pdb.set_trace()

        ### debug lowest reward; remove later
        # lowest_reward_indices = torch.topk(reward, k=min(5, len(s_groups)), largest=False)[1].tolist()
        # for low_reward_index in lowest_reward_indices:
        #     # print([self.student_tokenizer.decode(s_token_ids[group[0]:group[-1]+1]) for group in s_groups[:low_reward_index]])
        #     print(self.student_tokenizer.decode(s_token_ids[:s_groups[low_reward_index][0]]))
        #     print(f"-> {self.student_tokenizer.decode(s_token_ids[s_groups[low_reward_index][0]:s_groups[low_reward_index][-1]+1])}")
        #     print(f"student top-k: {self.student_tokenizer.convert_ids_to_tokens(torch.topk(s_lp_all[s_groups[low_reward_index][0]], k=20)[1])}")
        #     print(f"teacher top-k: {self.teacher_tokenizer.convert_ids_to_tokens(torch.topk(t_lp_all[t_groups[low_reward_index][0]], k=20)[1])}")
        #     print(f"reward: {reward[low_reward_index]}")
        #     print("")
        ### debug highest reward; remove later
        # highest_reward_indices = torch.topk(reward, k=min(5, len(s_groups)))[1].tolist()
        # for high_reward_index in highest_reward_indices:
        #     # print([self.student_tokenizer.decode(s_token_ids[group[0]:group[-1]+1]) for group in s_groups[:high_reward_index]])
        #     print(self.student_tokenizer.decode(s_token_ids[:s_groups[high_reward_index][0]]))
        #     print(f"-> {self.student_tokenizer.decode(s_token_ids[s_groups[high_reward_index][0]:s_groups[high_reward_index][-1]+1])}")
        #     print(f"student top-k: {self.student_tokenizer.convert_ids_to_tokens(torch.topk(s_lp_all[s_groups[high_reward_index][0]], k=20)[1])}")
        #     print(f"teacher top-k: {self.teacher_tokenizer.convert_ids_to_tokens(torch.topk(t_lp_all[t_groups[high_reward_index][0]], k=20)[1])}")
        #     print(f"reward: {reward[high_reward_index]}")
        #     print("")
        # import pdb; pdb.set_trace()
        ### debug first K indices; remove later
        # for reward_index in range(min(5, len(s_groups))):
        #     # print([self.student_tokenizer.decode(s_token_ids[group[0]:group[-1]+1]) for group in s_groups[:low_reward_index]])
        #     print(self.student_tokenizer.decode(s_token_ids[:s_groups[reward_index][0]]))
        #     print(f"-> {self.student_tokenizer.decode(s_token_ids[s_groups[reward_index][0]:s_groups[reward_index][-1]+1])}")
        #     print(f"student top-k: {self.student_tokenizer.convert_ids_to_tokens(torch.topk(s_lp_all[s_groups[reward_index][0]], k=20)[1])}")
        #     print(f"teacher top-k: {self.teacher_tokenizer.convert_ids_to_tokens(torch.topk(t_lp_all[t_groups[reward_index][0]], k=20)[1])}")
        #     print(f"reward: {reward[reward_index]}")
        #     print("")
        # import pdb; pdb.set_trace()
        # for group_index, group in enumerate(s_groups):
        #     if len(group) > 1:
        #         print(f"s_groups > 1: {group_index}, reward: {reward[group_index]}")
        #         print(self.student_tokenizer.decode(s_token_ids[:s_groups[group_index][0]]))
        #         print(f"-> {self.student_tokenizer.decode(s_token_ids[s_groups[group_index][0]:s_groups[group_index][-1]+1])}")
        #         import pdb; pdb.set_trace()
        # for group_index, group in enumerate(t_groups):
        #     if len(group) > 1:
        #         print(f"t_groups > 1: {group_index}, reward: {reward[group_index]}")
        #         print(self.teacher_tokenizer.decode(t_token_ids[:t_groups[group_index][0]]))
        #         print(f"-> {self.teacher_tokenizer.decode(t_token_ids[t_groups[group_index][0]:t_groups[group_index][-1]+1])}")
        #         import pdb; pdb.set_trace()

        ### debug reward for newlines; remove later
        # newline_indices = [index for index, token in enumerate(self.student_tokenizer.convert_ids_to_tokens(s_token_ids)) if 'Ċ' in token]
        # if len(newline_indices) > 0:
        #     print(reward[newline_indices])
        #     print(f"mean: {reward[newline_indices].mean()}")
        #     print(f"max: {reward[newline_indices].max()}")
        #     print(f"min: {reward[newline_indices].min()}")
        #     is_weights_per_pos = torch.exp(s_lp_per_pos.detach() - vllm_lp_per_pos)
        #     print(is_weights_per_pos[newline_indices])
        ### debug high IS indices; remove later
        # is_weights_per_pos = torch.exp(s_lp_per_pos.detach() - vllm_lp_per_pos)
        # high_is_indices = [index for index, is_weight in enumerate(is_weights_per_pos.tolist()) if is_weight >= 5.0]
        # if len(high_is_indices) > 0:
        #     print(reward[high_is_indices])
        #     print(f"mean: {reward[high_is_indices].mean()}")
        #     print(f"max: {reward[high_is_indices].max()}")
        #     print(f"min: {reward[high_is_indices].min()}")
        # import pdb; pdb.set_trace()
        ### debug reward for whitespace; remove later
        # whitespace_indices = [index for index, token_id in enumerate(s_token_ids) if token_id == 220]
        # if len(whitespace_indices) > 0:
        #     print(reward[whitespace_indices])
        #     print(f"mean: {reward[whitespace_indices].mean()}")
        #     print(f"max: {reward[whitespace_indices].max()}")
        #     print(f"min: {reward[whitespace_indices].min()}")

        #     is_in_group = []
        #     for whitespace_index in whitespace_indices:
        #         in_group = False
        #         for group in s_groups:
        #             if whitespace_index in group and len(group) > 1:
        #                 in_group = True
        #                 break
        #         is_in_group.append(in_group)
        #     print(is_in_group)
        #     import pdb; pdb.set_trace()

        clip_alpha = self._trainer.args.clip_alpha if self._trainer is not None else 0.0
        if clip_alpha > 0:
            reward_floor = math.log(clip_alpha) / (1 - clip_alpha)
            reward_ceiling = -reward_floor
            reward = reward.clamp(min=reward_floor, max=reward_ceiling)

        clipped_reward = reward.clone()
        s_groups_for_vis = list(s_groups)

        if self._trainer is not None and reward.numel() > 0:
            is_1to1 = [
                len(sg) == 1 and len(tg) == 1 for sg, tg in zip(s_groups, t_groups)
            ]
            reward_vals = reward.detach()
            all_stats = {
                "mean": reward_vals.mean().item(),
                "std": reward_vals.std().item() if reward_vals.numel() > 1 else 0.0,
                "min": reward_vals.min().item(),
                "max": reward_vals.max().item(),
            }
            self._opd_reward_stats_count += 1
            for k in all_stats:
                self._opd_reward_stats_accum[k] += all_stats[k]

            one_to_one_indices = [i for i, v in enumerate(is_1to1) if v]
            multi_indices = [i for i, v in enumerate(is_1to1) if not v]

            if one_to_one_indices:
                r_1to1 = reward_vals[one_to_one_indices]
                stats_1to1 = {
                    "mean": r_1to1.mean().item(),
                    "std": r_1to1.std().item() if r_1to1.numel() > 1 else 0.0,
                    "min": r_1to1.min().item(),
                    "max": r_1to1.max().item(),
                }
                self._opd_reward_1to1_stats_count += 1
                for k in stats_1to1:
                    self._opd_reward_1to1_stats_accum[k] += stats_1to1[k]

            if multi_indices:
                r_multi = reward_vals[multi_indices]
                stats_multi = {
                    "mean": r_multi.mean().item(),
                    "std": r_multi.std().item() if r_multi.numel() > 1 else 0.0,
                    "min": r_multi.min().item(),
                    "max": r_multi.max().item(),
                }
                self._opd_reward_multi_stats_count += 1
                for k in stats_multi:
                    self._opd_reward_multi_stats_accum[k] += stats_multi[k]

            # Sub-classify multi groups: 1-to-N, N-to-1, N-to-N
            idx_1toN = [
                i
                for i, (sg, tg) in enumerate(zip(s_groups, t_groups))
                if len(sg) == 1 and len(tg) > 1
            ]
            idx_Nto1 = [
                i
                for i, (sg, tg) in enumerate(zip(s_groups, t_groups))
                if len(sg) > 1 and len(tg) == 1
            ]
            idx_NtoN = [
                i
                for i, (sg, tg) in enumerate(zip(s_groups, t_groups))
                if len(sg) > 1 and len(tg) > 1
            ]

            self._opd_group_type_accum["total"] += len(s_groups)
            self._opd_group_type_accum["1to1"] += len(one_to_one_indices)
            self._opd_group_type_accum["1toN"] += len(idx_1toN)
            self._opd_group_type_accum["Nto1"] += len(idx_Nto1)
            self._opd_group_type_accum["NtoN"] += len(idx_NtoN)

            for idx_list, accum_attr, count_attr in [
                (
                    idx_1toN,
                    "_opd_reward_1toN_stats_accum",
                    "_opd_reward_1toN_stats_count",
                ),
                (
                    idx_Nto1,
                    "_opd_reward_Nto1_stats_accum",
                    "_opd_reward_Nto1_stats_count",
                ),
                (
                    idx_NtoN,
                    "_opd_reward_NtoN_stats_accum",
                    "_opd_reward_NtoN_stats_count",
                ),
            ]:
                if idx_list:
                    r_sub = reward_vals[idx_list]
                    sub_stats = {
                        "mean": r_sub.mean().item(),
                        "std": r_sub.std().item() if r_sub.numel() > 1 else 0.0,
                        "min": r_sub.min().item(),
                        "max": r_sub.max().item(),
                    }
                    setattr(self, count_attr, getattr(self, count_attr) + 1)
                    acc = getattr(self, accum_attr)
                    for k in sub_stats:
                        acc[k] += sub_stats[k]

            if self._opd_ignore_unbalanced_loss:
                balanced_indices = [
                    i
                    for i, (sg, tg) in enumerate(zip(s_groups, t_groups))
                    if len(sg) == len(tg)
                ]
                if balanced_indices:
                    keep = balanced_indices
                    s_groups = [s_groups[i] for i in keep]
                    t_groups = [t_groups[i] for i in keep]
                    s_lps = s_lps[keep]
                    t_lps = t_lps[keep]
                    reward = reward[keep]
                    if vllm_lp_per_pos is not None and group_vllm_lps:
                        group_vllm_lps = [group_vllm_lps[i] for i in keep]
                else:
                    return (_zero_grad_source.sum() * 0.0, 0)

            if self._opd_ignore_multi_loss and one_to_one_indices:
                keep = one_to_one_indices
                s_groups = [s_groups[i] for i in keep]
                t_groups = [t_groups[i] for i in keep]
                s_lps = s_lps[keep]
                t_lps = t_lps[keep]
                reward = reward[keep]
                if vllm_lp_per_pos is not None and group_vllm_lps:
                    group_vllm_lps = [group_vllm_lps[i] for i in keep]
            elif self._opd_ignore_multi_loss:
                return (_zero_grad_source.sum() * 0.0, 0)

            if getattr(self._trainer.args, "visualize_sampled_opd_reward", False):
                if self._trainer.accelerator.is_main_process:
                    display_reward = clipped_reward
                    if self._opd_ignore_multi_loss and multi_indices:
                        display_reward = clipped_reward.clone()
                        for i in multi_indices:
                            display_reward[i] = 0.0
                    self._print_reward_html(
                        s_token_ids, s_groups_for_vis, display_reward
                    )

        is_w = None
        if vllm_lp_per_pos is not None and group_vllm_lps:
            # Per-token IS weights (mirrors non-hybrid sampled_opd_loss)
            is_weights_per_pos = torch.exp(s_lp_per_pos.detach() - vllm_lp_per_pos)
            # A token whose vLLM logprob was NaN (trl's client returns None; see
            # _sampled_token_logprobs) has no IS weight: masked out below, and out of the stats.
            finite_per_pos = torch.isfinite(is_weights_per_pos)
            is_stats = _is_weight_stats(is_weights_per_pos[finite_per_pos])

            # Per-token truncation mask
            eps_lo = (
                getattr(self._trainer.args, "opd_is_epsilon_low", 0.5)
                if self._trainer
                else 0.5
            )
            eps_hi = (
                getattr(self._trainer.args, "opd_is_epsilon_high", 2.0)
                if self._trainer
                else 2.0
            )
            is_mask_per_pos = (
                finite_per_pos
                & (is_weights_per_pos >= eps_lo)
                & (is_weights_per_pos <= eps_hi)
            )
            is_stats["masked_fraction"] = 1.0 - is_mask_per_pos.float().mean().item()

            # Accumulate stats for logging
            if self._trainer is not None:
                self._opd_is_stats_count += 1
                for k in is_stats:
                    self._opd_is_stats_accum[k] += is_stats[k]

            # Compute per-group mean IS weight from surviving tokens
            group_is_weights = []
            filtered_s_lps = []
            filtered_rewards = []
            for g_idx, s_group in enumerate(s_groups):
                if not s_group:
                    continue
                group_mask = is_mask_per_pos[s_group]
                if not group_mask.any():
                    continue
                group_is_weights.append(is_weights_per_pos[s_group][group_mask].mean())
                filtered_s_lps.append(s_lps[g_idx])
                filtered_rewards.append(reward[g_idx])

            if not filtered_s_lps:
                return (_zero_grad_source.sum() * 0.0, 0)

            s_lps = torch.stack(filtered_s_lps)
            reward = torch.stack(filtered_rewards)
            is_w = torch.stack(group_is_weights)

        if s_lps.numel() == 0:
            return (_zero_grad_source.sum() * 0.0, 0)

        if self._opd_normalize_reward_with_baseline:
            return {
                "s_lps": s_lps,
                "reward": reward,
                "is_w": is_w,
                "n_groups": s_lps.numel(),
            }

        if is_w is not None:
            loss = -(s_lps * reward * is_w).mean()
        else:
            loss = -(s_lps * reward).mean()
        return (loss, s_lps.numel())

    def _print_reward_html(self, s_token_ids, s_groups, reward):
        reward_np = reward.detach().float().cpu()
        pos_max = reward_np.clamp(min=0).max().item()
        neg_min = reward_np.clamp(max=0).min().item()
        if pos_max < 1e-6:
            pos_max = 1e-6
        if neg_min > -1e-6:
            neg_min = -1e-6
        spans = []
        for g_idx, s_group in enumerate(s_groups):
            if not s_group:
                continue
            text = self.student_tokenizer.decode(
                s_token_ids[s_group[0] : s_group[-1] + 1]
            )
            val = reward_np[g_idx].item()
            if val >= 0:
                t = min(val / pos_max, 1.0)
            else:
                t = max(val / (-neg_min), -1.0)
            r = int(255 * (1 - max(0.0, t)))
            g = int(255 * (1 - abs(t)))
            b = int(255 * (1 - max(0.0, -t)))
            escaped = html.escape(text)
            spans.append(
                f'<span class="token" style="background-color: rgb({r},{g},{b})" '
                f'title="reward={reward_np[g_idx].item():.4f}">{escaped}</span>'
            )
        rmin = reward_np.min().item()
        rmax = reward_np.max().item()
        legend = (
            f'<div class="legend" style="display:flex;align-items:center;gap:4px;font-family:monospace;font-size:12px">'
            f"<span>{rmin:.2f}</span>"
            f'<div class="legend-bar" style="width:120px;height:14px;'
            f"background:linear-gradient(to right,rgb(255,0,0),rgb(255,255,255),rgb(0,0,255));"
            f'border:1px solid #ccc;border-radius:2px"></div>'
            f"<span>{rmax:.2f}</span>"
            f"</div>"
        )
        html_str = "\n<p>" + "".join(spans) + "</p>\n" + legend + "\n"
        self._reward_html_buffer.append(html_str)

    def _flush_reward_html(self, log_step):
        if not self._reward_html_buffer:
            return
        output_dir = self._trainer.args.output_dir
        config_name = os.path.basename(output_dir)
        vis_dir = os.path.join(output_dir, "vis_reward")
        os.makedirs(vis_dir, exist_ok=True)
        path = os.path.join(vis_dir, f"{config_name}_log-{log_step}.html")
        body = "\n".join(self._reward_html_buffer)
        doc = (
            "<!DOCTYPE html>\n<html>\n<head><meta charset='utf-8'>"
            "<style>.token{font-family:monospace;font-size:14px;padding:2px;white-space:pre-wrap;}</style>"
            "</head>\n<body>\n" + body + "\n</body>\n</html>"
        )
        with open(path, "w") as f:
            f.write(doc)
            f.flush()
        self._reward_html_buffer.clear()

    # --- Chunked region loss: same number, bounded memory ------------------------------------
    #
    # WHY IT EXISTS. Measured on this round's corpus by analysis/uld-align-closure.py (job
    # 1705751): student vocab 100,352, teacher vocab 248,077, so ONE supervised position costs
    # (100,352 + 248,077) x 4 B x 2 (the raw softmax and its group-merged copy) = 2.66 MiB of
    # float32 probability tensors, plus ~1.5 MiB more in the matched block. Region sizes on this
    # corpus are p50 185 tokens but p90 13,486 and max 18,867, and EVERY region of a row is held
    # in one autograd graph until backward -- so a p90 row asks for ~41 GiB of full-vocab probs
    # and ~23 GiB of matched block. With the 27B teacher resident on every rank there are ~3 GiB
    # free, and a direct measurement OOMed at :1615 on exactly this, on its first step.
    #
    # No configuration setting reaches it: uld_hybrid_unmatched_weight=0 skips the full-vocab
    # path and still leaves ~29 GiB of matched block, and max_length 8192 still leaves ~20 GiB.
    #
    # WHY IT IS EXACT. Both reductions are a sum over positions divided by a position count --
    # matched is generalized_jsd_loss(reduction="batchmean") = jsd.sum() / jsd.size(0), and
    # unmatched is F.l1_loss(reduction="sum") / n_aligned (:1695) -- and every per-position term
    # (softmax, group merge, sort, topk, tail mass) reads only its own group's positions. So the
    # region's loss decomposes over groups: accumulate the two sums chunk by chunk, divide once
    # at the end. Each chunk runs under torch.utils.checkpoint, so its tensors are freed after
    # its forward and recomputed during backward: peak memory becomes O(chunk) instead of
    # O(row), paid for with one extra forward per chunk.
    #
    # Exactness is not asserted here, it is measured: analysis/uld-chunked-exactness.py drives
    # this same method twice on identical inputs, once with uld_region_chunk_groups=0 and once
    # chunked, and compares the loss and the student-logit gradients.

    def _region_chunk_terms(
        self,
        s_logits,
        t_logits,
        idx_s,
        idx_t,
        sg,
        tg,
        student_matched_indices,
        teacher_matched_indices,
        student_unmatched_mask,
        teacher_unmatched_mask,
        want_matched,
        want_unmatched,
    ):
        """One chunk's matched and unmatched loss SUMS (no division). Runs under checkpoint.

        `idx_s`/`idx_t` gather this chunk's positions out of the region's logit slices; `sg`/`tg`
        are the chunk's alignment groups rebased onto that gathered order. The gather happens
        inside this function on purpose: done outside, the gathered [chunk, vocab] copy would be
        retained for backward, which is the memory this exists to avoid.
        """
        s_sel = s_logits[idx_s]
        t_sel = t_logits[idx_t]
        m_sum = s_logits.new_zeros(())
        u_sum = s_logits.new_zeros(())

        if want_unmatched:
            # Same tensors as :1502-1533 and the same sorted-L1 with the detached teacher tail
            # mass as :1654-1694, minus the final division by the aligned length.
            s_al = self._merge_probabilities_with_alignment_groups(
                F.softmax(s_sel / self.student_temperature, dim=-1), sg
            )
            t_al = self._merge_probabilities_with_alignment_groups(
                F.softmax(t_sel / self.teacher_temperature, dim=-1), tg
            )
            t_un = t_al[:, teacher_unmatched_mask]
            s_un = s_al[:, student_unmatched_mask]
            if t_un.size(-1) > 0 and s_un.size(-1) > 0:
                t_usize, s_usize = t_un.size(-1), s_un.size(-1)
                s_sorted = s_un.sort(dim=-1, descending=True).values
                if t_usize > s_usize:
                    t_top = t_un.topk(s_usize, dim=-1).values
                    tail = (t_un.sum() - t_top.sum()).detach()
                    u_sum = F.l1_loss(s_sorted, t_top, reduction="sum") + tail
                else:
                    t_sorted = t_un.sort(dim=-1, descending=True).values
                    if t_usize < s_usize:
                        t_sorted = F.pad(t_sorted, (0, s_usize - t_usize))
                    u_sum = F.l1_loss(s_sorted, t_sorted, reduction="sum")

        if want_matched:
            keep = [
                k for k, (a, b) in enumerate(zip(sg, tg)) if len(a) == 1 and len(b) == 1
            ]
            if keep:
                pos_s = torch.tensor(
                    [sg[k][0] for k in keep], dtype=torch.long, device=s_sel.device
                )
                pos_t = torch.tensor(
                    [tg[k][0] for k in keep], dtype=torch.long, device=t_sel.device
                )
                s_ml = (s_sel[pos_s] / self.student_temperature)[
                    :, student_matched_indices
                ]
                t_ml = (t_sel[pos_t] / self.teacher_temperature)[
                    :, teacher_matched_indices
                ]
                beta = (
                    self._trainer._get_current_beta()
                    if self._trainer is not None
                    else self.beta
                )
                alpha = (
                    self._trainer._get_current_alpha()
                    if self._trainer is not None
                    else 0.0
                )
                m_sum = CustomGOLDTrainer.generalized_jsd_loss(
                    F.softmax(s_ml, dim=-1),
                    F.softmax(t_ml, dim=-1),
                    labels=None,
                    beta=beta,
                    temperature=1.0,
                    alpha=alpha,
                    reduction="sum",
                    logits_are_probs=True,
                )
        return m_sum, u_sum

    def _region_terms_chunked(
        self,
        s_logits_slice,
        t_logits_slice,
        s_groups,
        t_groups,
        student_matched_indices,
        teacher_matched_indices,
        student_unmatched_mask,
        teacher_unmatched_mask,
        want_matched,
        want_unmatched,
    ):
        """(matched_sum, unmatched_sum, n_1to1, n_groups) over the region, chunk by chunk."""
        from torch.utils.checkpoint import checkpoint

        n_groups = len(s_groups)
        chunk = max(1, int(self._region_chunk_groups))
        m_sum = s_logits_slice.new_zeros(())
        u_sum = s_logits_slice.new_zeros(())
        n_1to1 = 0
        for lo in range(0, n_groups, chunk):
            hi = min(lo + chunk, n_groups)
            sg_raw, tg_raw = s_groups[lo:hi], t_groups[lo:hi]
            n_1to1 += sum(
                1 for a, b in zip(sg_raw, tg_raw) if len(a) == 1 and len(b) == 1
            )
            pos_s = sorted({p for g in sg_raw for p in g})
            pos_t = sorted({p for g in tg_raw for p in g})
            base_s = {p: k for k, p in enumerate(pos_s)}
            base_t = {p: k for k, p in enumerate(pos_t)}
            sg = [[base_s[p] for p in g] for g in sg_raw]
            tg = [[base_t[p] for p in g] for g in tg_raw]
            ms, us = checkpoint(
                self._region_chunk_terms,
                s_logits_slice,
                t_logits_slice,
                torch.tensor(pos_s, dtype=torch.long, device=s_logits_slice.device),
                torch.tensor(pos_t, dtype=torch.long, device=t_logits_slice.device),
                sg,
                tg,
                student_matched_indices,
                teacher_matched_indices,
                student_unmatched_mask,
                teacher_unmatched_mask,
                want_matched,
                want_unmatched,
                use_reentrant=False,
            )
            m_sum = m_sum + ms
            u_sum = u_sum + us
        return m_sum, u_sum, n_1to1, n_groups

    def _compute_base_distillation_loss(
        self,
        student_logits,
        teacher_logits,
        student_labels,
        teacher_labels,
        student_input_ids,
        teacher_input_ids,
    ):
        """The non-hybrid ULD loss: trl 0.26.2's ULDLoss._compute_distillation_loss
        (trl v0.26.2 trl/experimental/gold/gold_trainer.py:325-423), owned here, with two
        fixes to it. The rest is TRL 0.26.2's arithmetic unchanged, less one dead branch:
        TRL's loop also tested `use_hybrid_loss and _vocab_mapping` per row and called the
        hybrid loss, which _compute_distillation_loss's dispatch makes unreachable here (it
        routes exactly that case to the hybrid path), so it is gone. trl 1.x replaced the
        path with a byte-offset alignment that raises on token-ID inputs without offsets,
        which is all this trainer passes (#676), so it calls this class's
        _build_alignment_groups_from_ids and the vendored 0.26.2 merge below.

        1. OFF BY ONE (upstream). TRL reads `logits[start : start + size]`, the logits AT the
           completion positions. Nothing shifts them first: compute_loss passes the models'
           logits and the labels in input coordinates (as TRL's own compute_loss does,
           gold_trainer.py:1504-1511 in 0.26.2), and there logits[t] predicts ids[t + 1]. So
           TRL scored each completion token with the distribution for the token after it,
           and the EOS with the distribution after the end. The JSD path aligns
           logits[:, :-1] with labels[:, 1:] (compute_loss), the teacher crop keeps
           `teacher_prompt_length - 1` for exactly this reason, and the hybrid path below
           reads start-1. So does this. (trl 1.18.2's opt-in
           `uld_token_merge_strategy="bayesian"` makes the same shift; its default,
           "observed", still reads the unshifted positions.)
        2. NO ALIGNMENT, NO LOSS. When _build_alignment_groups_from_ids returns no groups,
           _merge_probabilities_with_alignment_groups returns the probabilities UNMERGED
           (trl 0.26.2's gold_trainer.py:532-533, kept in the vendored merge). The L1 below
           then compares positions that were never paired: when the two completions have
           different lengths it raises a shape RuntimeError, unless one side has a single
           position, which F.l1_loss broadcasts against every position of the other with
           only a UserWarning; when they have the same length it is a distance between
           unaligned positions. Such a row is skipped as the
           hybrid path skips it (zero loss, graph kept) and counted in
           `_alignment_unaligned`, rather than killing a long run over one row. SKIPPING IS
           NOT ALLOWED TO BECOME THE WHOLE RUN: a rate near 1 is the signature of a
           systematic text mismatch between the two renders (a think scaffold in one span
           and not the other, or a student span that runs across later turns), and then
           every row trains at exactly zero gradient while the run looks healthy. The
           trainer's ULDAlignmentGuard reduces the counts across ranks at every optimizer
           step and raises when every row of a step is unaligned, or when more than
           `uld_max_unaligned_fraction` of a window's rows are.
        """
        student_answer_index, student_answer_size = self._get_start_and_size_answers(
            student_labels
        )
        teacher_answer_index, teacher_answer_size = self._get_start_and_size_answers(
            teacher_labels
        )
        # Always False on this class (__init__); kept so the code reads as TRL's.
        if self.skip_student_eos:
            student_answer_size = [size - 1 for size in student_answer_size]
        if self.skip_teacher_eos:
            teacher_answer_size = [size - 1 for size in teacher_answer_size]

        if (
            not student_answer_size
            or not teacher_answer_size
            or max(max(student_answer_size), max(teacher_answer_size)) <= 0
        ):
            return (
                torch.zeros(1, device=student_logits.device, requires_grad=True)
                * student_logits.sum()
                * 1e-8
            )

        distillation_losses = []
        for i in range(student_logits.size(0)):
            student_start = student_answer_index[i]
            student_size = student_answer_size[i]
            teacher_start = teacher_answer_index[i]
            teacher_size = teacher_answer_size[i]

            if (
                student_size <= 0
                or teacher_size <= 0
                or self._unpredictable_start(student_start, teacher_start)
            ):
                distillation_losses.append(student_logits[i].sum() * 0.0)
                continue

            # Fix 1: logits at start-1 .. start+size-2, so probs[k] predicts token_ids[k].
            student_probs = F.softmax(
                student_logits[i, student_start - 1 : student_start + student_size - 1]
                / self.student_temperature,
                dim=-1,
            )
            teacher_probs = F.softmax(
                teacher_logits[i, teacher_start - 1 : teacher_start + teacher_size - 1]
                / self.teacher_temperature,
                dim=-1,
            )
            student_token_ids = student_input_ids[
                i, student_start : student_start + student_size
            ].tolist()
            teacher_token_ids = teacher_input_ids[
                i, teacher_start : teacher_start + teacher_size
            ].tolist()

            if self.use_extended_uld:
                s_groups, t_groups = self._build_alignment_groups_from_ids(
                    student_token_ids, teacher_token_ids
                )
                self._alignment_rows += 1
                if not s_groups:
                    # Fix 2. Counted, and refused in bulk by ULDAlignmentGuard.
                    self._alignment_unaligned += 1
                    distillation_losses.append(student_logits[i].sum() * 0.0)
                    continue
                student_aligned = self._merge_probabilities_with_alignment_groups(
                    student_probs, s_groups
                )
                teacher_aligned = self._merge_probabilities_with_alignment_groups(
                    teacher_probs, t_groups
                )
            else:
                min_length = min(len(student_token_ids), len(teacher_token_ids))
                student_aligned = student_probs[:min_length, :]
                teacher_aligned = teacher_probs[:min_length, :]

            student_sorted = student_aligned.sort(dim=-1, descending=True).values
            teacher_sorted = teacher_aligned.sort(dim=-1, descending=True).values
            width = max(student_sorted.size(-1), teacher_sorted.size(-1))
            student_sorted = F.pad(student_sorted, (0, width - student_sorted.size(-1)))
            teacher_sorted = F.pad(teacher_sorted, (0, width - teacher_sorted.size(-1)))
            aligned_loss = F.l1_loss(student_sorted, teacher_sorted, reduction="sum")
            distillation_losses.append(aligned_loss / student_aligned.size(0))

        return self.distillation_weight * torch.stack(distillation_losses).mean()

    def _unpredictable_start(self, student_start, teacher_start) -> bool:
        """Whether either completion starts at position 0; if so, count it, and the caller
        skips the row (region) with a zero loss.

        Both ULD paths read the logits at start-1 .. start+size-2, because logits[t] predict
        token t + 1. A completion at position 0 has no predictor, and start-1 = -1 would
        silently slice from the END of the row instead. Every layout this trainer builds has
        a prompt: the collator's spans start after the response template or the rendered
        prompt, the single-region teacher crop keeps `teacher_prompt_length - 1` (the
        builder's common prompt length, or, for a teacher reusing the student's input, the
        batch's earliest completion: `earliest_completion_column`), and the per-turn
        teacher pack puts each region at block offset + 1. So a completion reaches
        position 0 only when its row's own input has no prompt: a construction error, not
        a data condition to degrade over, and ULDAlignmentGuard refuses any such row at
        the optimizer step, on every rank alike. It is not raised here: this runs on one
        rank's rows, and the other ranks would wait in backward's collective until the
        NCCL timeout.
        """
        if student_start < 1 or teacher_start < 1:
            self._alignment_unpredictable += 1
            return True
        return False

    def _merge_probabilities_with_alignment_groups(
        self, probs, alignment_groups, token_ids=None
    ):
        """trl 0.26.2's merge, vendored unchanged (#676 decision D-7: bit-identical numerics).

        Copied from trl v0.26.2 trl/experimental/gold/gold_trainer.py:521. For a group of
        several positions it multiplies their distributions (sums log-probs) and renormalises
        with a softmax. trl 1.x replaced it with an unnormalised "observed" merge that needs
        ``token_ids`` and raises on a multi-token group without them; ``token_ids`` is
        accepted here only so trl's signature still fits, and ignored.
        """
        if not alignment_groups:
            return probs

        vocab_size = probs.size(-1)
        target_len = len(alignment_groups)
        aligned_probs = torch.zeros(target_len, vocab_size, device=probs.device)

        for group_idx, group in enumerate(alignment_groups):
            if len(group) > 1:
                eps = 1e-8
                logp = torch.log(probs[group[0]].clamp_min(eps))
                for idx in group[1:]:
                    if idx < probs.size(0):
                        logp = logp + torch.log(probs[idx].clamp_min(eps))
                aligned_probs[group_idx] = torch.softmax(logp, dim=-1)
            elif len(group) == 1:
                aligned_probs[group_idx] = probs[group[0]]
            else:
                aligned_probs[group_idx] = torch.zeros_like(probs[0])

        return aligned_probs

    def _compute_distillation_loss(
        self,
        student_logits,
        teacher_logits,
        student_labels,
        teacher_labels,
        student_input_ids,
        teacher_input_ids,
        student_byte_offsets=None,
        teacher_byte_offsets=None,
    ):
        # trl 1.x's ULDLoss.__call__ passes the byte offsets its own alignment needs; this
        # override aligns from the token IDs (both paths below), so they are not used.
        import os
        import sys

        _dbg = (
            os.environ.get("ULD_DEBUG") == "1"
            and int(os.environ.get("RANK", "0")) == 0
            and not getattr(self, "_dbg_done", False)
        )
        if not (self.use_hybrid_loss and self._vocab_mapping is not None):
            if _dbg:
                _ss = self._get_start_and_size_answers(student_labels)[1]
                _ts = self._get_start_and_size_answers(teacher_labels)[1]
                print(
                    f"[ULD_DEBUG] BASE path (hybrid={self.use_hybrid_loss} vocab_map={self._vocab_mapping is not None}) "
                    f"s_sizes={_ss} t_sizes={_ts} s_ids={list(student_logits.shape)} t_ids={list(teacher_logits.shape)}",
                    file=sys.stderr,
                    flush=True,
                )
                self._dbg_done = True
            return self._compute_base_distillation_loss(
                student_logits,
                teacher_logits,
                student_labels,
                teacher_labels,
                student_input_ids,
                teacher_input_ids,
            )

        # ONE ENTRY PER SUPERVISED REGION, NOT PER ROW. On a single-region row these lists
        # are exactly what _get_start_and_size_answers returned, so every arm that renders
        # one region per row computes bit-identically to before. On a row whose regions are
        # per assistant turn, the loop below now visits each turn instead of slicing one
        # span across all of them (see contiguous_label_spans for why that slice was wrong).
        # `row_of` carries which row a region belongs to, because the tensors are still
        # indexed by row.
        row_of, student_answer_index, student_answer_size = [], [], []
        teacher_answer_index, teacher_answer_size = [], []
        n_mismatch = 0
        for i in range(student_logits.size(0)):
            s_spans = contiguous_label_spans(student_labels[i], self.ignore_index)
            t_spans = contiguous_label_spans(teacher_labels[i], self.ignore_index)
            if len(s_spans) != len(t_spans):
                # Pair what CAN be paired and count the row. A raise here would kill a
                # multi-hour run over one row; silence would hide exactly the failure this
                # change exists to remove. So it degrades per row and is reported as a
                # scalar (`uld/span_mismatch_rows`), and a companion check is
                # what refuses the mismatch before any GPU is booked.
                n_mismatch += 1
            for (s_start, s_size), (t_start, t_size) in zip(s_spans, t_spans):
                row_of.append(i)
                student_answer_index.append(s_start)
                student_answer_size.append(s_size)
                teacher_answer_index.append(t_start)
                teacher_answer_size.append(t_size)
        self._uld_span_mismatch_rows = (
            getattr(self, "_uld_span_mismatch_rows", 0) + n_mismatch
        )
        self._uld_regions_seen = getattr(self, "_uld_regions_seen", 0) + len(row_of)
        self._uld_rows_seen = getattr(self, "_uld_rows_seen", 0) + student_logits.size(
            0
        )

        if _dbg:
            print(
                f"[ULD_DEBUG] HYBRID path regions={len(row_of)} rows={student_logits.size(0)} "
                f"mismatch_rows={n_mismatch} s_sizes={student_answer_size} "
                f"t_sizes={teacher_answer_size}",
                file=sys.stderr,
                flush=True,
            )
            self._dbg_done = True

        # Handle edge case where all answer sizes are 0
        if (
            max(
                max(student_answer_size, default=0), max(teacher_answer_size, default=0)
            )
            <= 0
        ):
            return (
                torch.zeros(1, device=student_logits.device, requires_grad=True)
                * student_logits.sum()
                * 1e-8
            )

        device = student_logits.device
        # `batch_size` now counts REGIONS, because that is what the loop iterates and what
        # every list it appends to is indexed by. The tensors are still indexed by
        # `row_of[k]`.
        batch_size = len(row_of)
        uld_losses = []
        uld_token_counts = []
        opd_losses = []
        opd_token_counts = []
        opd_loss_weighted_sum = 0.0
        opd_token_total = 0
        opd_deferred = []

        w_m = self.hybrid_matched_weight
        w_u = self.hybrid_unmatched_weight
        w_opd = self._hybrid_sampled_opd_weight
        use_adaptive = w_m is None
        opd_only = (not use_adaptive) and w_m == 0.0 and w_u == 0.0 and w_opd > 0

        # Chunked region loss (see _region_terms_chunked). The gate is deliberately NARROW: it
        # covers the one combination this round's ULD arms use, and RAISES on anything else
        # rather than quietly running the whole-region path. A silent fallback is exactly how the
        # zero-gradient region skip survived four arms and a week; a config that asks for
        # chunking and does not get it must fail loudly at step 0.
        chunked = self._region_chunk_groups > 0 and not opd_only
        if chunked:
            unsupported = []
            if not (self.use_extended_uld and self._align_at_bytes):
                unsupported.append(
                    "needs use_extended_uld with uld_align_at_bytes (chunks ARE alignment groups)"
                )
            if not self._renormalize_probs:
                unsupported.append("needs uld_renormalize_probs=true")
            if self._matched_top_k:
                unsupported.append(
                    "uld_matched_top_k>0 unsupported (its mask is built per position over the full matched block)"
                )
            if self._uld_ignore_multi_loss:
                unsupported.append("uld_ignore_multi_loss unsupported")
            if w_opd > 0:
                unsupported.append(
                    "uld_hybrid_sampled_opd_weight>0 unsupported (OPD reads the whole-region prob tensors)"
                )
            if use_adaptive:
                unsupported.append(
                    "adaptive weights unsupported (set uld_hybrid_matched_weight)"
                )
            if unsupported:
                raise ValueError(
                    "uld_region_chunk_groups=%d, but this config is outside the chunked path: %s"
                    % (self._region_chunk_groups, "; ".join(unsupported))
                )

        # Pre-compute vocab masks and indices (same for all samples)
        # Skipped when only OPD loss is active since matched/unmatched components are unused.
        if not opd_only:
            student_vocab_size = student_logits.size(-1)
            teacher_vocab_size = teacher_logits.size(-1)

            if self._teacher_matched_ids:
                teacher_matched_indices = torch.tensor(
                    sorted(self._teacher_matched_ids), dtype=torch.long, device=device
                )
                student_matched_indices = torch.tensor(
                    [
                        self._vocab_mapping[tid.item()]
                        for tid in teacher_matched_indices
                    ],
                    dtype=torch.long,
                    device=device,
                )
            else:
                teacher_matched_indices = torch.tensor(
                    [], dtype=torch.long, device=device
                )
                student_matched_indices = torch.tensor(
                    [], dtype=torch.long, device=device
                )

            teacher_matched_mask = torch.zeros(
                teacher_vocab_size, dtype=torch.bool, device=device
            )
            student_matched_mask = torch.zeros(
                student_vocab_size, dtype=torch.bool, device=device
            )
            if len(teacher_matched_indices) > 0:
                teacher_matched_mask[teacher_matched_indices] = True
                student_matched_mask[student_matched_indices] = True

            teacher_unmatched_mask = ~teacher_matched_mask
            student_unmatched_mask = ~student_matched_mask

        for i in range(batch_size):
            # `i` indexes the REGION; `row` indexes the batch. They coincide exactly when
            # there is one region per row, which is every arm that ran before this change.
            row = row_of[i]
            student_start = student_answer_index[i]
            student_size = student_answer_size[i]
            teacher_start = teacher_answer_index[i]
            teacher_size = teacher_answer_size[i]

            # Trim each side's own think-block scaffolding off the HEAD of the region, so the two
            # streams start at the same generated content. Done HERE, on start/size, rather than
            # on the id lists at :1437 below, so that the probability slices, the id slices and
            # the alignment groups are all consistent by construction -- trimming only the ids
            # would align groups against logits for positions the groups no longer describe.
            # Gated on the byte-aligned path because that is where two different templates are
            # being compared; see _scaffold_prefix_len for why these positions cost no
            # supervision. `min` keeps a region that is ENTIRELY scaffolding from going negative;
            # it degrades to the zero-size branch just below, which is the honest outcome for a
            # turn with no generated content.
            if self.use_extended_uld and self._align_at_bytes:
                s_drop = min(
                    self._scaffold_prefix_len(
                        self.student_tokenizer,
                        student_input_ids[
                            row, student_start : student_start + student_size
                        ].tolist(),
                    ),
                    student_size,
                )
                t_drop = min(
                    self._scaffold_prefix_len(
                        self.teacher_tokenizer,
                        teacher_input_ids[
                            row, teacher_start : teacher_start + teacher_size
                        ].tolist(),
                    ),
                    teacher_size,
                )
                student_start += s_drop
                student_size -= s_drop
                teacher_start += t_drop
                teacher_size -= t_drop
                self._uld_scaffold_tokens_trimmed = (
                    getattr(self, "_uld_scaffold_tokens_trimmed", 0) + s_drop + t_drop
                )
                if s_drop or t_drop:
                    self._uld_scaffold_regions_trimmed = (
                        getattr(self, "_uld_scaffold_regions_trimmed", 0) + 1
                    )

            # The start check before every start-1 read below (the full-vocab softmax, the
            # chunked region terms, the OPD slices). The scaffold trim above only moves a
            # start forward.
            if (
                student_size <= 0
                or teacher_size <= 0
                or self._unpredictable_start(student_start, teacher_start)
            ):
                uld_losses.append(student_logits[row].sum() * 0.0)
                uld_token_counts.append(0)
                opd_losses.append(student_logits[row].sum() * 0.0)
                opd_token_counts.append(0)
                continue

            # Logits at positions start-1 .. start+size-2: probs[k] predicts token_ids[k]
            # (includes first answer token, excludes prediction after EOS)
            # When _renormalize_probs=True and w_u==0, we skip full-vocab softmax entirely
            # (matched loss works directly on logits at matched indices).
            # `not chunked`: these are the tensors the chunked path exists to avoid materialising
            # whole-region. It builds the same ones per chunk, inside checkpoint.
            need_full_probs = (
                not opd_only
                and not chunked
                and (not self._renormalize_probs or w_u > 0 or use_adaptive)
            )
            if need_full_probs:
                student_probs = F.softmax(
                    student_logits[
                        row, student_start - 1 : student_start + student_size - 1
                    ]
                    / self.student_temperature,
                    dim=-1,
                )
                teacher_probs = F.softmax(
                    teacher_logits[
                        row, teacher_start - 1 : teacher_start + teacher_size - 1
                    ]
                    / self.teacher_temperature,
                    dim=-1,
                )
            else:
                student_probs = None
                teacher_probs = None

            student_token_ids = student_input_ids[
                row, student_start : student_start + student_size
            ].tolist()
            teacher_token_ids = teacher_input_ids[
                row, teacher_start : teacher_start + teacher_size
            ].tolist()

            if self.use_extended_uld:
                if self._align_at_bytes:
                    s_groups, t_groups = self._build_alignment_groups_at_bytes_from_ids(
                        student_token_ids, teacher_token_ids
                    )
                else:
                    s_groups, t_groups = self._build_alignment_groups_from_ids(
                        student_token_ids, teacher_token_ids
                    )

                self._alignment_rows += 1
                if not s_groups:
                    # Alignment failed entirely (e.g. full Unicode mismatch + truncation);
                    # skip this sample with a zero-gradient-preserving loss. Counted, and
                    # refused in bulk by ULDAlignmentGuard, as on the base path.
                    self._alignment_unaligned += 1
                    uld_losses.append(student_logits[row].sum() * 0.0)
                    uld_token_counts.append(0)
                    opd_losses.append(student_logits[row].sum() * 0.0)
                    opd_token_counts.append(0)
                    continue

                if need_full_probs:
                    student_aligned = self._merge_probabilities_with_alignment_groups(
                        student_probs, s_groups
                    )
                    teacher_aligned = self._merge_probabilities_with_alignment_groups(
                        teacher_probs, t_groups
                    )

                    if self._uld_ignore_multi_loss:
                        one_to_one = [
                            idx
                            for idx, (sg, tg) in enumerate(zip(s_groups, t_groups))
                            if len(sg) == 1 and len(tg) == 1
                        ]
                        if one_to_one:
                            uld_student_aligned = student_aligned[one_to_one]
                            uld_teacher_aligned = teacher_aligned[one_to_one]
                        else:
                            uld_student_aligned = None
                            uld_teacher_aligned = None
                    else:
                        uld_student_aligned = student_aligned
                        uld_teacher_aligned = teacher_aligned
                else:
                    uld_student_aligned = None
                    uld_teacher_aligned = None
            else:
                if need_full_probs:
                    min_length = min(len(student_token_ids), len(teacher_token_ids))
                    student_aligned = student_probs[:min_length, :]
                    teacher_aligned = teacher_probs[:min_length, :]
                    uld_student_aligned = student_aligned
                    uld_teacher_aligned = teacher_aligned
                else:
                    uld_student_aligned = None
                    uld_teacher_aligned = None
                s_groups, t_groups = None, None

            if not opd_only:
                # --- Matched (JSD) component ---
                matched_loss = torch.tensor(0.0, device=device)
                matched_token_count = 0
                unmatched_loss = torch.tensor(0.0, device=device)
                chunk_align_n = None

                if chunked:
                    # Both components at once, group-chunk by group-chunk under checkpoint, so a
                    # p90 row's ~41 GiB of probability tensors never coexist. Same arithmetic as
                    # the two branches below: sums here, one division at the end.
                    m_sum, u_sum, n_1to1, n_align = self._region_terms_chunked(
                        student_logits[
                            row, student_start - 1 : student_start + student_size - 1
                        ],
                        teacher_logits[
                            row, teacher_start - 1 : teacher_start + teacher_size - 1
                        ],
                        s_groups,
                        t_groups,
                        student_matched_indices,
                        teacher_matched_indices,
                        student_unmatched_mask,
                        teacher_unmatched_mask,
                        want_matched=(w_m > 0 and len(teacher_matched_indices) > 0),
                        want_unmatched=(w_u > 0),
                    )
                    if n_1to1 > 0 and w_m > 0 and len(teacher_matched_indices) > 0:
                        # generalized_jsd_loss(reduction="batchmean") is jsd.sum() / jsd.size(0),
                        # and size(0) is the number of 1-to-1 groups.
                        matched_loss = m_sum / n_1to1
                        matched_token_count = int(len(student_matched_indices))
                    if n_align > 0 and w_u > 0:
                        # The whole-region path divides by uld_student_aligned.size(0), which is
                        # the group count while uld_ignore_multi_loss is off (gated above).
                        unmatched_loss = u_sum / n_align
                    # What the whole-region path would have appended to uld_token_counts: the
                    # aligned length when it built the aligned tensors, the 1-to-1 count when it
                    # did not. Region weighting has to match or the two paths differ in the mean.
                    chunk_align_n = n_align if w_u > 0 else n_1to1
                    self._uld_region_chunks = getattr(self, "_uld_region_chunks", 0) + (
                        (n_align + self._region_chunk_groups - 1)
                        // self._region_chunk_groups
                    )

                elif (
                    self._renormalize_probs
                    and len(teacher_matched_indices) > 0
                    and (use_adaptive or w_m > 0)
                ):
                    # --- Renormalized matched loss path ---
                    # Operates on 1-to-1 alignment groups only; skips full-vocab softmax merge.
                    one_to_one = (
                        [
                            (sg[0], tg[0])
                            for sg, tg in zip(s_groups, t_groups)
                            if len(sg) == 1 and len(tg) == 1
                        ]
                        if s_groups is not None
                        else []
                    )

                    if one_to_one:
                        student_positions = [s for s, t in one_to_one]
                        teacher_positions = [t for s, t in one_to_one]

                        student_logits_slice = student_logits[
                            row, student_start - 1 : student_start + student_size - 1
                        ]
                        teacher_logits_slice = teacher_logits[
                            row, teacher_start - 1 : teacher_start + teacher_size - 1
                        ]
                        student_logits_1to1 = (
                            student_logits_slice[student_positions]
                            / self.student_temperature
                        )
                        teacher_logits_1to1 = (
                            teacher_logits_slice[teacher_positions]
                            / self.teacher_temperature
                        )

                        student_matched_logits = student_logits_1to1[
                            :, student_matched_indices
                        ]
                        teacher_matched_logits = teacher_logits_1to1[
                            :, teacher_matched_indices
                        ]

                        if self._matched_top_k > 0:
                            num_matched = teacher_matched_logits.size(-1)
                            valid_mask = self._compute_topk_matched_mask(
                                student_matched_logits, device
                            )
                            student_matched_logits = student_matched_logits.masked_fill(
                                ~valid_mask, float("-inf")
                            )
                            teacher_matched_logits = teacher_matched_logits.masked_fill(
                                ~valid_mask, float("-inf")
                            )
                            student_matched_probs = F.softmax(
                                student_matched_logits, dim=-1
                            )
                            teacher_matched_probs = F.softmax(
                                teacher_matched_logits, dim=-1
                            )
                            matched_token_count = min(self._matched_top_k, num_matched)

                            # DEBUG: print overlapping top_k/top_p token ids per position
                            # print(self.student_tokenizer.decode(student_token_ids))
                            # for pos_idx in range(valid_mask.size(0)):
                            #     pos_mask = valid_mask[pos_idx]  # [num_matched]
                            #     if pos_mask.any():
                            #         matched_positions = pos_mask.nonzero(as_tuple=True)[0]
                            #         s_ids = student_matched_indices[matched_positions].tolist()
                            #         t_ids = teacher_matched_indices[matched_positions].tolist()
                            #         print(f"[debug] pos={pos_idx} overlapping={len(s_ids)} valid={self.student_tokenizer.convert_ids_to_tokens(s_ids[:16])}")
                            #     else:
                            #         print(f"[debug] pos={pos_idx} overlapping=0")
                            # import pdb; pdb.set_trace()
                            # END DEBUG

                            if valid_mask.any():
                                matched_loss = (
                                    self._compute_jsd_loss_for_matched_tokens(
                                        student_matched_probs,
                                        teacher_matched_probs,
                                        mask=valid_mask,
                                    )
                                )
                        else:
                            student_matched_probs = F.softmax(
                                student_matched_logits, dim=-1
                            )
                            teacher_matched_probs = F.softmax(
                                teacher_matched_logits, dim=-1
                            )
                            matched_token_count = student_matched_probs.size(-1)
                            matched_loss = self._compute_jsd_loss_for_matched_tokens(
                                student_matched_probs, teacher_matched_probs
                            )

                elif (
                    uld_student_aligned is not None
                    and len(teacher_matched_indices) > 0
                    and (use_adaptive or w_m > 0)
                ):
                    # --- Standard (non-renormalized) matched loss path ---
                    teacher_matched_probs = uld_teacher_aligned[
                        :, teacher_matched_indices
                    ]
                    student_matched_probs = uld_student_aligned[
                        :, student_matched_indices
                    ]
                    matched_token_count = teacher_matched_probs.size(-1)

                    if self._matched_top_k > 0:
                        # top-k on probs gives same ordering as on logits (softmax is monotonic)
                        num_matched = teacher_matched_probs.size(-1)
                        valid_mask = self._compute_topk_matched_mask(
                            student_matched_probs, device
                        )
                        matched_token_count = min(self._matched_top_k, num_matched)

                        # DEBUG: print overlapping top_k/top_p token ids per position
                        # print(self.student_tokenizer.decode(student_token_ids))
                        # for pos_idx in range(valid_mask.size(0)):
                        #     pos_mask = valid_mask[pos_idx]  # [num_matched]
                        #     if pos_mask.any():
                        #         matched_positions = pos_mask.nonzero(as_tuple=True)[0]
                        #         s_ids = student_matched_indices[matched_positions].tolist()
                        #         t_ids = teacher_matched_indices[matched_positions].tolist()
                        #         print(f"[debug] pos={pos_idx} overlapping={len(s_ids)} valid={self.student_tokenizer.convert_ids_to_tokens(s_ids[:16])}")
                        #     else:
                        #         print(f"[debug] pos={pos_idx} overlapping=0")
                        #     import pdb; pdb.set_trace()
                        # END DEBUG

                        if valid_mask.any():
                            matched_loss = self._compute_jsd_loss_for_matched_tokens(
                                student_matched_probs,
                                teacher_matched_probs,
                                mask=valid_mask,
                            )
                    else:
                        matched_loss = self._compute_jsd_loss_for_matched_tokens(
                            student_matched_probs, teacher_matched_probs
                        )

                # --- Unmatched (sorted L1) component ---
                # `unmatched_loss` is initialised with matched_loss above, so the chunked branch's
                # value survives. uld_student_aligned is None on that path, so this block is skipped.
                if uld_student_aligned is not None and (use_adaptive or w_u > 0):
                    teacher_unmatched_probs = uld_teacher_aligned[
                        :, teacher_unmatched_mask
                    ]
                    student_unmatched_probs = uld_student_aligned[
                        :, student_unmatched_mask
                    ]

                    if (
                        teacher_unmatched_probs.size(-1) > 0
                        and student_unmatched_probs.size(-1) > 0
                    ):
                        t_usize = teacher_unmatched_probs.size(-1)
                        s_usize = student_unmatched_probs.size(-1)
                        student_unmatched_sorted = student_unmatched_probs.sort(
                            dim=-1, descending=True
                        ).values
                        if t_usize > s_usize:
                            # A direct measurement OOMed on the line below as it used to be written: it
                            # sorted the FULL teacher unmatched block (151,457 cols for the
                            # granite/qwen pair) and F.pad'd the student's own 3,489 up to match,
                            # so 97.7% of the compared columns were structural zeros. F.pad's
                            # backward is a slice, so those padded columns contribute loss VALUE
                            # but receive EXACTLY ZERO gradient. Comparing the student against the
                            # teacher's top-s_usize and adding the dropped teacher tail mass back
                            # as a detached scalar is therefore algebraically identical -- verified
                            # |dloss| = 0.000e+00 and max|dgrad| = 0.000e+00, gradient still
                            # reaching all 3,489/3,489 student columns -- while cutting the
                            # marginal chain memory at R=3072 from ~13.9 GiB to ~0.15 GiB.
                            # topk returns values already sorted descending, matching the student.
                            # Valid because teacher probs are non-negative, so each dropped tail
                            # column j contributes |0 - t_j| = t_j to the L1 sum.
                            teacher_unmatched_top = teacher_unmatched_probs.topk(
                                s_usize, dim=-1
                            ).values
                            teacher_tail_mass = (
                                teacher_unmatched_probs.sum()
                                - teacher_unmatched_top.sum()
                            ).detach()
                            unmatched_loss = (
                                F.l1_loss(
                                    student_unmatched_sorted,
                                    teacher_unmatched_top,
                                    reduction="sum",
                                )
                                + teacher_tail_mass
                            )
                        else:
                            # Student at least as wide as the teacher. Here the padding is on the
                            # TEACHER side, and the student's extra columns are real parameters
                            # carrying real gradient, so there is nothing to fold away.
                            teacher_unmatched_sorted = teacher_unmatched_probs.sort(
                                dim=-1, descending=True
                            ).values
                            if t_usize < s_usize:
                                teacher_unmatched_sorted = F.pad(
                                    teacher_unmatched_sorted, (0, s_usize - t_usize)
                                )
                            unmatched_loss = F.l1_loss(
                                student_unmatched_sorted,
                                teacher_unmatched_sorted,
                                reduction="sum",
                            )
                        unmatched_loss = unmatched_loss / uld_student_aligned.size(0)

                # --- Resolve weights ---
                if use_adaptive:
                    eff_w_m = matched_token_count / max(1, teacher_vocab_size)
                    eff_w_u = 1.0 - eff_w_m
                else:
                    eff_w_m = w_m
                    eff_w_u = w_u

                aligned_loss = student_logits[row].sum() * 0.0
                if eff_w_m > 0:
                    aligned_loss = aligned_loss + eff_w_m * matched_loss
                if eff_w_u > 0:
                    aligned_loss = aligned_loss + eff_w_u * unmatched_loss

                # Store for logging (same as parent)
                self.last_matched_loss = matched_loss
                self.last_unmatched_loss = unmatched_loss
            else:
                aligned_loss = student_logits[row].sum() * 0.0
                self.last_matched_loss = torch.tensor(0.0, device=device)
                self.last_unmatched_loss = torch.tensor(0.0, device=device)

            # --- ULD loss tracking ---
            uld_losses.append(aligned_loss)
            if uld_student_aligned is not None:
                uld_token_counts.append(uld_student_aligned.size(0))
            elif chunked and chunk_align_n is not None:
                uld_token_counts.append(chunk_align_n)
            elif self._renormalize_probs and s_groups is not None:
                n_1to1 = sum(
                    1
                    for sg, tg in zip(s_groups, t_groups)
                    if len(sg) == 1 and len(tg) == 1
                )
                uld_token_counts.append(n_1to1)
            else:
                uld_token_counts.append(0)

            # --- Sampled OPD component ---
            opd_loss_i = student_logits[row].sum() * 0.0
            opd_count_i = 0
            if w_opd > 0 and s_groups is not None and t_groups is not None:
                vllm_lp_item = None
                if self._current_vllm_logprobs is not None:
                    vllm_lp_full = self._current_vllm_logprobs[row]
                    vllm_lp_item = vllm_lp_full[
                        student_start : student_start + student_size
                    ]

                student_logits_slice = student_logits[
                    row, student_start - 1 : student_start + student_size - 1
                ]
                teacher_logits_slice = teacher_logits[
                    row, teacher_start - 1 : teacher_start + teacher_size - 1
                ]
                opd_result = self._compute_sampled_opd_for_groups(
                    student_probs,
                    teacher_probs,
                    student_token_ids,
                    teacher_token_ids,
                    s_groups,
                    t_groups,
                    vllm_lp_per_pos=vllm_lp_item,
                    student_logits_slice=student_logits_slice,
                    teacher_logits_slice=teacher_logits_slice,
                )
                if isinstance(opd_result, dict):
                    opd_deferred.append((len(opd_losses), opd_result))
                    opd_count_i = opd_result["n_groups"]
                else:
                    opd_loss_val, opd_n = opd_result
                    opd_loss_i = w_opd * opd_loss_val
                    opd_count_i = opd_n
                    opd_loss_weighted_sum = (
                        opd_loss_weighted_sum + opd_loss_val.detach() * opd_n
                    )
                    opd_token_total += opd_n
            opd_losses.append(opd_loss_i)
            opd_token_counts.append(opd_count_i)

        if opd_deferred:
            all_rewards = torch.cat([buf["reward"] for _, buf in opd_deferred])
            reward_mean = all_rewards.mean()
            reward_std = all_rewards.std()
            for sample_idx, buf in opd_deferred:
                normalized_reward = (buf["reward"] - reward_mean) / (reward_std + 1e-8)
                if buf["is_w"] is not None:
                    opd_loss = -(buf["s_lps"] * normalized_reward * buf["is_w"]).mean()
                else:
                    opd_loss = -(buf["s_lps"] * normalized_reward).mean()
                opd_losses[sample_idx] = opd_losses[sample_idx] + w_opd * opd_loss
                opd_loss_weighted_sum += opd_loss.detach() * buf["n_groups"]
                opd_token_total += buf["n_groups"]
            self._opd_batch_norm_stats_count += 1
            self._opd_batch_norm_stats_accum["mean"] += reward_mean.item()
            self._opd_batch_norm_stats_accum["std"] += reward_std.item()

        def _weighted_avg(losses, counts):
            total = sum(counts)
            if total == 0:
                return torch.stack(losses).mean()
            return sum(l * c for l, c in zip(losses, counts)) / total

        distillation_loss = _weighted_avg(uld_losses, uld_token_counts) + _weighted_avg(
            opd_losses, opd_token_counts
        )
        self.last_sampled_opd_loss = (
            (opd_loss_weighted_sum / opd_token_total) if opd_token_total > 0 else None
        )
        return self.distillation_weight * distillation_loss

    def _compute_topk_matched_mask(self, matched_logits, device):
        """Compute per-position mask selecting student's top-k over the matched vocabulary.

        Args:
            matched_logits: [seq_len, num_matched] student logits (or probs) at matched indices.

        Returns:
            valid_mask: [seq_len, num_matched] boolean tensor with exactly top_k True per row.
        """
        top_k = self._matched_top_k
        seq_len, num_matched = matched_logits.shape
        k = min(top_k, num_matched)

        topk_indices = matched_logits.topk(k, dim=-1).indices  # [seq_len, k]

        valid_mask = torch.zeros(seq_len, num_matched, dtype=torch.bool, device=device)
        valid_mask.scatter_(1, topk_indices, True)

        return valid_mask

    def _compute_jsd_loss_for_matched_tokens(
        self, student_logits, teacher_logits, mask=None
    ):
        batch_seq_len, num_matched = student_logits.shape
        student_logits_reshaped = student_logits.view(-1, num_matched)
        teacher_logits_reshaped = teacher_logits.view(-1, num_matched)

        if self._trainer is not None:
            current_beta = self._trainer._get_current_beta()
            current_alpha = self._trainer._get_current_alpha()
        else:
            current_beta = self.beta
            current_alpha = 0.0

        if mask is not None:
            jsd = CustomGOLDTrainer.generalized_jsd_loss(
                student_logits_reshaped,
                teacher_logits_reshaped,
                labels=None,
                beta=current_beta,
                temperature=1.0,
                alpha=current_alpha,
                reduction="none",
                logits_are_probs=True,
            )
            masked_jsd = jsd * mask.float()
            n_valid_positions = mask.any(dim=-1).sum().clamp(min=1)
            return masked_jsd.sum() / n_valid_positions
        else:
            jsd_loss = CustomGOLDTrainer.generalized_jsd_loss(
                student_logits_reshaped,
                teacher_logits_reshaped,
                labels=None,
                beta=current_beta,
                temperature=1.0,
                alpha=current_alpha,
                reduction="batchmean",
                logits_are_probs=True,
            )
            return jsd_loss


class _StopAtStepCallback(TrainerCallback):
    """Save and stop at a fixed optimizer step without changing max_steps.

    max_steps is what the LR scheduler is built from, so ending a run early by lowering it
    reshapes the whole schedule -- and on a resume, the restored scheduler state is then read
    against a different curve. This leaves the schedule alone and ends the loop instead.
    """

    def __init__(self, stop_at_step: int):
        self.stop_at_step = stop_at_step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_at_step:
            control.should_save = True
            control.should_training_stop = True
        return control


def _all_reduce_sum(values, device, distributed: bool) -> list[float]:
    """Sum `values` over every rank (float64), or return them as they are on one process.

    Every rank must call this the same number of times, so callers gate it on configuration
    only, never on the values: a rank that skipped it would hang the others.
    """
    vec = torch.tensor([float(v) for v in values], dtype=torch.float64, device=device)
    if distributed and dist.is_available() and dist.is_initialized():
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)
    return vec.tolist()


def _guard_reduce(accelerator):
    """ULDAlignmentGuard's `reduce` for this accelerator: _all_reduce_sum over its ranks,
    or the values as they are when it is not distributed."""
    distributed = (
        getattr(accelerator, "distributed_type", DistributedType.NO)
        != DistributedType.NO
    )
    device = getattr(accelerator, "device", torch.device("cpu"))
    return lambda values: _all_reduce_sum(values, device, distributed)


class ULDAlignmentGuard:
    """Refuse to let skipped ULD rows become a run that trains on nothing.

    CustomULDLoss skips a row (a region, on the hybrid path) whose alignment yields no groups:
    zero loss, graph kept. That is the right answer for the rare row, and the WRONG answer
    when the two renders disagree systematically, because then every row is skipped, every
    step has exactly zero gradient, and the run checkpoints and reports success. That is not
    hypothetical here: a think scaffold in the student's span and not the teacher's, or a
    student span that runs across later turns, leaves every row unaligned (custom_gold_config's
    `uld_per_turn_regions` help records a run that trained 100 steps at loss 0.0 that way).

    So the counts are checked, after being summed over every rank (a rank-local count is
    wrong on N GPUs, and a rank-local raise would hang the other ranks):

    - at every optimizer step: if any row (region), on any rank, was skipped because a
      completion starts at position 0 (`_alignment_unpredictable`), raise. That is a
      construction error, not a rate to tolerate; it is counted rather than raised in the
      loss so that every rank raises here together. Rows counted after the last step (an
      evaluation) are reduced and refused at train end (`end_train`).
    - at every optimizer step: if every row that reached alignment, on every rank, was
      unaligned, raise. No loss term was computed for the step at all.
    - over each window of `window_steps` optimizer steps (default: the logging window's
      length): if more than `max_unaligned_fraction` of the window's rows were unaligned,
      raise. The final, partial window is judged at train end (`end_train`), so a run
      shorter than one window, and the steps after the last full one, are judged too.
    - for logging: the unaligned count, row count and rate since the last log line, as
      `train/alignment_unaligned_rows`, `train/alignment_rows` and
      `train/alignment_unaligned_rate`, the same on every rank. The window and the log line
      count separately. With the default window they cover the same steps when logging runs
      every logging_steps from step 1, but not, for instance, on a resumed run whose first
      step is off the logging grid.

    THERE IS NO MINIMUM ROW COUNT, deliberately. With a global batch of 1 (one GPU, batch 1,
    no accumulation), a single unaligned row is an all-unaligned step and raises. But that
    step computed no loss at all, so the premise of skipping (the other rows of the step
    still train) does not hold. And a minimum would blind the guard on exactly the small
    runs that most need it: the 2-step, 2-rank build test has 4 rows in total. A run that
    must tolerate a rare unaligned row should have more than one row per optimizer step
    (gradient_accumulation_steps).

    Rows with no completion on either side never reach alignment and are not counted here.
    """

    def __init__(self, max_unaligned_fraction: float, window_steps: int = 0):
        self.max_unaligned_fraction = float(max_unaligned_fraction)
        self.window_steps = int(window_steps)
        self._win_unaligned = 0
        self._win_rows = 0
        self._win_steps = 0
        self._last_step = 0
        self._log_unaligned = 0
        self._log_rows = 0

    def end_step(self, loss_fn, step: int, reduce, window_steps: int = 0) -> None:
        """Account for one optimizer step: drain `loss_fn`'s rank-local counters, reduce them
        with `reduce` (a list -> list sum over ranks), check, and accumulate.

        `window_steps` is the window to use when the guard was built with 0 (the trainer
        passes its logging window). Raises RuntimeError, on every rank alike, as above.
        """
        local = [
            loss_fn._alignment_unaligned,
            loss_fn._alignment_rows,
            loss_fn._alignment_unpredictable,
        ]
        loss_fn._alignment_unaligned = 0
        loss_fn._alignment_rows = 0
        loss_fn._alignment_unpredictable = 0
        unaligned, rows, unpredictable = (int(round(v)) for v in reduce(local))
        self._refuse_unpredictable(unpredictable, f"ULD step {step}")
        self._log_unaligned += unaligned
        self._log_rows += rows
        if rows > 0 and unaligned == rows:
            raise RuntimeError(
                f"ULD step {step}: all {rows} rows that reached alignment, across all ranks, "
                "yielded no alignment groups, so this step computed no loss at all. The "
                "student's and the teacher's completion texts do not line up. Known causes: "
                "last_message_only=false, under which the student's span covers every "
                "assistant turn while the teacher renders only the last; and a "
                "response_template that leaves template scaffolding (e.g. granite-4.2's "
                "'\\n<think></think>') in the student's labels but not in the teacher's "
                "completion. Set ULD_DEBUG=1 for the spans of the first batch."
            )
        self._win_unaligned += unaligned
        self._win_rows += rows
        self._win_steps += 1
        self._last_step = step
        window = max(1, int(self.window_steps or window_steps or 1))
        if self._win_steps >= window:
            self._judge_window()

    def end_train(self, loss_fn=None, reduce=None) -> None:
        """Judge what no optimizer step has: the position-0 completions counted since the
        last one (an evaluation after it), and the final, partial window, if training
        ended inside one.

        With `loss_fn`, its `_alignment_unpredictable` is drained and summed over ranks with
        `reduce`, as in end_step. The trainer calls this on every rank, once, at train end,
        so that reduction is rank-uniform; it runs before either check can raise. The
        window's counts are already summed over ranks. Raises on every rank alike."""
        if loss_fn is not None:
            local = [loss_fn._alignment_unpredictable]
            loss_fn._alignment_unpredictable = 0
            (unpredictable,) = (int(round(v)) for v in reduce(local))
            self._refuse_unpredictable(unpredictable, "ULD, at train end")
        if self._win_steps:
            self._judge_window()

    @staticmethod
    def _refuse_unpredictable(unpredictable: int, where: str) -> None:
        if unpredictable > 0:
            raise RuntimeError(
                f"{where}: {unpredictable} rows (regions, on the hybrid path) across all "
                "ranks have a completion at position 0, so no logit predicts their first "
                "token; they were skipped. Every layout the trainer builds puts a prompt "
                "before the completion, so these inputs were built wrongly (a row with no "
                "prompt, for one). Rows evaluated since the previous step count too."
            )

    def _judge_window(self) -> None:
        win_unaligned, win_rows = self._win_unaligned, self._win_rows
        first, last = self._last_step - self._win_steps + 1, self._last_step
        self._win_unaligned = self._win_rows = self._win_steps = 0
        if win_rows > 0 and win_unaligned / win_rows > self.max_unaligned_fraction:
            raise RuntimeError(
                f"ULD steps {first}-{last}: {win_unaligned} of {win_rows} rows "
                f"({win_unaligned / win_rows:.1%}) yielded no alignment groups and "
                "contributed no loss, more than uld_max_unaligned_fraction="
                f"{self.max_unaligned_fraction}. A rare unaligned row is skipped; this many "
                "means the student's and the teacher's completions do not line up for a "
                "whole class of rows (multi-turn rows under last_message_only=false are the "
                "known one)."
            )

    def pop_log(self) -> dict:
        """The counts since the last call, for the log line, and reset them. Empty when no
        row reached alignment (an arm that does not align, or no step since)."""
        if self._log_rows <= 0:
            return {}
        out = {
            "train/alignment_unaligned_rows": self._log_unaligned,
            "train/alignment_rows": self._log_rows,
            "train/alignment_unaligned_rate": self._log_unaligned / self._log_rows,
        }
        self._log_unaligned = self._log_rows = 0
        return out


class _ULDAlignmentGuardCallback(TrainerCallback):
    """Run the trainer's ULDAlignmentGuard once per optimizer step, on every rank.

    on_pre_optimizer_step, not on_step_end, so a step whose rows were all skipped is refused
    before the optimizer applies it. (Under DeepSpeed, accelerate's backward has already
    run engine.step() at the accumulation boundary, so there the refusal follows that one
    update.) It is called on every rank at every optimizer step, which is what keeps the
    guard's all-reduce rank-uniform.
    """

    def __init__(self, trainer):
        self._trainer = trainer

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self._trainer._uld_alignment_step_end(state)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        # The steps after the last full window, or a run shorter than one, and an
        # evaluation's position-0 rows after the last optimizer step.
        self._trainer._uld_alignment_train_end()
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        # Rows the loss counted during evaluation are not a training step's; the training
        # counters were drained at the step that preceded the evaluation. Not
        # `_alignment_unpredictable`: a completion at position 0 is refused wherever it
        # came from, so an evaluated one is left for the guard to refuse at the next step,
        # or at train end after the last one.
        loss_fn = self._trainer.uld_loss_fn
        loss_fn._alignment_unaligned = 0
        loss_fn._alignment_rows = 0
        return control


class _VLLMWeightSyncCallback(TrainerCallback):
    """Push the student's weights to vLLM at the end of every vllm_sync_frequency-th step.

    trl 0.26.2's GOLDVLLMSyncCallback (trl/experimental/gold/gold_trainer.py:704), which trl 1.x
    removed when it moved the sync into its buffered _prepare_inputs. This trainer bypasses that
    path (D-1), so without this the server would keep generating from the initial weights.
    Every rank runs it: the ZeRO-3 gather inside the sync is a collective.
    """

    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        """On a resume the server still holds the initial checkpoint: push the restored
        weights before the first rollout (trl 1.x also syncs before its first generation).
        """
        t = self.trainer
        if state.global_step > 0:
            t._move_model_to_vllm()
            t._last_vllm_sync_step = state.global_step

    def on_step_end(self, args, state, control, **kwargs):
        t = self.trainer
        if (
            state.global_step != t._last_vllm_sync_step
            and state.global_step % t.vllm_sync_frequency == 0
            and getattr(t.accelerator, "sync_gradients", True)
        ):
            t._move_model_to_vllm()
            t._last_vllm_sync_step = state.global_step


class CustomGOLDTrainer(GOLDTrainer):
    """Custom trainer extending GOLDTrainer"""

    def _set_signature_columns_if_needed(self):
        """Keep `args.length_column_name` past column pruning, so the length-grouped sampler works.

        WHY THIS OVERRIDE EXISTS. `remove_unused_columns` is True for this run, and upstream's
        pruning is an ALLOWLIST, not a signature check: GOLDTrainer._set_signature_columns_if_needed
        (trl/experimental/gold/gold_trainer.py:988) appends the seven columns its collator needs and
        nothing else. Meanwhile Trainer._get_dataloader prunes the dataset BEFORE it builds the
        sampler, and _get_train_sampler then asks `length_column_name in train_dataset.column_names`
        (transformers/trainer.py:1014-1019).

        So a precomputed length column is pruned away between being written and being read, and the
        failure is SILENT: the sampler falls through to `lengths=None` and re-derives every length by
        materializing each row (trainer_pt_utils.py:532-539). Same ordering, but a full pass over the
        corpus first. Setting train_sampling_strategy="group_by_length" in a yaml and nothing else
        would look like it worked.

        Appending unconditionally is safe whether or not the column exists -- _remove_unused_columns
        intersects the allowlist with the actual columns (transformers/trainer.py:1055).
        """
        super()._set_signature_columns_if_needed()
        col = getattr(self.args, "length_column_name", None)
        if (
            col
            and self._signature_columns is not None
            and col not in self._signature_columns
        ):
            self._signature_columns.append(col)

    def _get_train_sampler(self, *args, **kwargs):
        """Say out loud whether length-grouped batching actually got its lengths.

        WHY THIS IS A GUARD AND NOT A DEBUG PRINT. The docstring above warns that setting
        train_sampling_strategy="group_by_length" "would look like it worked" if the length column
        is pruned -- transformers falls through to `lengths=None` and re-derives every length with
        a plain list comprehension over the dataset (trainer_pt_utils.py:539). It emits no warning
        and no progress bar. MEASURED cost of that fallback: on the 802,027-row deliverable corpus
        it added **at least 830 s** of host-side work before step 1 with every GPU at 0%, against a
        724 s baseline startup, confirmed directly (killed at 1659 s having produced no step). On a
        2,000-row corpus the same config starts in 200 s, confirmed by a separate direct measurement,
        i.e. the cost scales with
        row count -- which is what identified it as a per-row pass rather than a fixed setup cost.
        Length grouping is supposed to SAVE 15-21% of step time; paying a silent serial corpus pass
        per launch loses more than that on any run under a few hundred steps.

        So the choice is reported, once, before it costs anything: which sampler was selected, and
        if it is the length-grouped one, whether the lengths came from the column or from the
        fallback. A run that is about to pay the fallback now says so in its first seconds.
        """
        strategy = getattr(self.args, "train_sampling_strategy", "random")
        if strategy == "group_by_length":
            col = getattr(self.args, "length_column_name", None)
            ds = (
                kwargs.get("train_dataset")
                or (args[0] if args else None)
                or self.train_dataset
            )
            cols = list(getattr(ds, "column_names", None) or [])
            # The same two conditions transformers checks at trainer.py:1014-1019, in the same
            # order, so this reports the branch that will actually be taken rather than a guess.
            is_hf = type(ds).__module__.startswith("datasets.")
            have = bool(col) and col in cols
            if is_hf and have:
                print(
                    f"[sampler] group_by_length: lengths from column {col!r} -- no extra pass",
                    flush=True,
                )
            else:
                why = (
                    "dataset is not a datasets.Dataset "
                    f"(got {type(ds).__module__}.{type(ds).__name__})"
                    if not is_hf
                    else f"column {col!r} absent; dataset has {len(cols)} columns: {sorted(cols)}"
                )
                # flush=True is load-bearing, not tidiness. Under `accelerate launch` with the
                # launcher's output redirected, stdout is BLOCK-buffered, so this line sits in a
                # buffer until something else fills it -- which for a serial per-row pass is after
                # the stall it exists to warn about. A direct measurement spent 57 minutes in dataset
                # preparation and this guard printed NOTHING the whole time, which is how a warning
                # about an hour-long stall becomes useless.
                print(
                    f"[sampler] WARNING group_by_length will RE-DERIVE every length "
                    f"({len(ds) if hasattr(ds,'__len__') else '?'} rows, serial, no progress "
                    f"output) because {why}. Expect a long GPU-idle stall before step 1.",
                    flush=True,
                )
        else:
            print(f"[sampler] train_sampling_strategy={strategy!r}", flush=True)
        # Skip GOLDTrainer's: trl 1.x returns a shuffled RepeatSampler there, which ignores
        # train_sampling_strategy / length_column_name (trl/experimental/gold/gold_trainer.py
        # _get_train_sampler). This trainer keeps the Trainer's sampler (decision D-1).
        return super(GOLDTrainer, self)._get_train_sampler(*args, **kwargs)

    # THE DATA AND GENERATION PATH STAYS THIS TRAINER'S (#676 decision D-1). trl 1.x's GOLDTrainer
    # generates inside _prepare_inputs from a buffer that spans a gradient-accumulation window
    # (fed by get_train_dataloader / get_batch_samples), drawing on-policy once per window.
    # This trainer draws lambda per micro-step in training_step, keeps group_by_length and
    # overlap_generation, and generates there. So the three hooks below skip GOLDTrainer and
    # return to SFTTrainer/Trainer behaviour; with them, GOLDTrainer's buffer is never filled.

    def _prepare_inputs(self, inputs):
        """The batch as Trainer prepares it (device placement), never GOLD's buffered slice."""
        return super(GOLDTrainer, self)._prepare_inputs(inputs)

    def get_train_dataloader(self):
        """Trainer's dataloader: one micro-batch per step, not one generation batch per window."""
        return super(GOLDTrainer, self).get_train_dataloader()

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        """Trainer's batch fetch, so num_items_in_batch counts what this trainer trains on."""
        return super(GOLDTrainer, self).get_batch_samples(
            epoch_iterator, num_batches, device
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self._last_log_time = time.time()
        if int(getattr(self.args, "stop_at_step", 0) or 0) > 0:
            self.add_callback(_StopAtStepCallback(int(self.args.stop_at_step)))

        # GENERATION EFFICIENCY COUNTERS, reset at every log emit like the t_* timers beside them.
        #
        # WHY THESE EXIST. `t_gen` said generation was expensive; it could not say WHY, and the
        # answer had to be reconstructed from a vLLM log by hand: `Processed prompts: 1/1` at 134
        # output tok/s, 910 of 1,350 requests sitting at exactly the 8192-token cap. That is the
        # whole diagnosis of the mb1 x --data-parallel-size 8 waste, and none of it was in ClearML.
        # These five counters put it there, so the next arm's cost model is read off a curve.
        #
        # RANK-LOCAL AND NOT REDUCED, deliberately -- exactly like _timing_generation above. `log()`
        # emits on rank 0 only, so these describe rank 0's share. That is the right scale for
        # n_gen_batch (a global count, identical on every rank because it is the gathered list's
        # length) and a per-rank one for the token counters. Adding an all_reduce to make them
        # global would put a new collective in the on-policy path, which is where a direct measurement's
        # rank-divergent guard cost 2,081 s for zero steps; a per-rank number that is honestly
        # labelled beats a global one that risks that.
        self._gen_calls = 0
        self._gen_batch_total = 0  # sum over calls of the prompts the SERVER saw
        self._gen_prompt_rows = 0  # this rank's prompts, i.e. sum of micro-batch sizes
        self._gen_tokens = 0  # completion tokens decoded for this rank
        self._gen_truncated = 0  # completions that hit max_completion_length exactly

        reinit_sinks_to = getattr(self.args, "reinit_sinks_to", None)
        if reinit_sinks_to is not None:
            unwrapped_student = self.accelerator.unwrap_model(self.model)
            inner = getattr(unwrapped_student, "model", unwrapped_student)
            layers = getattr(inner, "layers", None)
            if layers is None:
                raise ValueError(
                    "reinit_sinks_to is set but the student model has no `.model.layers` "
                    "attribute; cannot locate attention sinks."
                )

            deepspeed_plugin = self.accelerator.state.deepspeed_plugin
            zero_stage_3 = (
                deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
            )
            if zero_stage_3:
                import deepspeed

                gather_ctx = lambda params: deepspeed.zero.GatheredParameters(
                    params, modifier_rank=0
                )
            else:
                gather_ctx = lambda params: nullcontext()

            sink_params = [
                layer.self_attn.sinks
                for layer in layers
                if getattr(getattr(layer, "self_attn", None), "sinks", None) is not None
            ]
            n_reinit = 0
            with gather_ctx(sink_params):
                if (not zero_stage_3) or self.accelerator.is_main_process:
                    with torch.no_grad():
                        for sinks in sink_params:
                            sinks.fill_(float(reinit_sinks_to))
                            n_reinit += 1
                else:
                    n_reinit = len(sink_params)

            if self.accelerator.is_main_process:
                observed = (
                    sink_params[0].detach().float().mean().item()
                    if sink_params
                    else float("nan")
                )
                print(
                    f"[reinit_sinks_to={reinit_sinks_to}] Reinitialized "
                    f"{n_reinit} attention-sink parameter tensors "
                    f"(layer 0 mean={observed:.4f})."
                )

        # Step-phase timing accumulators (reset every log interval)
        self._timing_generation = 0.0
        self._timing_generation_last_call = 0.0
        self._timing_training_step_total = 0.0
        self._timing_weight_sync = 0.0
        self._timing_weight_sync_allgather = 0.0
        self._timing_weight_sync_broadcast = 0.0
        self._timing_weight_sync_params = 0

        # New: overlap-mode breakdown (rank 0 captures gen_actual/gen_wait via future timestamps)
        self._timing_generation_actual = 0.0  # async gen wall-clock (submit → done)
        self._timing_generation_wait = 0.0  # main-thread block on .result()
        self._timing_training_pure = 0.0  # super().training_step() only

        # Overlap generation state
        self._overlap_buffer = None
        self._overlap_executor = None
        if self.args.overlap_generation:
            from concurrent.futures import ThreadPoolExecutor

            self._overlap_executor = ThreadPoolExecutor(max_workers=1)

        # trl 1.x keeps its vLLM state on self.vllm_generation (trl/generation/vllm_generation.py),
        # which GOLDTrainer.__init__ built: in server mode it created the client and joined the
        # weight-transfer group with the server at vllm_server_host/port (or vllm_server_base_url).
        # The names below are what this trainer's methods read.
        self.vllm_mode = self.args.vllm_mode if self.use_vllm else None
        self.vllm_client = None
        if self.use_vllm:
            self._configure_vllm_generation()
            self.vllm_client = getattr(self.vllm_generation, "vllm_client", None)
            # trl 0.26.2 synced weights from a GOLDVLLMSyncCallback; trl 1.x syncs inside its
            # buffered _prepare_inputs, which this trainer bypasses (D-1). Same callback, so
            # the server sees the trained weights every vllm_sync_frequency steps as before.
            self.add_callback(_VLLMWeightSyncCallback(self))

        # More servers than the one GOLDTrainer connected to: add a client per extra server and
        # put the set where VLLMGeneration reads it, so generation and weight sync reach all.
        if (
            self.use_vllm
            and self.vllm_mode == "server"
            and self.accelerator.is_main_process
            and self.args.vllm_num_servers > 1
        ):
            hosts_str = os.environ.get("VLLM_SERVER_HOSTS", "")
            ports_str = os.environ.get("VLLM_SERVER_PORTS", "")
            nccl_ports_str = os.environ.get("VLLM_NCCL_COORDINATOR_PORTS", "")
            if not hosts_str or not ports_str or not nccl_ports_str:
                raise ValueError(
                    "VLLM_SERVER_HOSTS, VLLM_SERVER_PORTS, and VLLM_NCCL_COORDINATOR_PORTS "
                    "environment variables must be set when vllm_num_servers > 1."
                )
            hosts = hosts_str.split(",")
            ports = [int(p) for p in ports_str.split(",")]
            nccl_ports = [int(p) for p in nccl_ports_str.split(",")]
            if len(hosts) != self.args.vllm_num_servers:
                raise ValueError(
                    f"Expected {self.args.vllm_num_servers} hosts in VLLM_SERVER_HOSTS, got {len(hosts)}."
                )
            first = self.vllm_client
            servers = list(zip(hosts, ports, nccl_ports))
            first_url = first.base_url.rstrip("/")
            extra = [(h, p, n) for h, p, n in servers if f"http://{h}:{p}" != first_url]
            if len(extra) != len(servers) - 1:
                raise ValueError(
                    f"vllm_server_host/port ({first_url}) must be one of the servers in "
                    f"VLLM_SERVER_HOSTS/PORTS ({servers}); it is the one trl already joined."
                )
            first_nccl = [n for h, p, n in servers if f"http://{h}:{p}" == first_url][0]
            if first_nccl != self.args.vllm_group_port:
                # trl joined that server's weight-transfer group on vllm_group_port; a
                # different port listed for it would be silently ignored.
                raise ValueError(
                    f"VLLM_NCCL_COORDINATOR_PORTS lists {first_nccl} for {first_url}, but trl "
                    f"joined it on vllm_group_port={self.args.vllm_group_port}; set them equal."
                )
            clients = [first]
            for host, port, nccl_port in extra:
                client = VLLMClient(
                    host=host,
                    server_port=port,
                    group_port=nccl_port,
                    connection_timeout=self.args.vllm_server_timeout,
                )
                client.init_communicator(device=self.accelerator.device)
                clients.append(client)
            self.vllm_client = MultiVLLMClient(clients)
            self.vllm_generation.vllm_client = self.vllm_client
            print(f"MultiVLLMClient initialized with {len(clients)} servers: {servers}")

        if (
            self.use_uld_loss
            and self.uld_loss_fn is not None
            and not self.args.use_old_uld_loss
        ):
            self.uld_loss_fn = CustomULDLoss(
                config=self.args,
                student_tokenizer=self.processing_class,
                teacher_tokenizer=self.teacher_tokenizer,
                trainer=self,
            )
            if self.args.uld_reuse_student_input and self.uld_loss_fn.jaccard != 1.0:
                raise ValueError(
                    f"uld_reuse_student_input requires identical tokenizers (jaccard=1.0), "
                    f"but got jaccard={self.uld_loss_fn.jaccard}"
                )

        # Gated on configuration alone (the loss class), so every rank registers it and the
        # guard's per-step all-reduce runs on every rank alike.
        self._uld_alignment_guard = None
        if self.use_uld_loss and isinstance(
            getattr(self, "uld_loss_fn", None), CustomULDLoss
        ):
            self._uld_alignment_guard = ULDAlignmentGuard(
                self.args.uld_max_unaligned_fraction,
                self.args.uld_unaligned_window_steps,
            )
            self.add_callback(_ULDAlignmentGuardCallback(self))

        if self.use_uld_loss and self.teacher_tokenizer is not None:
            _s = [{"role": "user", "content": ""}]
            _prompt = self.teacher_tokenizer.apply_chat_template(
                _s, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            _full = self.teacher_tokenizer.apply_chat_template(
                _s + [{"role": "assistant", "content": ""}],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            self._teacher_turn_suffix = _full[len(_prompt) :]
        else:
            self._teacher_turn_suffix = ""

        # Parsed ONCE, and echoed, because a mis-spelled template variable is silently ignored
        # by every jinja template: a typo here would look exactly like the flag not working.
        self._uld_teacher_template_kwargs = parse_template_kwargs(
            getattr(self.args, "uld_teacher_template_kwargs", None)
        )
        if self._uld_teacher_template_kwargs:
            print(
                f"[uld] per-turn teacher template kwargs: "
                f"{self._uld_teacher_template_kwargs}",
                flush=True,
            )

        # trl 0.26.2's GOLDTrainer set use_liger_gkd_loss from use_liger_kernel; trl 1.x's has
        # no liger path, so the flag is set here from both switches, as before.
        self.use_liger_gkd_loss = bool(
            self.args.use_liger_fused_jsd
            or getattr(self.args, "use_liger_kernel", False)
        )

        if self.use_liger_gkd_loss:
            from gb_steps_post_training.distillation.liger_losses import (
                LigerFusedLinearSkewedJSDLoss,
            )

            self.liger_jsd_loss = LigerFusedLinearSkewedJSDLoss(
                weight_hard_loss=0.0,
                weight_soft_loss=1.0,
                beta=self.args.beta,
                alpha=self.args.alpha,
                ignore_index=-100,
                temperature=self.args.temperature,
                compiled=False,
                use_kl_interpolation=self.args.use_kl_interpolation,
            )

        # Hidden state matching loss setup
        if self.args.use_hidden_loss:
            # Infer hidden dimensions from models
            unwrapped_student = self.accelerator.unwrap_model(self.model)
            unwrapped_teacher = self.accelerator.unwrap_model(self.teacher_model)
            student_dim = unwrapped_student.config.hidden_size
            teacher_dim = unwrapped_teacher.config.hidden_size

            # Infer dtype from student model parameters
            model_dtype = next(unwrapped_student.parameters()).dtype

            # Create projection: teacher_dim -> student_dim. Skip when widths
            # already match (avoids a random matmul that would scramble the
            # cosine geometry and adds no value).
            if student_dim != teacher_dim:
                self.hidden_proj = nn.Linear(teacher_dim, student_dim, bias=False).to(
                    device=self.accelerator.device, dtype=model_dtype
                )
            else:
                self.hidden_proj = None

            # Default to last layer if not specified
            num_layers = unwrapped_student.config.num_hidden_layers
            if self.args.hidden_loss_layers is None:
                self.args.hidden_loss_layers = [num_layers - 1]

            # Note: the liger path always operates on the last decoder layer
            # (captured via the lm_head pre-hook); hidden_loss_layers only
            # governs the standard branch.

            # Accumulators for logging
            self._hidden_loss_total = 0.0
            self._hidden_loss_step_equiv = 0.0

        self._sampled_opd_sum = 0.0
        self._sampled_opd_step_eq = 0.0

        if self.args.opd_importance_sampling:
            self._is_stats_accum = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "masked_fraction": 0.0,
            }
            self._is_stats_count = 0

            if self.args.use_uld_loss:
                from transformers.convert_slow_tokenizer import bytes_to_unicode

                self._unicode_to_byte = {v: k for k, v in bytes_to_unicode().items()}

    def _configure_vllm_generation(self) -> None:
        """Set the sampling parameters trl's VLLMGeneration sends to this trainer's values.

        GOLDTrainer builds VLLMGeneration from trl's fields (temperature, top_k, logprobs=None);
        this trainer samples with vllm_temperature, maps a non-positive top_k to vLLM's -1, needs
        the sampled token's logprob for importance sampling (logprobs=0: one entry per token, the
        sampled one, as trl's GRPO trainer requests), and passes min_completion_length as vLLM's
        min_tokens. Both generation paths read these: VLLMGeneration.generate and the overlap
        path's direct client call (_vllm_request_kwargs).
        """
        vg = self.vllm_generation
        gc = self.generation_config
        vg.temperature = self.args.vllm_temperature
        vg.top_k = gc.top_k if gc.top_k and gc.top_k > 0 else -1
        vg.top_p = getattr(self.args, "top_p", 1.0)
        vg.repetition_penalty = getattr(self.args, "repetition_penalty", 1.0)
        vg.min_p = getattr(self.args, "min_p", 0.0) or 0.0
        vg.max_completion_length = gc.max_new_tokens
        vg.logprobs = 0
        min_tokens = int(getattr(self.args, "min_completion_length", 0) or 0)
        if min_tokens > 0:
            vg.generation_kwargs["min_tokens"] = min_tokens

    def _vllm_request_kwargs(self) -> dict:
        """VLLMClient.generate keywords, built from VLLMGeneration the way it builds them."""
        vg = self.vllm_generation
        return {
            "n": 1,
            "repetition_penalty": vg.repetition_penalty,
            "temperature": vg.temperature,
            "top_p": vg.top_p,
            "top_k": vg.top_k,
            "min_p": vg.min_p,
            "max_tokens": vg.max_completion_length,
            "logprobs": vg.logprobs,
            "structured_outputs_regex": vg.structured_outputs_regex,
            "generation_kwargs": vg.generation_kwargs,
        }

    @staticmethod
    def _prompt_token_ids(inputs, max_prompt_tokens: int) -> list[list[int]]:
        """This rank's prompts as token IDs: collator left padding dropped, the last
        max_prompt_tokens kept (as the text path truncated from the left)."""
        ids = []
        for row, mask in zip(inputs["prompts"], inputs["prompt_attention_mask"]):
            row_ids = row[mask.bool()].tolist()
            if max_prompt_tokens > 0 and len(row_ids) > max_prompt_tokens:
                row_ids = row_ids[-max_prompt_tokens:]
            ids.append(row_ids)
        return ids

    @staticmethod
    def _sampled_token_logprobs(logprobs, completion_ids) -> list[list[float]]:
        """Per-token logprob of the sampled token, as floats, NaN where vLLM reported NaN.

        trl's client returns ``[sequence][token][k]`` sorted by probability, NaN as ``None``.
        With logprobs=0 there is one entry per token, the sampled one, so index 0 is it (how trl
        reads it). ``None`` becomes NaN, which the importance-sampling terms mask out. With no
        logprobs returned at all, NaN everywhere: every token is then masked from IS, rather than
        weighted against a made-up value.
        """
        if logprobs is None:
            return [[math.nan] * len(c) for c in completion_ids]
        return [
            [math.nan if tok[0] is None else float(tok[0]) for tok in seq]
            for seq in logprobs
        ]

    def _get_current_lmbda(self) -> float:
        """
        Compute current lmbda value based on training progress.

        Returns:
            Current lmbda value, linearly interpolated if lmbda_schedule='linear'
        """
        if self.args.lmbda_schedule == "constant":
            return self.lmbda

        if self.args.lmbda_schedule == "linear":
            # The ramp runs from lmbda_ramp_start_step for lmbda_ramp_steps steps (0 = until
            # max_steps). At the defaults (0, 0) this is the original global_step / max_steps.
            start = int(getattr(self.args, "lmbda_ramp_start_step", 0) or 0)
            length = int(getattr(self.args, "lmbda_ramp_steps", 0) or 0)
            if length <= 0:
                length = self.state.max_steps - start
            current_step = self.state.global_step

            if length <= 0:
                return self.args.lmbda_init

            progress = min(1.0, max(0.0, (current_step - start) / length))
            current_lmbda = self.args.lmbda_init + progress * (
                self.args.lmbda - self.args.lmbda_init
            )
            return current_lmbda

        return self.lmbda

    def _draw_on_policy(self, current_lmbda: float) -> bool:
        """Whether THIS step draws from the teacher, decided identically on every rank.

        Delegates to distillation.on_policy_draw, which carries the full account of why this is not
        `random.random() <= current_lmbda` any more. The short version: both call sites below guard a
        collective, the old guard read Python's per-process global RNG, and when the ranks disagreed
        a direct measurement lost a two-node allocation to an UnpicklingError on one rank and a 1800 s NCCL
        watchdog abort on the rest -- with zero optimizer steps logged.
        """
        return draw_on_policy(
            current_lmbda,
            seed=getattr(self.args, "seed", 42),
            global_step=getattr(self.state, "global_step", 0),
            device=self.accelerator.device,
        )

    def _get_current_beta(self) -> float:
        """
        Compute current beta value based on training progress.

        Returns:
            Current beta value, linearly interpolated if beta_schedule='linear'
        """
        if self.args.beta_schedule == "constant":
            return self.beta

        if self.args.beta_schedule == "linear":
            max_steps = self.state.max_steps
            current_step = self.state.global_step

            if max_steps <= 0:
                return self.args.beta_init

            progress = min(1.0, current_step / max_steps)
            current_beta = self.args.beta_init + progress * (
                self.beta - self.args.beta_init
            )
            return current_beta

        return self.beta

    def _get_current_alpha(self) -> float:
        if self.args.alpha_schedule == "constant":
            return self.args.alpha

        if self.args.alpha_schedule == "linear":
            max_steps = self.state.max_steps
            current_step = self.state.global_step

            if max_steps <= 0:
                return self.args.alpha_init

            progress = min(1.0, current_step / max_steps)
            return self.args.alpha_init + progress * (
                self.args.alpha - self.args.alpha_init
            )

        return self.args.alpha

    def create_optimizer(self):
        """Override to include hidden projection parameters in the optimizer."""
        optimizer = super().create_optimizer()
        if self.args.use_hidden_loss and getattr(self, "hidden_proj", None) is not None:
            # With DeepSpeed ZeRO-3 (zero3_init_flag), hidden_proj params are already
            # registered and included in the optimizer. Only add manually without DeepSpeed.
            if not self.is_deepspeed_enabled:
                self.optimizer.add_param_group(
                    {
                        "params": list(self.hidden_proj.parameters()),
                        "lr": self.args.learning_rate,
                    }
                )
        return optimizer

    def _prepare_dataset_with_original_text(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: (
            PreTrainedTokenizerBase
            | BaseImageProcessor
            | FeatureExtractionMixin
            | ProcessorMixin
        ),
        args,
        packing: bool,
        formatting_func: Callable[[dict], str] | None,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        """
        Prepare dataset while preserving original text for cross-tokenizer distillation.
        """
        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().main_process_first():
            # Apply the formatting function if any
            if formatting_func is not None:
                if isinstance(
                    dataset, Dataset
                ):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = (
                        f"Applying formatting function to {dataset_name} dataset"
                    )

                def _func(example):
                    return {"text": formatting_func(example)}

                dataset = dataset.map(_func, batched=False, **map_kwargs)

            # Convert the dataset to ChatML if needed
            if isinstance(
                dataset, Dataset
            ):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Converting {dataset_name} dataset to ChatML"
            column_names = next(iter(dataset)).keys()
            dataset = dataset.map(
                maybe_convert_to_chatml,
                remove_columns=(
                    "conversations" if "conversations" in column_names else None
                ),
                **map_kwargs,
            )

            # Apply the chat template if needed and preserve original text
            first_example = next(iter(dataset))
            if not is_conversational(first_example):
                if isinstance(
                    dataset, Dataset
                ):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Adding EOS to {dataset_name} dataset"

                def add_eos(example, eos_token):
                    if "text" in example and not example["text"].endswith(
                        eos_token
                    ):  # language modeling case
                        example["text"] = example["text"] + eos_token
                    elif "completion" in example and not example["completion"].endswith(
                        eos_token
                    ):
                        example["completion"] = example["completion"] + eos_token
                    return example

                dataset = dataset.map(
                    add_eos,
                    fn_kwargs={"eos_token": processing_class.eos_token},
                    remove_columns=(
                        "messages" if "messages" in column_names else None
                    ),  # renamed to "text"
                    **map_kwargs,
                )

            # Tokenize the dataset while preserving original text
            if isinstance(
                dataset, Dataset
            ):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = (
                    f"Tokenizing {dataset_name} dataset (preserving original text)"
                )

            def tokenize_with_original_text(
                example, processing_class, dataset_text_field, assistant_only_loss
            ):
                """Modified tokenization function that preserves original text."""
                result = {}

                if "prompt" in example:  # prompt-completion case
                    raise ValueError(
                        "Unsupported format: prompt-completion case is not supported yet"
                    )
                else:  # language modeling or conversational case
                    if is_conversational(example):
                        # For conversational data (ChatML), extract prompt and completion properly
                        messages = example["messages"]
                        prompt_messages = example["messages"][:-1]

                        # `tools` REACHES THIS LINE IN THREE SHAPES, and the original assumed one.
                        #
                        #   JSON STRING  -- upstream en_sft_4.1 stores it as the string "[]".
                        #   LIST         -- our own data-prep parsed it in order to
                        #                   render and then emitted the parsed list
                        #                   (prep_corpus.py:423), so `json.loads(a list)` raised
                        #                     TypeError: the JSON object must be str, bytes or
                        #                     bytearray, not list
                        #                   on the deliverable corpus. 17.3% (138,395 rows) of
                        #                   en-sft-4.1-0.2-16K carries tools and NOT ONE of those
                        #                   rows is inside the 2,000-row probe head that every run
                        #                   so far has used, so every run was green and the full
                        #                   epoch would have died in preprocessing. Found by
                        #                   a companion check, confirmed directly, at sampled
                        #                   row 668,300 of 802,027.
                        #   NaN / None   -- the LOADER's doing, not the corpus's: gold.py:521 reads
                        #                   a .jsonl with `pd.read_json`, and pandas fills a column
                        #                   absent from a row with NaN, a float. `NaN or "[]"` does
                        #                   not rescue that, because NaN is truthy. 82.7% (663,632 rows) of the
                        #                   deliverable corpus's rows have no tools key at all.
                        #
                        # Normalised to the STRING form, not the list form, for three reasons:
                        # `result["tools"]` below must have ONE Arrow type across all 802k rows, the
                        # generation path json.loads it again (:3075), and the string is what a
                        # corpus with no tools column already yields. prep_corpus.py now emits the
                        # string as well; this stays tolerant so an already-built corpus does not
                        # have to be re-prepped to be trainable.
                        tools_raw = example.get("tools")
                        if isinstance(tools_raw, str):
                            tools_raw = tools_raw or "[]"
                            tools = json.loads(tools_raw)
                        elif isinstance(tools_raw, (list, tuple)):
                            tools = list(tools_raw)
                            tools_raw = json.dumps(tools)
                        else:
                            tools, tools_raw = [], "[]"
                        result["tools"] = (
                            tools_raw  # keep as string for Arrow compatibility
                        )

                        # Same NaN hazard, opposite contract: `documents` is forwarded to the
                        # template AS A LIST (a string would be iterated character by character --
                        # confirmed directly), so it is NOT serialised here. It still has to survive
                        # pandas' NaN for a row that lacks the column, which the plain
                        # `.get("documents", [])` default does not: the key EXISTS with value NaN.
                        documents = example.get("documents")
                        if not isinstance(documents, (list, tuple)):
                            documents = []

                        # PER-ROW, not per-run. `enable_thinking` only ever reaches
                        # chat_template.jinja:190, which is INSIDE the add_generation_prompt block:
                        # True emits `assistant\n<think>\n`, False emits `assistant\n<think></think>`.
                        # The full render and the `input_ids` call below are invariant to it, so this
                        # moves the ULD prompt/completion BOUNDARY and nothing else -- it cannot
                        # change training tokenization for any existing arm.
                        #
                        # A single hardcoded False was correct only for a homogeneous non-thinking
                        # corpus. The agentic mixture is bimodal (graphsyn/tau/when2call/hermes carry
                        # inline reasoning, toucan/anchor/apigen do not), and under one global flag
                        # 59.2% of it raises below at line 1881 -- 117,030 of 197,795 rows, job
                        # 1473449. Selecting per row is provably optimal there: its residual equals
                        # the set that fails under EVERY flag, exactly.
                        #
                        # The flag travels with the row because prep is where it can be VERIFIED:
                        # mixture-render-fix.py re-renders each row under its own flag and writes
                        # only the ones whose startswith guard holds. Defaulting to False keeps every
                        # corpus that predates the field rendering byte-identically.
                        row_thinking = bool(example.get("render_thinking", False))
                        # result["documents"] = documents

                        prompt_text = processing_class.apply_chat_template(
                            prompt_messages,
                            tokenize=False,
                            add_generation_prompt=True,  # Add assistant prompt
                            documents=documents,
                            tools=tools,
                            enable_thinking=row_thinking,  # per-row: see row_thinking above (was a hardcoded False)
                            **example.get("chat_template_kwargs", {}),
                        )

                        # Get the full conversation with assistant response
                        full_text = processing_class.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False,
                            documents=documents,
                            tools=tools,
                            enable_thinking=row_thinking,  # per-row: see row_thinking above (was a hardcoded False)
                            **example.get("chat_template_kwargs", {}),
                        )

                        # Extract completion as everything after the prompt
                        # This ensures we capture any extra tokens (like <think> tags) that the template adds
                        if full_text.startswith(prompt_text):
                            completion_text = full_text[len(prompt_text) :]
                        else:
                            raise ValueError(
                                "Failed to extract completion text: full text does not start with prompt text. "
                            )
                            # print("Failed to extract completion text: full text does not start with prompt text.")
                            return None

                        # Store original text for cross-tokenizer distillation
                        result["original_prompt_text"] = prompt_text
                        result["original_completion_text"] = completion_text

                        prompt_input_ids = processing_class.apply_chat_template(
                            prompt_messages,
                            tokenize=True,
                            add_generation_prompt=True,  # Add assistant prompt
                            return_dict=False,
                            documents=documents,
                            tools=tools,
                            enable_thinking=row_thinking,  # per-row: see row_thinking above (was a hardcoded False)
                            **example.get("chat_template_kwargs", {}),
                        )
                        result["prompts"] = prompt_input_ids

                        # Process the conversation normally
                        processed = processing_class.apply_chat_template(
                            example["messages"],
                            documents=documents,
                            tools=tools,
                            return_dict=True,
                            return_assistant_tokens_mask=assistant_only_loss,
                            **example.get("chat_template_kwargs", {}),
                        )
                        if (
                            "assistant_masks" in processed
                            and 1 not in processed["assistant_masks"]
                        ):
                            raise RuntimeError(
                                "You're using `assistant_only_loss=True`, but at least one example has no "
                                "assistant tokens. This usually means the tokenizer's chat template doesn't "
                                "generate assistant masks — it may be missing the `{% generation %}` tag. Please "
                                "check the template and ensure it's correctly configured to support assistant "
                                "masking."
                            )
                        result.update(
                            {
                                k: processed[k]
                                for k in ("input_ids", "assistant_masks")
                                if k in processed
                            }
                        )
                        # Strip trailing post-EOS tokens (e.g. newline from Granite's chat template)
                        eos_id = processing_class.eos_token_id
                        if eos_id is not None and len(result["input_ids"]) >= 2:
                            ids = result["input_ids"]
                            for k in range(len(ids) - 1, -1, -1):
                                if ids[k] == eos_id:
                                    if k < len(ids) - 1:
                                        result["input_ids"] = ids[: k + 1]
                                        if "assistant_masks" in result:
                                            result["assistant_masks"] = result[
                                                "assistant_masks"
                                            ][: k + 1]
                                    break
                        # Add attention_mask if not already present
                        if "attention_mask" not in result:
                            result["attention_mask"] = [1] * len(result["input_ids"])

                        # Token length as a COLUMN, so that HF's LengthGroupedSampler can read it
                        # directly (trainer.py:1014-1027) instead of taking its `lengths=None` path,
                        # which materializes every row as a python dict just to call len() on one
                        # field -- here that field sits beside original_prompt_text,
                        # original_completion_text, prompts and assistant_masks, all of which get
                        # deserialized and thrown away. Cost not measured, but it is a full pass over
                        # the corpus before step 1.
                        #
                        # INERT AS SHIPPED: nothing in this tree reads a "length" column, and
                        # train_sampling_strategy defaults to "random", so this only becomes
                        # load-bearing if that is deliberately set to "group_by_length". It is added
                        # ahead of that decision so the decision is cheap to act on, not to
                        # pre-empt it -- length-grouped batching is NOT ordering-neutral.
                        result["length"] = len(result["input_ids"])
                    else:
                        raise ValueError(
                            "Unsupported format: non-conversational data is not supported yet"
                        )

                return result

            print(f"dataset: {len(dataset)}")
            dataset = dataset.map(
                tokenize_with_original_text,
                fn_kwargs={
                    "processing_class": processing_class,
                    "dataset_text_field": args.dataset_text_field,
                    "assistant_only_loss": args.assistant_only_loss,
                },
                **map_kwargs,
            )

            if (
                args.max_length is None
                or args.max_completion_length is None
                or args.lmbda == 0.0
            ):
                # Filter out skipped examples (where tokenization returned None)
                dataset = dataset.filter(
                    lambda x: x.get("prompts") is not None,
                    **map_kwargs,
                )
            else:
                # Also filter out examples with long prompts
                dataset = dataset.filter(
                    lambda x: x.get("prompts") is not None
                    and len(x["prompts"])
                    < args.max_length - args.max_completion_length,
                    **map_kwargs,
                )
            print(f"dataset: {len(dataset)} (after filtering)")

            # Pack or truncate
            if packing:
                if args.max_length is None:
                    raise ValueError(
                        "When packing is enabled, `max_length` can't be `None`."
                    )
                if isinstance(
                    dataset, Dataset
                ):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Packing {dataset_name} dataset"

                # THE SUPERVISION MASK MUST SURVIVE PACKING, and it did not.
                #
                # This allowlist previously kept only input_ids and the two original_* text fields,
                # so `assistant_masks` was dropped before pack_dataset ran. That mask is what
                # `completion_boundary: all_assistant` supervision IS -- CustomDataCollatorForChatML
                # labels the spans it marks. Without it the collator silently falls back to a
                # positional split (utils.py:450), so enabling packing would not fail, it would
                # train on DIFFERENT TOKENS than the config asks for. Nothing in the pipeline
                # notices: the loss still decreases and the manifest still reports all_assistant.
                #
                # Packing them is safe: _pack_bfd (trl/data_utils.py:645) iterates EVERY list column,
                # slices each to seq_length and packs them together, so a per-token mask travels with
                # the tokens it belongs to.
                #
                # DELIBERATELY NOT KEPT, because packing invalidates them rather than moving them:
                #   attention_mask  a packed row needs BLOCK-DIAGONAL attention so concatenated
                #                   sequences cannot attend across their boundary. An all-ones mask
                #                   carried through would silently permit exactly that.
                #   position_ids    pack_dataset appends `seq_lengths` specifically so position_ids
                #                   can be RECONSTRUCTED per packed segment afterwards; a pre-packing
                #                   copy would number the tokens wrongly.
                # Both are regenerated downstream from seq_lengths, which is the intended contract.
                columns_to_keep = [
                    "input_ids",
                    "original_prompt_text",
                    "original_completion_text",
                    "assistant_masks",
                    "completion_mask",
                ]
                if args.use_uld_loss:
                    columns_to_keep.append("messages")
                existing_columns = set(dataset.column_names)
                columns_to_select = [
                    col for col in columns_to_keep if col in existing_columns
                ]

                # Refuse rather than mis-supervise -- but only where the mask is actually owed.
                #
                # SCOPE, because an earlier version of this guard was unconditional and would have
                # broken a legitimate config. `assistant_masks` is produced only when
                # `assistant_only_loss=True` (tokenize_with_original_text passes it straight through
                # as `return_assistant_tokens_mask` at line ~1903). With assistant_only_loss=False
                # the mask is never requested, its absence here is correct, and packing without it
                # is the intended behaviour. So the condition is "requested but missing", not
                # "missing" -- the latter is a valid state.
                #
                # Where it IS requested, absence is fatal rather than warning-level: the collator
                # would fall back to a positional split (utils.py:450) and train on different tokens
                # than the config asks for, with the loss still decreasing and the manifest still
                # reporting the configured boundary. The check reads columns_to_select -- what
                # actually reaches pack_dataset -- so an edit to either the allowlist or the
                # upstream producer cannot silently reopen the hole.
                if (
                    getattr(args, "assistant_only_loss", False)
                    and "assistant_masks" not in columns_to_select
                ):
                    raise ValueError(
                        "packing is enabled and assistant_only_loss=True, but 'assistant_masks' is "
                        f"not in the dataset (columns: {sorted(existing_columns)}). That mask is "
                        "what assistant-span supervision consists of; packing without it would "
                        "silently train on a positional split instead. Two things produce this: a "
                        "chat template with no `{% generation %}` markers, or a `response_template` "
                        "that does not occur in the rendered text. Refusing to proceed."
                    )

                dataset = dataset.select_columns(columns_to_select)
                dataset = pack_dataset(
                    dataset, args.max_length, args.packing_strategy, map_kwargs
                )
            elif args.max_length is not None:
                if isinstance(
                    dataset, Dataset
                ):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Truncating {dataset_name} dataset"
                dataset = truncate_dataset(dataset, args.max_length, map_kwargs)

            if args.use_liger_kernel:
                required_columns = {
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "completion_mask",
                    "assistant_masks",
                    "original_prompt_text",
                    "original_completion_text",
                }
                if args.use_uld_loss:
                    required_columns.add("messages")
                dataset = dataset.select_columns(
                    required_columns.intersection(dataset.column_names)
                )

        return dataset

    @profiling_decorator
    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        """
        Perform a training step for the General Online Logit Distillation (GOLD) model.

        This method implements the on-policy learning approach described in the GOLD blog post. With probability
        `self.lmbda`, it generates new responses using the student model, which are then used for training instead of
        the offline original inputs.
        """
        _t0_step = time.perf_counter()
        self._timing_generation_last_call = 0.0
        on_policy = False
        _overlap_saved_inputs = None
        _overlap_future = None
        _overlap_n_per_process = 0
        _overlap_n_all_prompts = 0
        current_lmbda = self._get_current_lmbda()
        # --- Overlap generation path (lookahead scheduling) ---
        if (
            self.args.overlap_generation
            and self.use_vllm
            and self.vllm_mode == "server"
        ):
            IS_enabled = (
                self.args.opd_importance_sampling
                or self.args.uld_hybrid_sampled_opd_weight > 0
            )

            # Step 1: Determine if current step is on-policy
            if self._overlap_buffer is not None:
                on_policy = True
                buf = self._overlap_buffer
                inputs["input_ids"] = buf["input_ids"]
                inputs["attention_mask"] = buf["attention_mask"]
                inputs["labels"] = buf["labels"]
                inputs["on_policy"] = [True] * buf["input_ids"].shape[0]
                if IS_enabled:
                    inputs["vllm_logprobs"] = buf["vllm_logprobs"]
                inputs["original_prompt_text"] = buf["prompt_texts"]
                inputs["original_completion_text"] = buf["completion_texts"]
                self._textual_logs["prompt"].extend(gather_object(buf["prompt_texts"]))
                self._textual_logs["completion"].extend(
                    gather_object(buf["completion_texts"])
                )
                self._overlap_buffer = None

            elif current_lmbda >= 1.0:
                # Bootstrap: lmbda=1.0 but buffer is empty (first step) → sync generate
                on_policy = True
                result = self._generate_on_policy_outputs_vllm(
                    inputs, self.generation_config, self.processing_class.pad_token_id
                )
                (
                    new_input_ids,
                    new_attention_mask,
                    new_labels,
                    prompt_texts,
                    completion_texts,
                    vllm_logprobs,
                ) = result
                inputs["input_ids"] = new_input_ids
                inputs["attention_mask"] = new_attention_mask
                inputs["labels"] = new_labels
                inputs["on_policy"] = [True] * new_input_ids.shape[0]
                if IS_enabled:
                    inputs["vllm_logprobs"] = vllm_logprobs
                inputs["original_prompt_text"] = prompt_texts
                inputs["original_completion_text"] = completion_texts
                self._textual_logs["prompt"].extend(gather_object(prompt_texts))
                self._textual_logs["completion"].extend(gather_object(completion_texts))

            # Step 2: Decide if NEXT step should be on-policy → start async gen
            if self._draw_on_policy(current_lmbda):
                _overlap_saved_inputs = {k: v for k, v in inputs.items()}

                max_prompt_tokens = (
                    self.args.max_length - self.generation_config.max_new_tokens
                )
                prompt_id_lists = self._prompt_token_ids(inputs, max_prompt_tokens)
                _overlap_n_per_process = len(prompt_id_lists)
                all_prompt_ids = gather_object(prompt_id_lists)
                _overlap_n_all_prompts = len(all_prompt_ids)

                if self.accelerator.is_main_process:
                    _t_submit = time.perf_counter()
                    # The same request VLLMGeneration.generate sends, issued directly so it
                    # can run while this step trains.
                    _overlap_future = self._overlap_executor.submit(
                        self.vllm_client.generate,
                        prompts=all_prompt_ids,
                        **self._vllm_request_kwargs(),
                    )
                    _overlap_future._submit_time = _t_submit

                    def _on_done(fut, _trainer=self):
                        fut._done_time = time.perf_counter()

                    _overlap_future.add_done_callback(_on_done)

        # --- Standard (non-overlap) path ---
        elif self._draw_on_policy(current_lmbda):
            on_policy = True

            if self.use_vllm:
                result = self._generate_on_policy_outputs_vllm(
                    inputs, self.generation_config, self.processing_class.pad_token_id
                )
                (
                    new_input_ids,
                    new_attention_mask,
                    new_labels,
                    prompt_texts,
                    completion_texts,
                    vllm_logprobs,
                ) = result
            else:
                with unwrap_model_for_generation(
                    model,
                    self.accelerator,
                    generation_kwargs=self.generation_kwargs,  # Override model.generation_config with generation_kwargs to fix transformers#42762
                ) as unwrapped_model:
                    # trl 1.x dropped the pad_token_id argument: it masks after the
                    # first EOS and by the prompt mask, not by pad id.
                    result = self.generate_on_policy_outputs(
                        unwrapped_model,
                        inputs,
                        self.generation_config,
                    )
                    (
                        new_input_ids,
                        new_attention_mask,
                        new_labels,
                        prompt_texts,
                        completion_texts,
                    ) = result

            # Reconstruct messages for cross-tokenizer distillation
            if (
                self.use_uld_loss
                and "messages" in inputs
                and inputs["messages"] is not None
                and inputs["messages"][0] is not None
            ):
                on_policy_messages = []
                for msgs, comp_text in zip(inputs["messages"], completion_texts):
                    prompt_msgs = msgs[:-1]
                    clean_completion = comp_text
                    if (
                        hasattr(self.processing_class, "eos_token")
                        and self.processing_class.eos_token
                    ):
                        clean_completion = clean_completion.replace(
                            self.processing_class.eos_token, ""
                        )
                    on_policy_messages.append(
                        prompt_msgs
                        + [{"role": "assistant", "content": clean_completion}]
                    )

            if self.args.use_distillm2:
                inputs["input_ids"] = torch.cat(
                    [inputs["input_ids"], new_input_ids], dim=0
                )
                inputs["attention_mask"] = torch.cat(
                    [inputs["attention_mask"], new_attention_mask], dim=0
                )
                inputs["labels"] = torch.cat([inputs["labels"], new_labels], dim=0)
                inputs["on_policy"] += [True] * new_input_ids.shape[0]
                if (
                    self.use_uld_loss
                    and "messages" in inputs
                    and inputs["messages"] is not None
                    and inputs["messages"][0] is not None
                ):
                    inputs["messages"] = inputs["messages"] + on_policy_messages
            else:
                inputs["input_ids"] = new_input_ids
                inputs["attention_mask"] = new_attention_mask
                inputs["labels"] = new_labels
                inputs["on_policy"] = [True] * new_input_ids.shape[0]
                if (
                    self.use_uld_loss
                    and "messages" in inputs
                    and inputs["messages"] is not None
                    and inputs["messages"][0] is not None
                ):
                    inputs["messages"] = on_policy_messages

            # Store vLLM logprobs for importance sampling (only from vLLM path)
            if self.use_vllm and (
                self.args.opd_importance_sampling
                or self.args.uld_hybrid_sampled_opd_weight > 0
            ):
                inputs["vllm_logprobs"] = vllm_logprobs

            # CRITICAL: Preserve original text for cross-tokenizer ULD loss
            inputs["original_prompt_text"] = prompt_texts
            inputs["original_completion_text"] = completion_texts

            # Log prompt and completion texts
            self._textual_logs["prompt"].extend(gather_object(prompt_texts))
            self._textual_logs["completion"].extend(gather_object(completion_texts))

        # loss = super().training_step(model, inputs, num_items_in_batch)
        _t0_train_pure = time.perf_counter()
        loss = super(GOLDTrainer, self).training_step(model, inputs, num_items_in_batch)
        self._timing_training_pure += time.perf_counter() - _t0_train_pure

        # Overlap: collect async generation results and store new buffer
        if _overlap_saved_inputs is not None:
            if self.accelerator.is_main_process:
                _t0_wait = time.perf_counter()
                vllm_response = _overlap_future.result()
                _wait_elapsed = time.perf_counter() - _t0_wait
                _done_time = getattr(_overlap_future, "_done_time", time.perf_counter())
                _gen_actual = _done_time - _overlap_future._submit_time
                self._timing_generation_wait += _wait_elapsed
                self._timing_generation_actual += _gen_actual
                completion_ids = vllm_response["completion_ids"]
                vllm_logprobs_raw = self._sampled_token_logprobs(
                    vllm_response["logprobs"], completion_ids
                )
            else:
                completion_ids = [None] * _overlap_n_all_prompts
                vllm_logprobs_raw = [None] * _overlap_n_all_prompts

            _t0_collect = time.perf_counter()
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            vllm_logprobs_raw = broadcast_object_list(vllm_logprobs_raw, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * _overlap_n_per_process,
                (self.accelerator.process_index + 1) * _overlap_n_per_process,
            )
            completion_ids = completion_ids[process_slice]
            vllm_logprobs_raw = vllm_logprobs_raw[process_slice]

            _prev_gen_timing = self._timing_generation
            result = self._generate_on_policy_outputs_vllm(
                _overlap_saved_inputs,
                self.generation_config,
                self.processing_class.pad_token_id,
                _prefetched=(completion_ids, vllm_logprobs_raw),
            )
            self._timing_generation = _prev_gen_timing
            (
                new_input_ids,
                new_attention_mask,
                new_labels,
                prompt_texts,
                completion_texts,
                vllm_logprobs,
            ) = result
            self._overlap_buffer = {
                "input_ids": new_input_ids,
                "attention_mask": new_attention_mask,
                "labels": new_labels,
                "prompt_texts": prompt_texts,
                "completion_texts": completion_texts,
                "vllm_logprobs": vllm_logprobs,
            }
            _collect_elapsed = time.perf_counter() - _t0_collect
            self._timing_generation += _collect_elapsed
            self._timing_generation_last_call = _collect_elapsed

        loss_scalar = float(loss.detach())
        # Guard the LOGGED off/on-policy loss metric against NaN from empty-mask microbatches
        # (a row with zero labeled/completion tokens -> JSD over empty set -> NaN). The real
        # backprop loss is handled separately; here we only protect the running-average METRIC
        # so one bad microbatch does not poison it forever (nan + x = nan).
        import math as _math

        _metric_finite = _math.isfinite(loss_scalar)
        ga = max(1, int(self.args.gradient_accumulation_steps))
        step_equiv = 1.0 / ga

        # Schedule values (current_lmbda/beta/alpha) are logged from inside the
        # overridden log(), NOT from here: this runs per micro-batch, and the
        # extra self.log() calls all-reduced and reset the ULD/alignment/IS
        # accumulators, so with logging_steps: 1 the regular log line reported
        # about zero for them.

        if on_policy:
            # print("on-policy")
            # print(inputs["input_ids"].shape)
            # for i in range(inputs["input_ids"].shape[0]):
            #     print('+' * 60)
            #     print(self.processing_class.decode(inputs["input_ids"][i]))
            # print('+' * 60)
            # valid_mask = inputs["labels"] != -100
            # decoded_labels = [self.processing_class.decode(ids[mask]) for ids, mask in zip(inputs["input_ids"], valid_mask)]
            # print(decoded_labels)
            if _metric_finite:
                self._on_policy_loss_total += loss_scalar
                self._on_policy_step_equiv += step_equiv
        else:
            # print("off-policy")
            # print(inputs["input_ids"].shape)
            # for i in range(inputs["input_ids"].shape[0]):
            #     print('-' * 60)
            #     print(self.processing_class.decode(inputs["input_ids"][i]))
            # print('-' * 60)
            # valid_mask = inputs["labels"] != -100
            # decoded_labels = [self.processing_class.decode(ids[mask]) for ids, mask in zip(inputs["input_ids"], valid_mask)]
            # print(decoded_labels)
            # import pdb; pdb.set_trace()
            if _metric_finite:
                self._off_policy_loss_total += loss_scalar
                self._off_policy_step_equiv += step_equiv

        self._timing_training_step_total += time.perf_counter() - _t0_step
        return loss

    def _move_model_to_vllm(self):
        """Push the student's current weights to vLLM through trl's VLLMGeneration.sync_weights.

        sync_weights does what this method did by hand under trl 0.26.2: gather each parameter
        under ZeRO-3 (on every rank, as the gather is a collective), merge PEFT adapters, stream
        the tensors to the server over the weight-transfer group in one update (or load them
        into the colocated engine), then reset the prefix cache. Under trl 1.x a per-tensor
        update_named_param would cost three HTTP calls and a server-side finalize per tensor.

        Timed as a whole: the gather/broadcast split the old loop logged is not observable from
        outside sync_weights, so weight_sync_allgather / weight_sync_broadcast stay 0.
        """
        t_total = time.perf_counter()
        self.vllm_generation.sync_weights()
        self._timing_weight_sync += time.perf_counter() - t_total
        self._timing_weight_sync_params = len(
            self.vllm_generation._weight_metadata or []
        )

    def _build_alignment_groups_at_bytes_from_completion_ids(
        self, completion_ids, new_completion_ids
    ):
        """
        Build alignment groups using byte-level matching to align original completion ids and new (retokenized) ids.
        (Mostly a duplication of CustomULDLoss's _build_alignment_groups_at_bytes_from_ids)
        """
        eos_id = self.processing_class.eos_token_id

        s_has_eos = len(completion_ids) > 0 and completion_ids[-1] == eos_id
        t_has_eos = len(new_completion_ids) > 0 and new_completion_ids[-1] == eos_id

        s_ids_core = completion_ids[:-1] if s_has_eos else completion_ids
        t_ids_core = new_completion_ids[:-1] if t_has_eos else new_completion_ids

        s_groups: list[list[int]] = []
        t_groups: list[list[int]] = []
        truncated = False
        max_group_size = 20

        if s_ids_core and t_ids_core:

            def to_byte_pieces(tok, ids):
                pieces = []
                for tid in ids:
                    token_str = tok.convert_ids_to_tokens(tid)
                    pieces.append(bytes(self._unicode_to_byte[c] for c in token_str))
                return pieces

            s_pieces = to_byte_pieces(self.processing_class, s_ids_core)
            t_pieces = to_byte_pieces(self.processing_class, t_ids_core)

            i = j = 0
            s_buf = b""
            t_buf = b""
            s_group: list[int] = []
            t_group: list[int] = []

            def flush():
                if s_group and t_group:
                    s_groups.append(s_group.copy())
                    t_groups.append(t_group.copy())

            while i < len(s_pieces) or j < len(t_pieces):
                if s_buf == t_buf and s_buf != b"":
                    flush()
                    s_buf = b""
                    t_buf = b""
                    s_group = []
                    t_group = []
                    continue

                if max_group_size > 0 and (
                    len(s_group) >= max_group_size or len(t_group) >= max_group_size
                ):
                    truncated = True
                    break

                if s_buf == b"" and i < len(s_pieces):
                    s_buf += s_pieces[i]
                    s_group.append(i)
                    i += 1
                    continue
                if t_buf == b"" and j < len(t_pieces):
                    t_buf += t_pieces[j]
                    t_group.append(j)
                    j += 1
                    continue

                if len(s_buf) <= len(t_buf):
                    if i < len(s_pieces):
                        s_buf += s_pieces[i]
                        s_group.append(i)
                        i += 1
                    elif j < len(t_pieces):
                        t_buf += t_pieces[j]
                        t_group.append(j)
                        j += 1
                else:
                    if j < len(t_pieces):
                        t_buf += t_pieces[j]
                        t_group.append(j)
                        j += 1
                    elif i < len(s_pieces):
                        s_buf += s_pieces[i]
                        s_group.append(i)
                        i += 1

            if not truncated:
                if s_buf == t_buf and s_group and t_group:
                    flush()
                elif s_group or t_group:
                    truncated = True

            while s_groups and not s_groups[-1]:
                s_groups.pop()
                t_groups.pop()

        if not truncated and s_has_eos and t_has_eos:
            s_groups.append([len(completion_ids) - 1])
            t_groups.append([len(new_completion_ids) - 1])

        return s_groups, t_groups

    @profiling_decorator
    def _generate_on_policy_outputs_vllm(
        self, inputs, generation_config, pad_token_id=None, _prefetched=None
    ):
        """Generate this rank's on-policy completions through vLLM and build the training batch.

        Prompts go to vLLM as TOKEN IDS, the trainer's own (``inputs["prompts"]`` with
        ``prompt_attention_mask``), the way trl 1.x's trainers send them. Until trl 0.26.2 the
        prompts were decoded to text and vLLM re-tokenised them, which could drift from the
        trainer's tokenisation; sending IDs removes that path, so the new batch's prompt part is
        exactly what generated the completion.

        Logprobs come back per token as trl's client returns them: requested with
        ``logprobs=0`` (``_configure_vllm_generation``), so each token has one entry, the sampled
        token's, at index 0, which is what trl's GRPO trainer reads. They are the values the
        server sampled with (trl's server runs vLLM with ``--logprobs-mode processed_logprobs``).
        A NaN logprob arrives as ``None`` and becomes NaN here; the importance-sampling terms
        mask NaN tokens out (``_finite_is_mask``).
        """
        _t0_gen = time.perf_counter()
        device = self.accelerator.device

        # This rank's prompts as token IDs: drop the collator's left padding, keep the last
        # max_prompt_tokens so prompt + completion fits max_length.
        max_prompt_tokens = self.args.max_length - generation_config.max_new_tokens
        prompt_id_lists = self._prompt_token_ids(inputs, max_prompt_tokens)

        # Decoded WITH special tokens for the cross-tokenizer ULD loss (unchanged).
        prompts_text_with_special = self.processing_class.batch_decode(
            inputs["prompts"],
            skip_special_tokens=False,
        )

        max_completion_length = generation_config.max_new_tokens

        # How many prompts the SERVING side is asked for in this one call. None when the work was
        # prefetched, because then no request was issued here and counting it would double-count the
        # call that actually made it.
        _gen_batch_seen = None

        if _prefetched is not None:
            completion_ids, vllm_logprobs_raw = _prefetched
        else:
            # trl's VLLMGeneration gathers the prompts to the main process (server) or the TP
            # group (colocate), generates once, and returns this rank's slice.
            _, completion_ids, logprobs, _, _ = self.vllm_generation.generate(
                prompts=prompt_id_lists, images=None, num_generations=1
            )
            vllm_logprobs_raw = self._sampled_token_logprobs(logprobs, completion_ids)
            _gen_batch_seen = len(prompt_id_lists) * (
                self.accelerator.num_processes
                if self.vllm_mode == "server"
                else self.vllm_generation.tensor_parallel_size
            )

        ### Re-tokenize completions
        if self.args.uld_rebuild_student_input:
            completions = self.processing_class.batch_decode(
                completion_ids, skip_special_tokens=False
            )  # , clean_up_tokenization_spaces=False)
            if self.args.opd_importance_sampling:
                new_completion_ids = self.processing_class(completions)["input_ids"]
                new_vllm_logprobs_raw = []
                for i in range(len(completion_ids)):
                    s_groups, t_groups = (
                        self._build_alignment_groups_at_bytes_from_completion_ids(
                            completion_ids[i], new_completion_ids[i]
                        )
                    )
                    # if len(t_groups) != len(new_completion_ids[i]):
                    #     print(f"Completion alignment went wrong: {len(t_groups)} != {len(new_completion_ids[i])}")
                    #     import pdb; pdb.set_trace()
                    # new_vllm_lps = [0.0] * len(new_completion_ids[i]) # default: 0
                    new_vllm_lps = [100.0] * len(new_completion_ids[i])  # default: inf
                    for group_i, t_group in enumerate(t_groups):
                        if len(t_group) == 1:
                            new_vllm_lps[t_group[0]] = sum(
                                vllm_logprobs_raw[i][
                                    s_groups[group_i][0] : s_groups[group_i][-1] + 1
                                ]
                            )
                    # for group_i in range(len(t_groups)):
                    #     if len(t_groups[group_i]) == 1:
                    #         new_vllm_lps[group_i] = sum(vllm_logprobs_raw[i][s_groups[group_i][0]:s_groups[group_i][-1]+1])
                    new_vllm_logprobs_raw.append(new_vllm_lps)

                completion_ids = new_completion_ids
                vllm_logprobs_raw = new_vllm_logprobs_raw
            else:
                completion_ids = self.processing_class(completions)["input_ids"]

        # The prompt half of the new batch is exactly the token IDs vLLM generated from,
        # left-padded like the collator pads.
        prompt_ids = pad(
            [torch.tensor(ids, dtype=torch.long) for ids in prompt_id_lists],
            padding_value=pad_token_id,
            padding_side="left",
        ).to(device)
        prompt_mask = pad(
            [torch.ones(len(ids), dtype=torch.long) for ids in prompt_id_lists],
            padding_value=0,
            padding_side="left",
        ).to(device)

        completion_ids_tensors = [
            torch.tensor(ids, device=device) for ids in completion_ids
        ]
        # Manually pad/truncate completions to max_completion_length length before using pad function.
        # The per-row valid length is kept (completion_valid_lengths) so the masking
        # below can mask padding by POSITION: with pad == eos (gold.py's fallback
        # when the tokenizer defines no pad token) masking by token value also
        # removes every generated EOS from the labels and attention.
        padded_completion_ids_list = []
        completion_valid_lengths = []
        for completion_tensor in completion_ids_tensors:
            if len(completion_tensor) > max_completion_length:
                # Truncate if longer than max_completion_length
                padded_completion_ids_list.append(
                    completion_tensor[:max_completion_length]
                )
                completion_valid_lengths.append(max_completion_length)
            elif len(completion_tensor) < max_completion_length:
                # Pad if shorter than max_completion_length
                padding_needed = max_completion_length - len(completion_tensor)
                padded_tensor = torch.cat(
                    [
                        completion_tensor,
                        torch.full(
                            (padding_needed,),
                            pad_token_id,
                            device=device,
                            dtype=completion_tensor.dtype,
                        ),
                    ]
                )
                padded_completion_ids_list.append(padded_tensor)
                completion_valid_lengths.append(len(completion_tensor))
            else:
                # Already the right length
                padded_completion_ids_list.append(completion_tensor)
                completion_valid_lengths.append(len(completion_tensor))

        # Now all tensors are the same length, so we can stack them
        padded_completion_ids = torch.stack(padded_completion_ids_list)

        # Build padded logprobs tensor aligned with completion_ids
        padded_logprobs_list = []
        for lp_list in vllm_logprobs_raw:
            lp_tensor = torch.tensor(lp_list, device=device, dtype=torch.float32)
            if len(lp_tensor) > max_completion_length:
                padded_logprobs_list.append(lp_tensor[:max_completion_length])
            elif len(lp_tensor) < max_completion_length:
                padding_needed = max_completion_length - len(lp_tensor)
                padded_logprobs_list.append(
                    torch.cat(
                        [
                            lp_tensor,
                            torch.zeros(
                                padding_needed, device=device, dtype=torch.float32
                            ),
                        ]
                    )
                )
            else:
                padded_logprobs_list.append(lp_tensor)
        padded_completion_logprobs = torch.stack(padded_logprobs_list)

        # Ensure prompt_ids and padded_completion_ids are 2D
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if padded_completion_ids.ndim == 1:
            padded_completion_ids = padded_completion_ids.unsqueeze(0)

        new_input_ids = torch.cat([prompt_ids, padded_completion_ids], dim=1)

        # Build full-sequence logprobs: zeros for prompt, actual logprobs for completion
        prompt_logprobs_pad = torch.zeros(
            prompt_ids.shape[0], prompt_ids.shape[1], device=device, dtype=torch.float32
        )
        vllm_completion_logprobs = torch.cat(
            [prompt_logprobs_pad, padded_completion_logprobs], dim=1
        )

        # Ensure new_input_ids has the same shape as original input_ids
        original_seq_len = inputs["input_ids"].shape[1]
        new_seq_len = new_input_ids.shape[1]
        padding_added = 0
        if new_seq_len < original_seq_len:
            # Pad new_input_ids to match original length (from left)
            padding_added = original_seq_len - new_seq_len
            new_input_ids = torch.cat(
                [
                    torch.full(
                        (new_input_ids.shape[0], padding_added),
                        pad_token_id,
                        device=device,
                        dtype=new_input_ids.dtype,
                    ),
                    new_input_ids,
                ],
                dim=1,
            )
            vllm_completion_logprobs = torch.cat(
                [
                    torch.zeros(
                        vllm_completion_logprobs.shape[0],
                        padding_added,
                        device=device,
                        dtype=torch.float32,
                    ),
                    vllm_completion_logprobs,
                ],
                dim=1,
            )
        elif new_seq_len > original_seq_len:
            # Pad original inputs to match new length (from left)
            padding_needed = new_seq_len - original_seq_len
            inputs["input_ids"] = torch.cat(
                [
                    torch.full(
                        (inputs["input_ids"].shape[0], padding_needed),
                        pad_token_id,
                        device=device,
                        dtype=inputs["input_ids"].dtype,
                    ),
                    inputs["input_ids"],
                ],
                dim=1,
            )
            inputs["attention_mask"] = torch.cat(
                [
                    torch.zeros(
                        (inputs["attention_mask"].shape[0], padding_needed),
                        device=device,
                        dtype=inputs["attention_mask"].dtype,
                    ),
                    inputs["attention_mask"],
                ],
                dim=1,
            )
            inputs["labels"] = torch.cat(
                [
                    torch.full(
                        (inputs["labels"].shape[0], padding_needed),
                        -100,
                        device=device,
                        dtype=inputs["labels"].dtype,
                    ),
                    inputs["labels"],
                ],
                dim=1,
            )

        new_attention_mask = torch.ones_like(new_input_ids, device=device)
        new_labels = new_input_ids.clone()

        # Mask padding by POSITION, never by token value: with pad == eos (gold.py
        # sets pad_token = eos_token when the tokenizer defines none) value-masking
        # also removes every generated EOS from the labels and the attention mask,
        # so the model gets no KD signal on EOS and EOS-based truncation detection
        # misfires. Three pad regions, all positionally known: the left-pad block
        # of `padding_added` columns, the tokenizer's own prompt padding (from the
        # prompt's attention mask), and each row's right-padded completion tail
        # (columns at or past its valid length). A pad token generated INSIDE a
        # completion is a real token and is not masked.
        pad_positions = torch.zeros(
            new_input_ids.shape, device=device, dtype=torch.bool
        )
        if padding_added > 0:
            pad_positions[:, :padding_added] = True
        prompt_width = prompt_ids.shape[1]
        pad_positions[:, padding_added : padding_added + prompt_width] = (
            prompt_mask == 0
        )
        completion_width = padded_completion_ids.shape[1]
        valid_completion = torch.arange(completion_width, device=device).unsqueeze(
            0
        ) < torch.tensor(completion_valid_lengths, device=device).unsqueeze(1)
        pad_positions[:, padding_added + prompt_width :] = ~valid_completion
        new_labels[pad_positions] = -100
        new_attention_mask[pad_positions] = 0

        # Mask prompt tokens in labels (account for any left padding added)
        prompt_lengths = prompt_ids.shape[1] + padding_added
        new_labels[:, :prompt_lengths] = -100

        # valid_mask = new_labels != -100
        # decoded_labels = [self.tokenizer.decode(ids[mask]) for ids, mask in zip(new_input_ids, valid_mask)]
        # if new_seq_len > original_seq_len:
        #     print("new_seq_len > original_seq_len")
        #     import pdb; pdb.set_trace()

        # IMPORTANT: Preserve original text for cross-tokenizer ULD loss
        # Use prompts_text_with_special (with special tokens) for ULD loss computation
        # Extract completion texts from the generated completion IDs
        completion_texts = []
        for comp_ids in completion_ids:
            completion_text = self.processing_class.decode(
                comp_ids, skip_special_tokens=False
            )
            completion_texts.append(completion_text)

        _gen_elapsed = time.perf_counter() - _t0_gen
        self._timing_generation += _gen_elapsed
        self._timing_generation_last_call = _gen_elapsed

        # Counted only when a request was actually issued from here. `completion_ids` is this rank's
        # slice by now in every branch, so the token counts are per-rank while _gen_batch_seen is the
        # global request size -- the asymmetry is intentional and is what log() documents.
        if _gen_batch_seen is not None:
            self._gen_calls += 1
            self._gen_batch_total += _gen_batch_seen
            self._gen_prompt_rows += len(prompt_id_lists)
            self._gen_tokens += sum(len(c) for c in completion_ids)
            # `>=`, not `==`: vLLM stops AT the cap, but a re-tokenize round trip
            # (uld_rebuild_student_input) can land a hair over it, and a silent 0 here would read as
            # "nothing truncated" -- the opposite of the finding this counter exists to record.
            #
            # Guarded on None because `max_completion_length` is `generation_config.max_new_tokens`,
            # which is legitimately None for unbounded generation (it is passed straight to vLLM as
            # max_tokens). `len(c) >= None` is a TypeError, and this runs in the generation hot path
            # of every on-policy step -- a counter must not be able to kill a run.
            if max_completion_length:
                self._gen_truncated += sum(
                    1 for c in completion_ids if len(c) >= max_completion_length
                )

        return (
            new_input_ids,
            new_attention_mask,
            new_labels,
            prompts_text_with_special,
            completion_texts,
            vllm_completion_logprobs,
        )

    @staticmethod
    def cross_entropy_loss(
        student_logits,
        labels=None,
        reduction="mean",
    ):
        """
        Compute standard cross entropy loss on ground truth labels.

        Args:
            student_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size) - already shifted
            labels:
                Tensor of shape (batch_size, sequence_length) with -100 for padding tokens to ignore - already shifted
            reduction:
                Specifies the reduction to apply to the output (default: 'mean')

        Returns:
            loss: Scalar tensor with the cross entropy loss
        """
        if labels is not None:
            mask = labels != -100
            student_logits = student_logits.reshape(-1, student_logits.size(-1))[
                mask.view(-1)
            ]
            labels = labels.reshape(-1)[mask.view(-1)]

        loss = F.cross_entropy(
            student_logits,
            labels,
            reduction=reduction,
        )

        return loss

    @staticmethod
    def compute_hidden_loss(
        student_hidden_states,
        teacher_hidden_states,
        hidden_proj,
        layer_ids,
        shifted_labels,
    ):
        """
        Compute cosine similarity loss between projected teacher and student hidden states.

        Loss = 1 - cosine_similarity, so 0 = perfectly aligned, 1 = orthogonal, 2 = opposite.

        Args:
            student_hidden_states: Tuple of hidden states from student model (includes embedding layer at index 0)
            teacher_hidden_states: Tuple of hidden states from teacher model
            hidden_proj: nn.Linear projection from teacher_dim to student_dim, or None
                when student and teacher hidden sizes match (in which case no projection is applied)
            layer_ids: List of decoder layer indices to match (0-indexed)
            shifted_labels: Labels tensor already shifted (labels[:, 1:]), with -100 for tokens to ignore
        """
        # Mask: only compute loss on completion tokens
        mask = shifted_labels != -100  # [B, T-1]

        total_loss = 0.0
        for lid in layer_ids:
            # hidden_states[0] = embeddings, hidden_states[k+1] = decoder layer k output
            hs = student_hidden_states[lid + 1][:, :-1, :]  # [B, T-1, student_dim]
            ht = teacher_hidden_states[lid + 1][:, :-1, :]  # [B, T-1, teacher_dim]
            if hidden_proj is not None:
                ht = hidden_proj(ht)  # [B, T-1, student_dim]
            # cosine similarity per token, loss = 1 - cos_sim
            cos_sim = F.cosine_similarity(hs, ht, dim=-1)  # [B, T-1]
            cos_loss = 1.0 - cos_sim  # [B, T-1]
            total_loss += (cos_loss * mask).sum() / mask.sum().clamp_min(1)

        return total_loss / max(len(layer_ids), 1)

    @staticmethod
    def sampled_opd_loss(
        student_logits,
        teacher_logits,
        labels,
        temperature=1.0,
        clip_alpha=0.0,
        entropy_filter=False,
        entropy_threshold=1.0,
        vllm_logprobs=None,
        is_epsilon_low=0.5,
        is_epsilon_high=2.0,
    ):
        """
        Sampled on-policy distillation loss (REINFORCE-style policy gradient).

        Instead of summing over the full vocabulary (as in reverse KL), this only
        computes loss on the actually sampled token at each position:
            loss = -log_q(x) * reward,  where reward = (log_p(x) - log_q(x)).detach()

        When clip_alpha > 0, overly negative rewards are clipped at log(clip_alpha) / (1 - clip_alpha)
        to prevent instability from tokens where the student vastly outscores the teacher.

        When entropy_filter=True, only the top `entropy_threshold` fraction of tokens
        (by student entropy, batch-global) contribute to the loss.

        When vllm_logprobs is provided, applies truncated importance sampling to correct
        for the distribution mismatch between the vLLM inference model and the HF training
        model. Tokens with IS weights outside [is_epsilon_low, is_epsilon_high] are masked.

        Args:
            student_logits: (batch_size, seq_len, vocab_size) - already shifted
            teacher_logits: (batch_size, seq_len, vocab_size) - already shifted
            labels: (batch_size, seq_len) with -100 for ignored positions - already shifted
            temperature: softmax temperature
            clip_alpha: reward clipping parameter; when > 0, clips reward at log(clip_alpha) / (1 - clip_alpha)
            entropy_filter: whether to filter tokens by student entropy
            entropy_threshold: fraction of tokens to keep (e.g. 0.2 = top 20% highest entropy)
            vllm_logprobs: (batch_size, seq_len) per-token log-probs from vLLM, already shifted
            is_epsilon_low: lower bound for IS weight truncation
            is_epsilon_high: upper bound for IS weight truncation

        Returns:
            loss if vllm_logprobs is None, else (loss, is_stats_dict)
        """
        mask = labels != -100
        student_logits = student_logits.reshape(-1, student_logits.size(-1))[
            mask.view(-1)
        ]
        teacher_logits = teacher_logits.reshape(-1, teacher_logits.size(-1))[
            mask.view(-1)
        ]
        sampled_tokens = labels.reshape(-1)[mask.view(-1)]

        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)

        student_lp = student_log_probs.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(
            -1
        )
        teacher_lp = teacher_log_probs.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(
            -1
        )

        if entropy_filter and entropy_threshold < 1.0:
            entropy = -(student_log_probs.exp() * student_log_probs).sum(dim=-1)
            quantile = torch.quantile(entropy.float(), 1.0 - entropy_threshold)
            high_entropy_mask = entropy >= quantile
            student_lp = student_lp[high_entropy_mask]
            teacher_lp = teacher_lp[high_entropy_mask]

        is_stats = None
        if vllm_logprobs is not None:
            vllm_lp = vllm_logprobs.reshape(-1)[mask.view(-1)]
            if entropy_filter and entropy_threshold < 1.0:
                vllm_lp = vllm_lp[high_entropy_mask]

            # IS weight = q_train(x) / q_inf(x) = exp(log_q_train - log_q_inf)
            is_weights = torch.exp(student_lp.detach() - vllm_lp)
            # NaN vLLM logprob (trl's client returns None): no IS weight, so the token is
            # masked out, and kept out of the stats.
            finite = torch.isfinite(is_weights)
            is_stats = _is_weight_stats(is_weights[finite])

            # Truncation mask: keep tokens where IS weight is in [eps_low, eps_high]
            is_mask = (
                finite
                & (is_weights >= is_epsilon_low)
                & (is_weights <= is_epsilon_high)
            )
            is_stats["masked_fraction"] = 1.0 - is_mask.float().mean().item()

            student_lp = student_lp[is_mask]
            teacher_lp = teacher_lp[is_mask]
            is_w = is_weights[is_mask]

        reward = (teacher_lp - student_lp).detach()

        if clip_alpha > 0:
            reward_floor = math.log(clip_alpha) / (1 - clip_alpha)
            reward = reward.clamp(min=reward_floor)

        if student_lp.numel() == 0:
            loss = student_lp.sum() * 0.0
        else:
            if is_stats is not None:
                loss = -(student_lp * reward * is_w).mean()
            else:
                loss = -(student_lp * reward).mean()

        if is_stats is not None:
            return loss, is_stats
        return loss

    @staticmethod
    def distillm2_loss(
        student_logits,
        teacher_logits,
        labels=None,
        temperature=1.0,
        alpha=0.0,
        on_policy=None,
    ):
        """
        Compute DISTILLM-2 style loss with different KL divergences for on-policy and off-policy data.

        On-policy data (student-generated): Uses reverse KL (mode-seeking)
        Off-policy data (dataset): Uses forward KL (mean-seeking)

        Args:
            student_logits: Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits: Tensor of shape (batch_size, sequence_length, vocab_size)
            labels: Tensor of shape (batch_size, sequence_length) with -100 for padding tokens
            temperature: Softmax temperature (default: 1.0)
            alpha: Skewing parameter for interpolating distributions (default: 0.0)
            on_policy: List of booleans indicating which samples are on-policy

        Returns:
            loss: Scalar tensor (batchmean reduction)
        """
        batch_size, seq_len, vocab_size = student_logits.shape

        if labels is not None:
            mask = labels != -100
            flat_mask = mask.view(-1)
            student_logits = student_logits.reshape(-1, vocab_size)[flat_mask]
            teacher_logits = teacher_logits.reshape(-1, vocab_size)[flat_mask]
            num_tokens = mask.sum()

            on_policy_mask = torch.tensor(
                on_policy, dtype=torch.bool, device=student_logits.device
            )
            token_on_policy = (
                on_policy_mask.unsqueeze(1).expand(-1, seq_len).reshape(-1)[flat_mask]
            )
        else:
            num_tokens = student_logits.size(0) * student_logits.size(1)
            student_logits = student_logits.reshape(-1, vocab_size)
            teacher_logits = teacher_logits.reshape(-1, vocab_size)
            on_policy_mask = torch.tensor(
                on_policy, dtype=torch.bool, device=student_logits.device
            )
            token_on_policy = (
                on_policy_mask.unsqueeze(1).expand(-1, seq_len).reshape(-1)
            )

        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)

        total_loss = 0.0

        # On-policy tokens: Reverse KL (mode-seeking)
        # Reverse KL: sum_x p_student(x) * (log p_student(x) - log p_teacher(x))
        if token_on_policy.any():
            s_on = student_log_probs[token_on_policy]
            t_on = teacher_log_probs[token_on_policy]
            if alpha > 0:
                t_on = torch.logaddexp(
                    s_on + math.log(alpha), t_on + math.log(1 - alpha)
                )
            total_loss += (torch.exp(s_on) * (s_on - t_on)).sum()

        # Off-policy tokens: Forward KL (mean-seeking)
        # Forward KL: sum_x p_teacher(x) * (log p_teacher(x) - log p_student(x))
        off_policy = ~token_on_policy
        if off_policy.any():
            s_off = student_log_probs[off_policy]
            t_off = teacher_log_probs[off_policy]
            if alpha > 0:
                s_off = torch.logaddexp(
                    s_off + math.log(1 - alpha), t_off + math.log(alpha)
                )
            total_loss += F.kl_div(s_off, t_off, reduction="sum", log_target=True)

        return total_loss / num_tokens

    @staticmethod
    def generalized_jsd_loss(
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1.0,
        alpha=0.0,
        use_adaptive_kld=False,
        use_distillm2_like=False,
        use_reversed_distillm2_like=False,
        use_kl_interpolation=False,
        reduction="batchmean",
        logits_are_probs=False,
        on_policy=False,
        tokenizer=None,
    ):
        """
        Compute the generalized Jensen-Shannon Divergence loss for knowledge distillation using F.kl_div. See Eq. (1)
        of https://huggingface.co/papers/2306.13649 for the definition.

        Args:
            student_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            labels:
                Tensor of shape (batch_size, sequence_length) with -100 for padding tokens to ignore when computing
                loss
            beta:
                Interpolation coefficient between 0 and 1 (default: 0.5)
            temperature:
                Softmax temperature (default: 1.0)
            use_kl_interpolation:
                When True and 0 < beta < 1 with alpha == 0 (and no adaptive KLD / distillm2-like),
                use the convex FKL/RKL interpolation `(1 - beta) * FKL + beta * RKL` instead of the
                standard mixture-distribution generalized JSD. Default: False.
            reduction:
                Specifies the reduction to apply to the output (default: 'batchmean')

        Returns:
            loss: Scalar tensor with the generalized JSD loss
        """

        # Masking first
        if labels is not None:
            mask = labels != -100
            # Flatten and filter early
            student_logits = student_logits.reshape(-1, student_logits.size(-1))[
                mask.view(-1)
            ]
            teacher_logits = teacher_logits.reshape(-1, teacher_logits.size(-1))[
                mask.view(-1)
            ]

        if logits_are_probs:
            student_log_probs = torch.log(student_logits.clamp_min(1e-8))
            teacher_log_probs = torch.log(teacher_logits.clamp_min(1e-8))
        else:
            # Apply temperature scaling to logits before computing probabilities
            student_logits = student_logits / temperature
            teacher_logits = teacher_logits / temperature
            # Compute log probabilities for student and probabilities for teacher
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        # labels = labels.reshape(-1)[mask.view(-1)]
        # _, teacher_topk_indices = torch.topk(teacher_log_probs, k=10, dim=-1)
        # teacher_topk_ratio = sum([labels[i] in teacher_topk_indices[i] for i in range(labels.shape[0])]) / labels.shape[0]
        # print(f"teacher top-k: {100.0 * teacher_topk_ratio}")

        # _, student_topk_indices = torch.topk(student_log_probs, k=10, dim=-1)
        # student_topk_ratio = sum([labels[i] in student_topk_indices[i] for i in range(labels.shape[0])]) / labels.shape[0]
        # print(f"student top-k: {100.0 * student_topk_ratio}")

        # print(f"on-policy: {on_policy}")
        # import pdb; pdb.set_trace()

        if use_distillm2_like:
            """
            Compute DISTILLM-2 like loss
            """
            if on_policy:
                """Compute Reverse KL"""
                if alpha > 0:
                    # skew teacher_log_probs
                    teacher_log_probs = torch.logaddexp(
                        student_log_probs + math.log(alpha),
                        teacher_log_probs + math.log(1 - alpha),
                    )
                jsd = torch.exp(student_log_probs) * (
                    student_log_probs - teacher_log_probs
                )
            else:
                """Compute Forward KL"""
                if alpha > 0:
                    # skew student_log_probs
                    student_log_probs = torch.logaddexp(
                        student_log_probs + math.log(1 - alpha),
                        teacher_log_probs + math.log(alpha),
                    )
                jsd = F.kl_div(
                    student_log_probs,
                    teacher_log_probs,
                    reduction="none",
                    log_target=True,
                )
        elif use_reversed_distillm2_like:
            """
            Compute reversed DISTILLM-2 like loss
            """
            if not on_policy:
                """Compute Reverse KL"""
                if alpha > 0:
                    # skew teacher_log_probs
                    teacher_log_probs = torch.logaddexp(
                        student_log_probs + math.log(alpha),
                        teacher_log_probs + math.log(1 - alpha),
                    )
                jsd = torch.exp(student_log_probs) * (
                    student_log_probs - teacher_log_probs
                )
            else:
                """Compute Forward KL"""
                if alpha > 0:
                    # skew student_log_probs
                    student_log_probs = torch.logaddexp(
                        student_log_probs + math.log(1 - alpha),
                        teacher_log_probs + math.log(alpha),
                    )
                jsd = F.kl_div(
                    student_log_probs,
                    teacher_log_probs,
                    reduction="none",
                    log_target=True,
                )
        elif alpha > 0:
            if beta == 0:
                # skew student_log_probs
                student_log_probs = torch.logaddexp(
                    student_log_probs + math.log(1 - alpha),
                    teacher_log_probs + math.log(alpha),
                )
                jsd = F.kl_div(
                    student_log_probs,
                    teacher_log_probs,
                    reduction="none",
                    log_target=True,
                )
            elif beta == 1:
                # skew teacher_log_probs
                teacher_log_probs = torch.logaddexp(
                    student_log_probs + math.log(alpha),
                    teacher_log_probs + math.log(1 - alpha),
                )
                # compute reverse KL
                jsd = torch.exp(student_log_probs) * (
                    student_log_probs - teacher_log_probs
                )
            else:
                # skew student_log_probs
                skewed_student_log_probs = torch.logaddexp(
                    student_log_probs + math.log(1 - alpha),
                    teacher_log_probs + math.log(alpha),
                )
                # skew teacher_log_probs
                skewed_teacher_log_probs = torch.logaddexp(
                    student_log_probs + math.log(alpha),
                    teacher_log_probs + math.log(1 - alpha),
                )
                jsd = (1 - beta) * F.kl_div(
                    skewed_student_log_probs,
                    teacher_log_probs,
                    reduction="none",
                    log_target=True,
                ) + beta * torch.exp(student_log_probs) * (
                    student_log_probs - skewed_teacher_log_probs
                )
        elif use_adaptive_kld:
            # Sort teacher log probabilities in descending order
            sorted_log_probs, sorted_indices = torch.sort(
                teacher_log_probs, descending=True, dim=-1
            )

            # Convert sorted log probs to probs, compute cumulative probs, and find indices where cumsum <= 0.5
            top_mask_sorted = torch.cumsum(torch.exp(sorted_log_probs), dim=-1) <= 0.5

            # Create mask in original vocabulary order
            top_mask = torch.zeros_like(teacher_log_probs, dtype=torch.bool)
            top_mask.scatter_(-1, sorted_indices, top_mask_sorted)

            # Compute both KLDs and combine (avoid creating 4 separate masked tensors)
            jsd = torch.where(
                top_mask,
                # KLD for high-prob tokens
                F.kl_div(
                    student_log_probs,
                    teacher_log_probs,
                    reduction="none",
                    log_target=True,
                ),
                # Reverse KLD for low-prob tokens
                torch.exp(student_log_probs) * (student_log_probs - teacher_log_probs),
            )
        else:
            if beta == 0:
                jsd = F.kl_div(
                    student_log_probs,
                    teacher_log_probs,
                    reduction="none",
                    log_target=True,
                )
            elif beta == 1:
                # compute reverse KL
                jsd = torch.exp(student_log_probs) * (
                    student_log_probs - teacher_log_probs
                )
            elif use_kl_interpolation:
                # Convex FKL/RKL interpolation (degenerate limit of the alpha > 0 branch with alpha = 0).
                fkl = F.kl_div(
                    student_log_probs,
                    teacher_log_probs,
                    reduction="none",
                    log_target=True,
                )
                rkl = torch.exp(student_log_probs) * (
                    student_log_probs - teacher_log_probs
                )
                jsd = (1 - beta) * fkl + beta * rkl
            else:
                # Compute the log of the mixture distribution
                # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
                mixture_log_probs = torch.logaddexp(
                    student_log_probs + math.log(1 - beta),
                    teacher_log_probs + math.log(beta),
                )

                # Compute KL divergences using F.kl_div
                # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
                kl_teacher = F.kl_div(
                    mixture_log_probs,
                    teacher_log_probs,
                    reduction="none",
                    log_target=True,
                )
                kl_student = F.kl_div(
                    mixture_log_probs,
                    student_log_probs,
                    reduction="none",
                    log_target=True,
                )

                # Compute the Generalized Jensen-Shannon Divergence
                jsd = beta * kl_teacher + (1 - beta) * kl_student

        # Apply reduction
        if reduction == "batchmean":
            return (
                jsd.sum() / mask.sum()
                if labels is not None
                else jsd.sum() / jsd.size(0)
            )
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd

    def _uld_alignment_step_end(self, state) -> None:
        """One optimizer step of ULDAlignmentGuard, from _ULDAlignmentGuardCallback. The
        window defaults to the logging window's length (see ULDAlignmentGuard for when the
        two cover different steps)."""
        self._uld_alignment_guard.end_step(
            self.uld_loss_fn,
            state.global_step
            + 1,  # the step being taken; global_step counts completed ones
            _guard_reduce(self.accelerator),
            window_steps=int(
                getattr(state, "logging_steps", 0) or self.args.logging_steps or 1
            ),
        )

    def _uld_alignment_train_end(self) -> None:
        """ULDAlignmentGuard's train-end judgement, from _ULDAlignmentGuardCallback's
        on_train_end, which every rank reaches once."""
        self._uld_alignment_guard.end_train(
            self.uld_loss_fn, _guard_reduce(self.accelerator)
        )

    def _student_rows_end_with_eos(self, inputs) -> list[bool]:
        """Per row: did the student's sequence end with EOS, i.e. was it NOT truncated?

        Hoisted out of compute_loss because the per-turn render needs it BEFORE the teacher
        inputs are built (the last turn's EOS is dropped for a truncated row) while the
        single-region builder needs it after. Two copies of this would be two chances for
        the teacher's final position to disagree with the student's.
        """
        eos_id = self.processing_class.eos_token_id
        out = []
        for row in range(inputs["input_ids"].size(0)):
            s_mask = inputs["attention_mask"][row].bool()
            if s_mask.any():
                s_last = inputs["input_ids"][
                    row, s_mask.nonzero(as_tuple=True)[0][-1]
                ].item()
                out.append(s_last == eos_id)
            else:
                out.append(False)
        return out

    def _pack_teacher_regions(
        self, hidden, teacher_input_ids, teacher_labels, teacher_spans
    ):
        """Gather the positions the per-turn regions name, packed into [B, Kmax, ...].

        Each region (start, size) contributes positions start-1 .. start+size-1 inclusive.
        The returned labels mark the FIRST position of every block as ignored and the rest
        with the block's token ids, so `contiguous_label_spans` recovers exactly one span
        per region at (block_offset + 1, size) -- the same extraction the student side uses,
        which is why the loss needs no teacher-specific coordinate handling.

        Rows are padded to the batch's widest region set with index 0 and label -100. Those
        columns are unreachable: a span is only ever read through the label spans, and
        padding carries no label.
        """
        B, T, D = hidden.shape
        device = hidden.device
        idx_rows, lab_rows = [], []
        for b in range(B):
            positions, labels = [], []
            for start, size in teacher_spans[b]:
                lo = max(0, start - 1)
                hi = min(T, start + size)
                if hi - lo < 2:  # a region with no predictable position
                    continue
                block = list(range(lo, hi))
                positions.extend(block)
                # first position of the block predicts, it is not predicted
                labels.append(-100)
                labels.extend(teacher_input_ids[b, lo + 1 : hi].tolist())
            idx_rows.append(positions)
            lab_rows.append(labels)
        kmax = max((len(p) for p in idx_rows), default=0)
        kmax = max(kmax, 1)
        idx = torch.zeros(B, kmax, dtype=torch.long, device=device)
        lab = torch.full((B, kmax), -100, dtype=teacher_labels.dtype, device=device)
        for b, (positions, labels) in enumerate(zip(idx_rows, lab_rows)):
            if positions:
                idx[b, : len(positions)] = torch.tensor(
                    positions, dtype=torch.long, device=device
                )
                lab[b, : len(labels)] = torch.tensor(
                    labels, dtype=teacher_labels.dtype, device=device
                )
        packed_hidden = hidden.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))
        packed_ids = teacher_input_ids.gather(1, idx)
        return packed_hidden, packed_ids, lab

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):

        if self.use_uld_loss and self.teacher_tokenizer is not None:
            # Initialised before the branch chain because only ONE of those branches can
            # build turn segments. Left None by the others, and refused below if per-turn
            # regions were asked for and a branch that cannot honour them was taken --
            # falling back to one region per row is precisely the silent 46.6% token loss
            # this flag exists to end.
            teacher_segment_rows = None
            # Per-sample: append EOS on the teacher side only if the student sequence also ends
            # with EOS (i.e. was not truncated). Computed once here rather than at each use --
            # the per-turn render needs it while building segments, the single-region builder
            # needs it afterwards, and two copies are two chances for the teacher's final
            # position to disagree with the student's.
            append_eos = (
                None
                if self.args.uld_reuse_student_input
                else self._student_rows_end_with_eos(inputs)
            )
            if self.args.uld_reuse_student_input:
                teacher_input_ids = inputs["input_ids"]
                teacher_labels = inputs["labels"]
                teacher_attention_mask = inputs["attention_mask"]
                # Not inputs["prompts"].shape[1], the batch's longest prompt: these rows are
                # left-padded, so a row's completion can start before that column, and the
                # crop below would cut into it.
                teacher_prompt_length = earliest_completion_column(inputs["labels"])
            elif (
                "messages" in inputs
                and inputs["messages"] is not None
                and inputs["messages"][0] is not None
            ):
                prompt_texts = []
                completion_texts = []
                tools_list = inputs.get("tools", None)
                # PER-TURN REGIONS. When on, the teacher is rendered as one supervised region
                # per assistant turn, mirroring what the student's collator already does, and
                # `teacher_segment_rows` replaces the (prompt_texts, completion_texts) pair.
                per_turn = getattr(self.args, "uld_per_turn_regions", False)
                if per_turn and self.args.is_gptoss_teacher:
                    raise ValueError(
                        "uld_per_turn_regions is not supported with is_gptoss_teacher: the gpt-oss "
                        "path rewrites the single completion's channel markers "
                        "(removeprefix('<|channel|>final<|message|>') and the analysis-channel "
                        "insertion), which is a one-region-per-row transformation. Supporting it "
                        "means deciding what those markers mean for an intermediate turn, and "
                        "guessing that silently is how a teacher region ends up off by a marker."
                    )
                if (
                    per_turn
                    and self.args.uld_rebuild_student_input_with_teacher_template
                ):
                    raise ValueError(
                        "uld_per_turn_regions is not supported with "
                        "uld_rebuild_student_input_with_teacher_template: that debug path rebuilds "
                        "the STUDENT from (prompt_texts, completion_texts), which the per-turn "
                        "render never fills -- it emits turn segments instead. It would rebuild "
                        "the student batch from two empty lists."
                    )
                teacher_segment_rows = [] if per_turn else None
                for i, msgs in enumerate(inputs["messages"]):
                    # system_prompt = "You are a helpful assistant. Focus on diversity in your response."
                    # msgs = [{"role": "system", "content": system_prompt}] + msgs

                    # Qwen3.5 chat template uses |items on tool_call.arguments, requiring a dict not a JSON string
                    if (
                        self.args.teacher_model_name_or_path
                        and self.args.teacher_model_name_or_path.rstrip("/").endswith(
                            "Qwen3.5-4B"
                        )
                    ):
                        for msg in msgs:
                            if "tool_calls" in msg and msg["tool_calls"]:
                                for tc in msg["tool_calls"]:
                                    if isinstance(tc.get("arguments"), str):
                                        tc["arguments"] = json.loads(tc["arguments"])
                                    if "function" in tc and isinstance(
                                        tc["function"].get("arguments"), str
                                    ):
                                        tc["function"]["arguments"] = json.loads(
                                            tc["function"]["arguments"]
                                        )

                    prompt_msgs = msgs[:-1]
                    tools_raw = tools_list[i] if tools_list is not None else None
                    tools = json.loads(tools_raw) if tools_raw else None
                    tools_kwargs = {"tools": tools} if tools else {}
                    if self.args.is_gptoss_teacher:
                        teacher_prompt = self.teacher_tokenizer.apply_chat_template(
                            prompt_msgs,
                            tokenize=False,
                            add_generation_prompt=True,
                            reasoning_effort="low",
                            **tools_kwargs,
                        )
                        teacher_full = self.teacher_tokenizer.apply_chat_template(
                            msgs,
                            tokenize=False,
                            add_generation_prompt=False,
                            reasoning_effort="low",
                            **tools_kwargs,
                        )
                        teacher_completion = teacher_full[len(teacher_prompt) :]

                        # Strip final channel prefix/suffix
                        teacher_completion = teacher_completion.removeprefix(
                            "<|channel|>final<|message|>"
                        ).removesuffix("<|return|>")
                        # Add empty analysis channel
                        if self.args.add_empty_analysis:
                            teacher_prompt += "<|channel|>analysis<|message|><|end|>"
                        # Add final channel prefix
                        teacher_prompt += "<|channel|>final<|message|>"
                    else:
                        # For <think> reasoning data the teacher must render with enable_thinking=True,
                        # else the prompt gets an empty <think></think> while the full render has a filled
                        # <think>...</think> -> full.startswith(prompt) is False -> the slice below is
                        # corrupted -> teacher labels misalign -> ULD loss 0. Gated by config; default
                        # False preserves prior (non-thinking) behavior. The turn-suffix (computed with
                        # enable_thinking=False) still matches the completion tail and strips <|im_end|>\n.
                        # Per ROW, not per run: the corpus is bimodal and one global value renders the
                        # other half wrong. The config flag stays the gate, so an arm that leaves it
                        # false renders exactly as it did before this line existed.
                        # The nothink half's two prompts still spell the empty block differently
                        # (Granite `<think></think>`, Qwen `<think>\n\n</think>\n\n`, confirmed directly).
                        # That is prompt-side and never enters an alignment group: the loss aligns
                        # the answer region, and those are byte-identical, confirmed by a separate
                        # direct measurement.
                        _teacher_think = (
                            self.args.uld_teacher_enable_thinking
                            and _row_teacher_enable_thinking(msgs)
                        )
                        if per_turn:
                            # One region per assistant turn. The per-row thinking flag, the tools
                            # and the turn suffix are all reused unchanged -- only the number of
                            # regions changes, which is the point: nothing about HOW a turn is
                            # rendered should differ between the two modes, or the modes would not
                            # be comparable.
                            # Extra template variables ride in the SAME dict the splitter splats
                            # into every apply_chat_template call, so the prefix, through and
                            # full renders it compares are produced under identical template
                            # state -- the only way that comparison stays valid. Merged here
                            # rather than into tools_kwargs above so the single-region path below
                            # is byte-identical to before.
                            _turn_stats: dict = {}
                            teacher_segment_rows.append(
                                teacher_turn_segments(
                                    self.teacher_tokenizer,
                                    msgs,
                                    {
                                        **tools_kwargs,
                                        **self._uld_teacher_template_kwargs,
                                    },
                                    enable_thinking=_teacher_think,
                                    turn_suffix=self._teacher_turn_suffix,
                                    eos_str=self.teacher_tokenizer.eos_token,
                                    append_eos=append_eos[i],
                                    stats=_turn_stats,
                                )
                            )
                            self._uld_relaxed_boundaries = getattr(
                                self, "_uld_relaxed_boundaries", 0
                            ) + _turn_stats.get("relaxed_ws_boundaries", 0)
                            continue
                        teacher_prompt = self.teacher_tokenizer.apply_chat_template(
                            prompt_msgs,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=_teacher_think,
                            **tools_kwargs,
                        )
                        teacher_full = self.teacher_tokenizer.apply_chat_template(
                            msgs,
                            tokenize=False,
                            add_generation_prompt=False,
                            enable_thinking=_teacher_think,
                            **tools_kwargs,
                        )
                        teacher_completion = teacher_full[len(teacher_prompt) :]
                        # No <think> move here, deliberately, and do not restore one. It existed because a
                        # GLOBAL enable_thinking put the teacher's opening "<think>\n" in the prompt while the
                        # student's per-row flag (:1875) left the student's in the completion. With the flag
                        # derived per row above, both templates put it in the PROMPT, so both answer regions
                        # begin at the same byte: measured byte-identical on 5247/5247 mixture rows,
                        # confirmed directly, and 94.0% of student answer tokens covered by the trainer's own matcher
                        # against 0.0% before, confirmed by a separate direct measurement. Re-adding the move would offset the teacher by
                        # two tokens and overflow every alignment group again.
                        if self._teacher_turn_suffix and teacher_completion.endswith(
                            self._teacher_turn_suffix
                        ):
                            teacher_completion = teacher_completion[
                                : -len(self._teacher_turn_suffix)
                            ]

                    prompt_texts.append(teacher_prompt)
                    completion_texts.append(teacher_completion)
            elif (
                "original_prompt_text" in inputs
                and "original_completion_text" in inputs
            ):
                # Fallback: use student-formatted text (existing behavior)
                prompt_texts = inputs["original_prompt_text"]
                completion_texts = inputs["original_completion_text"]
            else:
                # Last resort fallback: decode student input_ids
                # WARNING: This may not work perfectly for cross-tokenizer distillation
                full_sequences = inputs["input_ids"]
                full_texts = self.processing_class.batch_decode(
                    full_sequences, skip_special_tokens=False
                )

                # Try to split prompt/completion using original prompt length
                prompt_lengths = inputs["prompts"].shape[1]
                prompt_texts = self.processing_class.batch_decode(
                    inputs["prompts"], skip_special_tokens=False
                )
                completion_texts = [
                    full.replace(prompt, "", 1)
                    for full, prompt in zip(full_texts, prompt_texts, strict=True)
                ]

            teacher_spans = None
            if (
                getattr(self.args, "uld_per_turn_regions", False)
                and teacher_segment_rows is None
            ):
                raise ValueError(
                    "uld_per_turn_regions=True but the teacher was rendered by a path that cannot "
                    "produce per-turn regions. Per-turn rendering needs the ROW's messages: it is "
                    "available only on the `messages` branch. uld_reuse_student_input=True has no "
                    "teacher render at all, and the two fallbacks (original_prompt_text, and "
                    "decoding student input_ids) carry a prompt/completion split already collapsed "
                    "to one region. Continuing would supervise the last turn only, which is the "
                    "behaviour this flag was set to avoid."
                )
            if not self.args.uld_reuse_student_input:
                if teacher_segment_rows is not None:
                    (
                        teacher_input_ids,
                        teacher_labels,
                        teacher_attention_mask,
                        teacher_spans,
                    ) = build_teacher_inputs_from_turn_segments(
                        self.teacher_tokenizer,
                        teacher_segment_rows,
                    )
                    # No teacher_prompt_length: there is no column all regions start at. Set to
                    # None so any code path that still reads one fails loudly instead of
                    # cropping the batch at a number that means nothing here.
                    teacher_prompt_length = None
                else:
                    (
                        teacher_input_ids,
                        teacher_labels,
                        teacher_attention_mask,
                        teacher_prompt_length,
                    ) = build_teacher_inputs_from_texts_with_eos_control(
                        self.teacher_tokenizer,
                        prompt_texts,
                        completion_texts,
                        append_eos=append_eos,
                    )

                teacher_input_ids = teacher_input_ids.to(self.accelerator.device)
                teacher_labels = teacher_labels.to(self.accelerator.device)
                teacher_attention_mask = teacher_attention_mask.to(
                    self.accelerator.device
                )

            ## rebuild with teacher chat template (for debugging)
            if self.args.uld_rebuild_student_input_with_teacher_template:
                (
                    student_input_ids_rebuilt,
                    student_labels_rebuilt,
                    student_attention_mask_rebuilt,
                    student_prompt_length_rebuilt,
                ) = build_teacher_inputs_from_texts_with_eos_control(
                    self.processing_class,
                    prompt_texts,
                    completion_texts,
                    append_eos=append_eos,
                )
                inputs["input_ids"] = student_input_ids_rebuilt.to(
                    self.accelerator.device
                )
                inputs["labels"] = student_labels_rebuilt.to(self.accelerator.device)
                inputs["attention_mask"] = student_attention_mask_rebuilt.to(
                    self.accelerator.device
                )
                inputs["prompts"] = inputs["input_ids"][
                    :, :student_prompt_length_rebuilt
                ]

            outputs_student = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            )

            # Run teacher forward capturing hidden states (avoid materializing full [B, seq_len, V] logits)
            self.teacher_model.eval()
            unwrapped_teacher = self.accelerator.unwrap_model(self.teacher_model)
            teacher_head = unwrapped_teacher.get_output_embeddings()
            captured_teacher_hidden = {}

            def _capture_teacher_hidden(module, args, kwargs):
                captured_teacher_hidden["value"] = args[0]
                return args, kwargs

            hook_t = teacher_head.register_forward_pre_hook(
                _capture_teacher_hidden, with_kwargs=True
            )
            orig_t_fwd = teacher_head.forward
            teacher_head.forward = lambda x: x
            try:
                with torch.no_grad():
                    self.teacher_model(
                        input_ids=teacher_input_ids,
                        attention_mask=teacher_attention_mask,
                        use_cache=False,
                    )
            finally:
                hook_t.remove()
                teacher_head.forward = orig_t_fwd

            if teacher_spans is not None:
                # PER-TURN: GATHER, NOT CROP. The crop below works because the single-region
                # builder left-pads every prompt to a common length, so every row's answer
                # starts at the same column. Per-turn regions are scattered and differ per row,
                # so there is no such column; a crop wide enough to contain them all would be
                # the whole sequence, and at max_length 32768 a full [B, T, V_teacher] tensor is
                # ~10 GB in bf16 for a single row.
                #
                # Each region contributes positions start-1 .. start+size-1 INCLUSIVE (size+1
                # of them). That leading position is what makes the packed layout obey the same
                # convention as the uncropped one: within a block, logits[0..size-1] predict
                # ids[1..size], so the loss's `logits[t_start-1 : t_start+t_size-1]` and
                # `ids[t_start : t_start+t_size]` are both correct with t_start = block+1, and
                # the loop needs no separate coordinate system for the teacher.
                teacher_hidden, teacher_input_ids, teacher_labels = (
                    self._pack_teacher_regions(
                        captured_teacher_hidden["value"],
                        teacher_input_ids,
                        teacher_labels,
                        teacher_spans,
                    )
                )
            else:
                # Slice hidden states to completion-only (from teacher_prompt_length - 1 onward)
                # Position teacher_prompt_length - 1 is the last prompt token whose logit predicts the first answer token
                teacher_hidden = captured_teacher_hidden["value"][
                    :, teacher_prompt_length - 1 :, :
                ]
            del captured_teacher_hidden

            # Apply logits scaling (Granite models divide logits by config.logits_scaling)
            teacher_scaling = getattr(unwrapped_teacher.config, "logits_scaling", 1.0)
            if teacher_scaling != 1.0:
                teacher_hidden = teacher_hidden / teacher_scaling

            # Project through lm_head (only completion positions → saves [B, prompt_len, V] memory)
            gather_params = [teacher_head.weight]
            teacher_bias = getattr(teacher_head, "bias", None)
            if teacher_bias is not None:
                gather_params.append(teacher_bias)

            if self.is_deepspeed_enabled:
                import deepspeed

                ctx = deepspeed.zero.GatheredParameters(
                    gather_params, modifier_rank=None
                )
            else:
                ctx = nullcontext()

            with torch.no_grad(), ctx:
                teacher_logits = F.linear(
                    teacher_hidden, teacher_head.weight, teacher_bias
                )

            del teacher_hidden

            # Slice teacher_input_ids and teacher_labels to match the sliced logits.
            # Already done by _pack_teacher_regions on the per-turn path, which has to gather
            # the ids and labels with the SAME index tensor as the hidden states or the two
            # would describe different positions.
            if teacher_spans is None:
                teacher_input_ids = teacher_input_ids[:, teacher_prompt_length - 1 :]
                teacher_labels = teacher_labels[:, teacher_prompt_length - 1 :]
        else:
            if self.use_liger_gkd_loss:
                # Update beta/alpha dynamically if using a schedule
                if self.args.beta_schedule == "linear":
                    self.liger_jsd_loss.beta = self._get_current_beta()
                if self.args.alpha_schedule == "linear":
                    self.liger_jsd_loss.alpha = self._get_current_alpha()

                # Capture hidden states via hooks and replace lm_head with identity to avoid
                # materializing full [B*T, V] logits. Goes through model wrappers so ZeRO-3
                # parameter gathering and autocast work correctly.
                unwrapped_student = self.accelerator.unwrap_model(model)
                student_head = unwrapped_student.get_output_embeddings()
                captured_student = {}

                def _capture_student(module, args, kwargs):
                    captured_student["value"] = args[0]
                    return args, kwargs

                hook_s = student_head.register_forward_pre_hook(
                    _capture_student, with_kwargs=True
                )
                orig_s_fwd = student_head.forward
                student_head.forward = lambda x: x
                try:
                    model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        use_cache=False,
                    )
                finally:
                    hook_s.remove()
                    student_head.forward = orig_s_fwd

                self.teacher_model.eval()
                unwrapped_teacher = self.accelerator.unwrap_model(self.teacher_model)
                teacher_head = unwrapped_teacher.get_output_embeddings()
                captured_teacher = {}

                def _capture_teacher(module, args, kwargs):
                    captured_teacher["value"] = args[0]
                    return args, kwargs

                hook_t = teacher_head.register_forward_pre_hook(
                    _capture_teacher, with_kwargs=True
                )
                orig_t_fwd = teacher_head.forward
                teacher_head.forward = lambda x: x
                try:
                    with torch.no_grad():
                        self.teacher_model(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            use_cache=False,
                        )
                finally:
                    hook_t.remove()
                    teacher_head.forward = orig_t_fwd

                # hidden states (shifted)
                student_hidden = captured_student["value"][:, :-1]
                teacher_hidden = captured_teacher["value"][:, :-1]

                # Apply model-specific logits scaling (e.g. Granite divides logits by config.logits_scaling)
                student_scaling = getattr(
                    unwrapped_student.config, "logits_scaling", 1.0
                )
                if student_scaling != 1.0:
                    student_hidden = student_hidden / student_scaling

                teacher_scaling = getattr(
                    unwrapped_teacher.config, "logits_scaling", 1.0
                )
                if teacher_scaling != 1.0:
                    teacher_hidden = teacher_hidden / teacher_scaling

                # Compute hidden state matching loss before reshape destroys [B, T-1, D] structure.
                # The liger path always uses the last decoder layer (captured via lm_head pre-hook);
                # hidden_loss_layers is ignored here.
                if self.args.use_hidden_loss:
                    shift_labels_for_hidden = inputs["labels"][:, 1:]
                    mask_h = shift_labels_for_hidden != -100
                    ht = (
                        teacher_hidden
                        if self.hidden_proj is None
                        else self.hidden_proj(teacher_hidden)
                    )
                    cos_sim = F.cosine_similarity(student_hidden, ht, dim=-1)
                    hidden_loss = (
                        (1.0 - cos_sim) * mask_h
                    ).sum() / mask_h.sum().clamp_min(1)

                # Mask to completion tokens only before fused loss (same as non-liger path, line 990-991).
                # This dramatically reduces the number of tokens processed through the expensive
                # JSD computation and avoids CUDA OOM from torch.func intermediate retention.
                shift_labels = inputs["labels"][:, 1:]
                mask = shift_labels != -100
                student_hidden = student_hidden.reshape(-1, student_hidden.size(-1))[
                    mask.view(-1)
                ]
                teacher_hidden = teacher_hidden.reshape(-1, teacher_hidden.size(-1))[
                    mask.view(-1)
                ]
                true_labels = shift_labels.reshape(-1)[mask.view(-1)]

                # Free full [B, T, D] hidden state tensors before gathering lm_head weights
                del captured_student, captured_teacher
                empty_cache()

                # Gather sharded weights for fused loss (ZeRO-3)
                gather_params = [student_head.weight, teacher_head.weight]
                student_bias = getattr(student_head, "bias", None)
                teacher_bias = getattr(teacher_head, "bias", None)
                if student_bias is not None:
                    gather_params.append(student_bias)
                if teacher_bias is not None:
                    gather_params.append(teacher_bias)

                if self.is_deepspeed_enabled:
                    import deepspeed

                    ctx = deepspeed.zero.GatheredParameters(
                        gather_params, modifier_rank=None
                    )
                else:
                    ctx = nullcontext()

                with ctx:
                    loss = self.liger_jsd_loss(
                        student_input=student_hidden,
                        student_weight=student_head.weight,
                        teacher_input=teacher_hidden,
                        teacher_weight=teacher_head.weight,
                        true_labels=true_labels,
                        student_bias=student_bias,
                        teacher_bias=teacher_bias,
                    )

                # Release hidden states after loss computation
                del student_hidden, teacher_hidden, true_labels

                # Add hidden state matching loss (auxiliary) to liger JSD loss
                if self.args.use_hidden_loss:
                    lmbda_weight = 1.0 - self._get_current_lmbda()
                    loss = (
                        loss + self.args.hidden_loss_gamma * lmbda_weight * hidden_loss
                    )
                    ga = max(1, int(self.args.gradient_accumulation_steps))
                    self._hidden_loss_total += hidden_loss.item()
                    self._hidden_loss_step_equiv += 1.0 / ga
            else:
                # Original behavior for same tokenizer or when teacher_tokenizer is not provided
                _need_hidden = self.args.use_hidden_loss
                outputs_student = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=_need_hidden,
                )

                self.teacher_model.eval()
                with torch.no_grad():
                    outputs_teacher = self.teacher_model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        output_hidden_states=_need_hidden,
                    )

                # Let the loss function handle masking - just do standard next-token shift
                shifted_student_logits = outputs_student.logits[:, :-1, :]
                shifted_teacher_logits = outputs_teacher.logits[:, :-1, :]
                shifted_labels = inputs["labels"][:, 1:]

                if self.args.use_ce_loss:
                    loss = self.cross_entropy_loss(
                        student_logits=shifted_student_logits,
                        labels=shifted_labels,
                    )
                elif self.args.use_sampled_opd_loss:
                    progress = self.state.global_step / max(self.state.max_steps, 1)
                    switched = (
                        self.args.opd_switch_steps > 0
                        and progress >= self.args.opd_switch_steps
                    )
                    vllm_lp = None
                    if self.args.opd_importance_sampling and "vllm_logprobs" in inputs:
                        vllm_lp = inputs["vllm_logprobs"][
                            :, 1:
                        ]  # shift to align with shifted_labels

                    opd_result = self.sampled_opd_loss(
                        student_logits=shifted_student_logits,
                        teacher_logits=shifted_teacher_logits,
                        labels=shifted_labels,
                        temperature=self.args.temperature,
                        clip_alpha=self.args.clip_alpha,
                        entropy_filter=switched,
                        entropy_threshold=self.args.opd_entropy_threshold,
                        vllm_logprobs=vllm_lp,
                        is_epsilon_low=self.args.opd_is_epsilon_low,
                        is_epsilon_high=self.args.opd_is_epsilon_high,
                    )

                    if isinstance(opd_result, tuple):
                        loss, is_stats = opd_result
                        self._is_stats_accum["mean"] += is_stats["mean"]
                        self._is_stats_accum["std"] += is_stats["std"]
                        self._is_stats_accum["min"] += is_stats["min"]
                        self._is_stats_accum["max"] += is_stats["max"]
                        self._is_stats_accum["masked_fraction"] += is_stats[
                            "masked_fraction"
                        ]
                        self._is_stats_count += 1
                    else:
                        loss = opd_result
                elif self.args.use_distillm2:
                    current_alpha = self._get_current_alpha()
                    loss = self.distillm2_loss(
                        student_logits=shifted_student_logits,
                        teacher_logits=shifted_teacher_logits,
                        labels=shifted_labels,
                        alpha=current_alpha,
                        on_policy=inputs["on_policy"],
                    )
                else:
                    current_beta = self._get_current_beta()
                    current_alpha = self._get_current_alpha()
                    loss = self.generalized_jsd_loss(
                        student_logits=shifted_student_logits,
                        teacher_logits=shifted_teacher_logits,
                        labels=shifted_labels,
                        beta=current_beta,
                        temperature=self.args.temperature,
                        alpha=current_alpha,
                        use_adaptive_kld=self.args.use_adaptive_kld,
                        use_distillm2_like=self.args.use_distillm2_like,
                        use_reversed_distillm2_like=self.args.use_reversed_distillm2_like,
                        use_kl_interpolation=self.args.use_kl_interpolation,
                        on_policy=any(inputs["on_policy"]),
                        tokenizer=self.processing_class,
                    )

                # Hidden state matching loss (auxiliary)
                if self.args.use_hidden_loss:
                    hidden_loss = self.compute_hidden_loss(
                        student_hidden_states=outputs_student.hidden_states,
                        teacher_hidden_states=outputs_teacher.hidden_states,
                        hidden_proj=self.hidden_proj,
                        layer_ids=self.args.hidden_loss_layers,
                        shifted_labels=shifted_labels,
                    )
                    lmbda_weight = 1.0 - self._get_current_lmbda()
                    loss = (
                        loss + self.args.hidden_loss_gamma * lmbda_weight * hidden_loss
                    )

                    # Track for logging
                    ga = max(1, int(self.args.gradient_accumulation_steps))
                    self._hidden_loss_total += hidden_loss.item()
                    self._hidden_loss_step_equiv += 1.0 / ga

                    # Free hidden states
                    del outputs_student, outputs_teacher

        if self.use_uld_loss:
            student_input_ids = inputs["input_ids"]

            # Use the *teacher* labels created above, not the student's.
            teacher_labels_for_loss = (
                teacher_labels if "teacher_labels" in locals() else inputs["labels"]
            )
            teacher_input_ids_for_loss = (
                teacher_input_ids
                if "teacher_input_ids" in locals()
                else inputs["input_ids"]
            )

            # Create properly masked student labels (fixing batch size > 1 issue)
            student_labels = inputs["labels"].clone()
            # Value-mask pad tokens in both label sets, except when pad == eos
            # (see mask_pad_labels_by_value). Positional masking (the
            # collator's prompt-length boundary) already covers padding.
            mask_pad_labels_by_value(
                student_labels,
                getattr(self.processing_class, "pad_token_id", None),
                getattr(self.processing_class, "eos_token_id", None),
            )
            teacher_tokenizer = getattr(self, "teacher_tokenizer", None)
            mask_pad_labels_by_value(
                teacher_labels,
                getattr(teacher_tokenizer, "pad_token_id", None),
                getattr(teacher_tokenizer, "eos_token_id", None),
            )

            vllm_lp = (
                inputs.get("vllm_logprobs")
                if self.args.uld_hybrid_sampled_opd_weight > 0
                and self.args.opd_importance_sampling
                else None
            )

            teacher_logits_for_loss = (
                teacher_logits
                if "teacher_logits" in locals()
                else outputs_teacher.logits
            )

            uld_kwargs = dict(
                student_logits=outputs_student.logits,
                teacher_logits=teacher_logits_for_loss,
                student_labels=student_labels,
                teacher_labels=teacher_labels_for_loss,
                student_input_ids=student_input_ids,
                teacher_input_ids=teacher_input_ids_for_loss,
            )
            if isinstance(self.uld_loss_fn, CustomULDLoss):
                uld_kwargs["vllm_logprobs"] = vllm_lp
            loss = self.uld_loss_fn(**uld_kwargs)

            # If ULD hybrid mode produced per-step matched/unmatched components, accumulate them for logging.
            # Use gradient_accumulation_steps to mirror Trainer's windowing behavior.
            if hasattr(self.uld_loss_fn, "last_matched_loss") and hasattr(
                self.uld_loss_fn, "last_unmatched_loss"
            ):
                try:
                    ga = max(1, int(self.args.gradient_accumulation_steps))
                except Exception:
                    ga = 1
                step_eq = 1.0 / ga
                # read scalar values for logging
                matched_val = (
                    self.uld_loss_fn.last_matched_loss.item()
                    if self.uld_loss_fn.last_matched_loss is not None
                    else 0.0
                )
                unmatched_val = (
                    self.uld_loss_fn.last_unmatched_loss.item()
                    if self.uld_loss_fn.last_unmatched_loss is not None
                    else 0.0
                )

                sampled_opd_val = (
                    self.uld_loss_fn.last_sampled_opd_loss.item()
                    if self.uld_loss_fn.last_sampled_opd_loss is not None
                    else 0.0
                )

                self._matched_sum += matched_val * step_eq
                self._unmatched_sum += unmatched_val * step_eq
                self._sampled_opd_sum += sampled_opd_val * step_eq
                self._matched_step_eq += step_eq
                self._unmatched_step_eq += step_eq
                self._sampled_opd_step_eq += step_eq

        empty_cache()

        return (loss, outputs_student) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"

        now = time.time()
        if mode == "train" and "loss" in logs:
            logs["time"] = round(now - self._last_log_time, 2)
            # Schedule values ride the REGULAR log line. They used to be logged
            # from training_step with their own self.log() call per micro-batch,
            # which arrived here, all-reduced and reset the ULD/alignment/IS
            # accumulators, and left the regular line reporting about zero for
            # them whenever logging_steps was 1.
            if self.args.lmbda_schedule == "linear":
                logs["train/current_lmbda"] = self._get_current_lmbda()
            if self.args.beta_schedule == "linear":
                logs["train/current_beta"] = self._get_current_beta()
            if self.args.alpha_schedule == "linear":
                logs["train/current_alpha"] = self._get_current_alpha()
            logs["t_gen"] = round(self._timing_generation, 1)
            logs["t_train"] = round(
                self._timing_training_step_total - self._timing_generation, 1
            )
            logs["t_sync"] = round(self._timing_weight_sync, 1)
            logs["t_sync_ag"] = round(self._timing_weight_sync_allgather, 1)
            logs["t_sync_bc"] = round(self._timing_weight_sync_broadcast, 1)
            logs["n_sync_params"] = self._timing_weight_sync_params
            if self.args.overlap_generation:
                logs["t_gen_actual"] = round(self._timing_generation_actual, 1)
                logs["t_gen_wait"] = round(self._timing_generation_wait, 1)
                logs["t_train_pure"] = round(self._timing_training_pure, 1)
                self._timing_generation_actual = 0.0
                self._timing_generation_wait = 0.0
                self._timing_training_pure = 0.0
            # GENERATION EFFICIENCY, emitted only on a step that generated -- an off-policy arm, or
            # a lmbda<1 arm on an off-policy step, would otherwise post a wall of zeros that reads
            # as "generation collapsed" rather than "no generation was asked for".
            #
            # n_gen_batch is THE number the mb1 waste is visible in: prompts per request, averaged
            # over the requests in this window. Against --data-parallel-size 8 it is also the
            # occupancy -- 8 means one sequence per replica and no batching benefit whatsoever,
            # 32 means four. n_gen_calls is the rounds per optimizer step, which is mb x gas / mb =
            # gas, so it should equal gradient_accumulation_steps on a fully on-policy step and is
            # the other half of the same fact.
            if self._gen_calls:
                logs["n_gen_calls"] = self._gen_calls
                logs["n_gen_batch"] = round(self._gen_batch_total / self._gen_calls, 1)
                logs["n_gen_tokens"] = self._gen_tokens
                logs["n_gen_truncated"] = self._gen_truncated
                if self._gen_prompt_rows:
                    logs["gen_tokens_per_row"] = round(
                        self._gen_tokens / self._gen_prompt_rows, 1
                    )
                    # PERCENT, so it does not sit invisibly at the bottom of a plot whose other two
                    # series are in the hundreds and thousands. Emitted ONLY when a cap exists to be
                    # hit: with max_completion_length unset the counter is structurally 0, and
                    # publishing that as "0% truncated" would assert something this arm never
                    # measured.
                    if getattr(self.args, "max_completion_length", None):
                        logs["gen_truncated_pct"] = round(
                            100.0 * self._gen_truncated / self._gen_prompt_rows, 2
                        )
                # Rank-0 decode throughput. Not a cluster total -- see the note in __init__ -- but
                # it is the quantity that says whether a bigger batch actually bought anything:
                # tokens/s should rise roughly with n_gen_batch while t_gen stays flat.
                if self._timing_generation > 0:
                    logs["gen_tokens_per_s"] = round(
                        self._gen_tokens / self._timing_generation, 1
                    )
            self._gen_calls = 0
            self._gen_batch_total = 0
            self._gen_prompt_rows = 0
            self._gen_tokens = 0
            self._gen_truncated = 0

            self._timing_generation = 0.0
            self._timing_training_step_total = 0.0
            self._timing_weight_sync = 0.0
            self._timing_weight_sync_allgather = 0.0
            self._timing_weight_sync_broadcast = 0.0
            self._last_log_time = now

        if mode == "train" and self.args.use_hidden_loss:
            device = (
                self.accelerator.device
                if hasattr(self.accelerator, "device")
                else torch.device("cpu")
            )

            hidden_vec = torch.tensor(
                [self._hidden_loss_total, self._hidden_loss_step_equiv],
                dtype=torch.float64,
                device=device,
            )

            if (
                getattr(self.accelerator, "distributed_type", DistributedType.NO)
                != DistributedType.NO
                and dist.is_available()
                and dist.is_initialized()
            ):
                dist.all_reduce(hidden_vec, op=dist.ReduceOp.SUM)

            hidden_sum, hidden_eq = hidden_vec.tolist()
            if hidden_eq > 0:
                logs["hidden_loss"] = round(hidden_sum / hidden_eq, 4)

            # Reset accumulators
            self._hidden_loss_total = 0.0
            self._hidden_loss_step_equiv = 0.0

        if mode == "train" and getattr(self.args, "uld_per_turn_regions", False):
            # `uld/regions_per_row` is the observable that says the per-turn render is actually
            # doing something: 1.00 means one region per row, i.e. a per-turn arm silently
            # behaving like a single-region one. `uld/span_mismatch_rows` must be 0 -- a
            # non-zero value means some rows were supervised on a prefix of their turns, which
            # is a quiet partial version of the 46.6% token loss this flag exists to end.
            #
            # Gated on the CONFIG, never on the counters. `_uld_rows_seen > 0` is data-dependent,
            # so a rank whose micro-batch took a different path would skip the all_reduce below
            # and hang the ones that did not (confirmed directly: 2,081 s, zero logged steps).
            #
            # THE FIRST THREE COUNTERS LIVE ON THE LOSS OBJECT, NOT ON self. They are incremented at
            # :1348-1350, inside CustomULDLoss, and those are the only writes in this file. Reading
            # them off the trainer -- which this block did until 2026-09-16 -- makes `rows` 0 on
            # every rank of every run, so `uld/regions_per_row` is emitted by nothing and
            # `uld/span_mismatch_rows` reports a hardcoded 0. A direct measurement was launched with exactly
            # those two as its pass criteria and neither could ever fail. `_uld_relaxed_boundaries`
            # is the odd one out and stays on self: it is written at :3853-3854, in this class.
            loss_fn = getattr(self, "uld_loss_fn", None)
            device = (
                self.accelerator.device
                if hasattr(self.accelerator, "device")
                else torch.device("cpu")
            )
            # The last three are the byte-path's own work counters, and they are here for the same
            # reason the first three are: a number nobody logs is a number nobody can falsify. The
            # scaffold pair says how much head trimming the render fix is actually doing (a sudden
            # zero on a thinking arm means the trim stopped firing, which is the zero-gradient
            # defect returning); `_uld_region_chunks` says how many chunked forward passes the
            # region loss paid for, which is the only direct read on the chunking's cost. All three
            # are appended to the SAME tensor rather than reduced separately, so the collective
            # count stays rank-uniform under the config gate above.
            span_vec = torch.tensor(
                [
                    getattr(loss_fn, "_uld_span_mismatch_rows", 0),
                    getattr(loss_fn, "_uld_regions_seen", 0),
                    getattr(loss_fn, "_uld_rows_seen", 0),
                    getattr(self, "_uld_relaxed_boundaries", 0),
                    getattr(loss_fn, "_uld_scaffold_tokens_trimmed", 0),
                    getattr(loss_fn, "_uld_scaffold_regions_trimmed", 0),
                    getattr(loss_fn, "_uld_region_chunks", 0),
                ],
                dtype=torch.float64,
                device=device,
            )
            if (
                getattr(self.accelerator, "distributed_type", DistributedType.NO)
                != DistributedType.NO
                and dist.is_available()
                and dist.is_initialized()
            ):
                dist.all_reduce(span_vec, op=dist.ReduceOp.SUM)
            mismatch, regions, rows, relaxed, scaf_tok, scaf_reg, chunks = (
                span_vec.tolist()
            )
            logs["uld/span_mismatch_rows"] = int(mismatch)
            if rows > 0:
                logs["uld/regions_per_row"] = round(regions / rows, 3)
            # `uld/relaxed_boundaries` counts turns whose start came from the divergence point
            # rather than from the generation-prompt render's length, because the two disagreed
            # by trailing whitespace alone (_turn_boundary). It is not an error -- without the
            # policy those rows would RAISE and kill the run -- but it is a rate worth watching:
            # against granite it should sit near a third of the turns, and a sudden zero on a
            # granite arm means the render changed under us.
            logs["uld/relaxed_boundaries"] = int(relaxed)
            logs["uld/scaffold_tokens_trimmed"] = int(scaf_tok)
            logs["uld/scaffold_regions_trimmed"] = int(scaf_reg)
            # Emitted only on an arm that chunks, so a zero here always means "chunking is off" and
            # never "chunking ran and did nothing" -- two states worth keeping distinguishable.
            if getattr(self.args, "uld_region_chunk_groups", 0):
                logs["uld/region_chunks"] = int(chunks)
            # Reset each accumulator on the object that owns it. `loss_fn is None` is possible on a
            # non-ULD arm that still carries the config flag, and it is NOT allowed to change
            # whether the all_reduce above runs -- that gate is the config, and only the config.
            if loss_fn is not None:
                loss_fn._uld_span_mismatch_rows = 0
                loss_fn._uld_regions_seen = 0
                loss_fn._uld_rows_seen = 0
                loss_fn._uld_scaffold_tokens_trimmed = 0
                loss_fn._uld_scaffold_regions_trimmed = 0
                loss_fn._uld_region_chunks = 0
            self._uld_relaxed_boundaries = 0

        if (
            mode == "train"
            and getattr(self.args, "uld_hybrid_sampled_opd_weight", 0.0) > 0
        ):
            device = (
                self.accelerator.device
                if hasattr(self.accelerator, "device")
                else torch.device("cpu")
            )
            opd_vec = torch.tensor(
                [self._sampled_opd_sum, self._sampled_opd_step_eq],
                dtype=torch.float64,
                device=device,
            )
            if (
                getattr(self.accelerator, "distributed_type", DistributedType.NO)
                != DistributedType.NO
                and dist.is_available()
                and dist.is_initialized()
            ):
                dist.all_reduce(opd_vec, op=dist.ReduceOp.SUM)
            opd_sum, opd_eq = opd_vec.tolist()
            if opd_eq > 0:
                logs["sampled_opd_loss"] = round(opd_sum / opd_eq, 4)
            self._sampled_opd_sum = 0.0
            self._sampled_opd_step_eq = 0.0

        if (
            mode == "train"
            and getattr(self.args, "opd_importance_sampling", False)
            and self._is_stats_count > 0
        ):
            n = self._is_stats_count
            logs["train/is_weight_mean"] = self._is_stats_accum["mean"] / n
            logs["train/is_weight_std"] = self._is_stats_accum["std"] / n
            logs["train/is_weight_min"] = self._is_stats_accum["min"] / n
            logs["train/is_weight_max"] = self._is_stats_accum["max"] / n
            logs["train/is_masked_fraction"] = (
                self._is_stats_accum["masked_fraction"] / n
            )
            self._is_stats_accum = {k: 0.0 for k in self._is_stats_accum}
            self._is_stats_count = 0

        if (
            mode == "train"
            and getattr(self.args, "uld_hybrid_sampled_opd_weight", 0.0) > 0
            and hasattr(self, "uld_loss_fn")
            and self.uld_loss_fn is not None
            and hasattr(self.uld_loss_fn, "_opd_is_stats_count")
            and self.uld_loss_fn._opd_is_stats_count > 0
        ):
            n = self.uld_loss_fn._opd_is_stats_count
            acc = self.uld_loss_fn._opd_is_stats_accum
            logs["train/hybrid_opd_is_weight_mean"] = acc["mean"] / n
            logs["train/hybrid_opd_is_weight_std"] = acc["std"] / n
            logs["train/hybrid_opd_is_weight_min"] = acc["min"] / n
            logs["train/hybrid_opd_is_weight_max"] = acc["max"] / n
            logs["train/hybrid_opd_is_masked_fraction"] = acc["masked_fraction"] / n
            self.uld_loss_fn._opd_is_stats_accum = {k: 0.0 for k in acc}
            self.uld_loss_fn._opd_is_stats_count = 0

        if (
            mode == "train"
            and getattr(self.args, "uld_hybrid_sampled_opd_weight", 0.0) > 0
            and hasattr(self, "uld_loss_fn")
            and self.uld_loss_fn is not None
        ):
            for suffix, attr_accum, attr_count in [
                ("", "_opd_reward_stats_accum", "_opd_reward_stats_count"),
                (
                    "_1to1",
                    "_opd_reward_1to1_stats_accum",
                    "_opd_reward_1to1_stats_count",
                ),
                (
                    "_multi",
                    "_opd_reward_multi_stats_accum",
                    "_opd_reward_multi_stats_count",
                ),
                (
                    "_1toN",
                    "_opd_reward_1toN_stats_accum",
                    "_opd_reward_1toN_stats_count",
                ),
                (
                    "_Nto1",
                    "_opd_reward_Nto1_stats_accum",
                    "_opd_reward_Nto1_stats_count",
                ),
                (
                    "_NtoN",
                    "_opd_reward_NtoN_stats_accum",
                    "_opd_reward_NtoN_stats_count",
                ),
            ]:
                count = getattr(self.uld_loss_fn, attr_count, 0)
                if count > 0:
                    acc = getattr(self.uld_loss_fn, attr_accum)
                    for k in acc:
                        logs[f"train/hybrid_opd_reward{suffix}_{k}"] = acc[k] / count
                    setattr(self.uld_loss_fn, attr_accum, {k: 0.0 for k in acc})
                    setattr(self.uld_loss_fn, attr_count, 0)

            if self.uld_loss_fn._opd_batch_norm_stats_count > 0:
                n = self.uld_loss_fn._opd_batch_norm_stats_count
                acc_norm = self.uld_loss_fn._opd_batch_norm_stats_accum
                logs["train/hybrid_opd_batch_reward_baseline_mean"] = (
                    acc_norm["mean"] / n
                )
                logs["train/hybrid_opd_batch_reward_baseline_std"] = acc_norm["std"] / n
                self.uld_loss_fn._opd_batch_norm_stats_accum = {
                    k: 0.0 for k in acc_norm
                }
                self.uld_loss_fn._opd_batch_norm_stats_count = 0

            acc = self.uld_loss_fn._opd_group_type_accum
            total = acc["total"]
            if total > 0:
                logs["train/hybrid_opd_group_total"] = total
                for gtype in ("1to1", "1toN", "Nto1", "NtoN"):
                    logs[f"train/hybrid_opd_group_ratio_{gtype}"] = acc[gtype] / total
                self.uld_loss_fn._opd_group_type_accum = {k: 0 for k in acc}

        if (
            mode == "train"
            and hasattr(self, "uld_loss_fn")
            and self.uld_loss_fn is not None
            and self.uld_loss_fn._alignment_total > 0
        ):
            total = self.uld_loss_fn._alignment_total
            trunc = self.uld_loss_fn._alignment_truncated
            logs["train/alignment_total"] = total
            logs["train/alignment_truncated"] = trunc
            logs["train/alignment_truncation_rate"] = trunc / total
            self.uld_loss_fn._alignment_total = 0
            self.uld_loss_fn._alignment_truncated = 0

        # Already summed over ranks, step by step, by the guard; no collective here.
        if mode == "train" and getattr(self, "_uld_alignment_guard", None) is not None:
            logs.update(self._uld_alignment_guard.pop_log())

        if (
            mode == "train"
            and hasattr(self, "uld_loss_fn")
            and self.uld_loss_fn is not None
            and getattr(self.args, "visualize_sampled_opd_reward", False)
            and self.uld_loss_fn._reward_html_buffer
        ):
            self.uld_loss_fn._flush_reward_html(self.state.global_step)

        super().log(logs, start_time)
