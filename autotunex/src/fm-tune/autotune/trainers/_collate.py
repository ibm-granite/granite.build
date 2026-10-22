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

"""Batch collation for the HF SFT drivers: no compute spent on pad tokens.

``tokenize_batch_arrays`` left-pads every row to ``max_length``
(``[pad | prompt chunk | target + EOS]``), and the rows are stored that way in
every backend (HF ``datasets``, the Arrow mmap file, Ray Data tensor blocks).
Stacking them as-is — ``default_data_collator`` — sent every micro-batch
through the model at ``B x max_length``: with ``max_length: 1024`` and rows of
a few hundred tokens, most of the FLOPs in every projection, MLP and the LM
head went to padding. ``flash_attention_2`` already skips pads *inside
attention* (``_upad_input``), which made the waste easy to miss; the linear
layers never did.

The fix lives at collate time, so the tokenizer, the stored rows and the row
count are untouched — ``resolve_epoch_steps``, ``max_steps`` and the Ray Data
sample cap stay exactly as they were. Two modes, picked by
:func:`resolve_padding_mode`:

* ``"dynamic"`` — :class:`DynamicPadCollator`. Rows are right-aligned, so the
  batch's leading all-pad columns are dropped with one slice, keeping
  ``round_up(longest row, 8)`` columns. Numerically the same loss (pads are
  masked and ``-100`` either way; RoPE is shift-invariant). Works for every
  attention implementation and model. Paired with length grouping
  (:class:`LengthGroupedTrainer`, or ``RayShardIterableDataset(group_by_length=True)``)
  so the longest row in a batch is close to the rest.
* ``"padding_free"`` — :class:`FlatteningCollator`. Each micro-batch becomes one
  ``[1, sum(L_i)]`` varlen sequence with per-sequence ``position_ids`` and
  flash-attention ``cu_seq_lens``: zero pad tokens. Still one row per sample,
  so a step sees the same samples as before. Requires flash attention, and is
  gated off for models whose forward is not per-sequence under flattening.

``label_names`` must be ``["labels"]`` in either mode. The drivers used to pass
``["input_ids", "attention_mask"]``, which only worked because both keys were
always present: the flattened batch has no ``attention_mask``, so HF's
``prediction_step`` would see ``has_labels=False``, return no loss, and
``eval_loss`` — the metric ``load_best_model_at_end`` selects on — would vanish.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from transformers import DataCollatorWithFlattening, Trainer
from transformers.trainer_pt_utils import LengthGroupedSampler

logger = logging.getLogger(__name__)

PADDING_DYNAMIC = "dynamic"
PADDING_FREE = "padding_free"

# Columns a row carries; anything else (e.g. a stored ``length``) is dropped.
_ROW_KEYS = ("input_ids", "attention_mask", "labels")

# Model families whose forward is not per-sequence under flattening: Mamba/SSM
# state (and RWKV recurrence) would run straight across sequence boundaries.
_RECURRENT_KEYWORDS = ("hybrid", "mamba", "ssm", "rwkv", "jamba", "zamba", "bamba", "falcon_h1", "nemotron_h")


def row_lengths(attention_mask: Any) -> np.ndarray:
    """Real-token count per row of a ``[N, max_length]`` attention mask."""
    return np.asarray(attention_mask).sum(axis=1).astype(np.int64)


def log_padding_efficiency(lengths: Sequence[int], max_length: int, split: str = "train") -> None:
    """Log the share of a fully padded ``N x max_length`` grid that is real tokens."""
    lengths = np.asarray(lengths)
    if not len(lengths) or not max_length:
        return
    logger.info(
        f"[AutoTune] {split}: {len(lengths)} rows, mean length {lengths.mean():.0f} / max_length {max_length} "
        f"({100.0 * lengths.sum() / (len(lengths) * max_length):.1f}% real tokens when padded to max_length)"
    )


def _is_recurrent(model_config: Any) -> bool:
    if model_config is None:
        return False
    names = [str(getattr(model_config, "model_type", "") or "")]
    names += [str(a) for a in (getattr(model_config, "architectures", None) or [])]
    return any(kw in name.lower() for name in names for kw in _RECURRENT_KEYWORDS)


def resolve_padding_mode(
    training_config: Mapping[str, Any],
    *,
    attn_implementation: Optional[str],
    device_kind: str,
    model_config: Any = None,
    peft_type: Optional[str] = None,
    is_sr: bool = False,
) -> str:
    """Pick ``"padding_free"`` or ``"dynamic"`` from ``training_config["padding_free"]``.

    ``auto`` (the default) picks padding-free only when every precondition
    holds; ``true`` raises on the first one that does not, rather than quietly
    training in a different mode than requested; ``false`` always picks dynamic.
    """
    requested = training_config.get("padding_free", "auto")
    if isinstance(requested, str):
        requested = requested.strip().lower()
    if requested in (False, "false", "no", "off"):
        return PADDING_DYNAMIC
    if requested not in ("auto", True, "true", "yes", "on"):
        raise ValueError(f"padding_free must be auto|true|false, got {requested!r}")

    blockers = []
    if not str(attn_implementation or "").startswith("flash_attention"):
        blockers.append(
            f"attn_implementation={attn_implementation!r} (needs flash_attention_*; eager/sdpa would let "
            "flattened sequences attend to each other)"
        )
    if device_kind != "cuda":
        blockers.append(f"device={device_kind!r} (flash attention is CUDA-only)")
    if is_sr:
        blockers.append("tuning_algo=sr (the dual-stream forward is not verified against varlen batches)")
    if peft_type == "ALORA":
        blockers.append("peft_type=ALORA (aLoRA locates its invocation offset per row)")
    if _is_recurrent(model_config):
        blockers.append(
            f"model_type={getattr(model_config, 'model_type', None)!r} (recurrent/SSM state would cross "
            "sequence boundaries)"
        )

    if not blockers:
        return PADDING_FREE
    if requested == "auto":
        logger.info(f"[AutoTune] padding_free=auto -> dynamic padding: {blockers[0]}")
        return PADDING_DYNAMIC
    raise ValueError(f"padding_free=true is not supported here: {'; '.join(blockers)}")


def _as_tensor(value: Any) -> torch.Tensor:
    """A row from any backend (list, numpy, tensor) as a ``long`` tensor."""
    if isinstance(value, torch.Tensor):
        return value.long()
    return torch.as_tensor(np.asarray(value), dtype=torch.long)


def _stack(features: List[Mapping[str, Any]], key: str) -> torch.Tensor:
    return torch.stack([_as_tensor(f[key]) for f in features])


class DynamicPadCollator:
    """Stack left-padded fixed-width rows, cropped to the batch's longest row.

    Every row is right-aligned (``[pad | prompt | target]``), so the columns
    left of the longest row are padding in every row and can be sliced off
    without re-padding anything — no ``pad_token_id`` needed.
    """

    def __init__(self, pad_to_multiple_of: int = 8):
        self.pad_to_multiple_of = max(1, int(pad_to_multiple_of))

    def __call__(self, features: List[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {k: _stack(features, k) for k in _ROW_KEYS}
        width = batch["input_ids"].shape[1]
        longest = int(batch["attention_mask"].sum(dim=1).max())
        m = self.pad_to_multiple_of
        keep = min(width, max(m, -(-longest // m) * m))
        return {k: v[:, width - keep :] for k, v in batch.items()}


class FlatteningCollator:
    """Strip each row's padding, then flatten the batch into one varlen sequence.

    Delegates to HF's :class:`~transformers.DataCollatorWithFlattening`, which
    resets ``position_ids`` per sequence, emits flash-attention
    ``cu_seq_lens_*`` / ``max_length_*`` and no ``attention_mask``, and writes
    ``-100`` on each sequence's first label. That first token is always a
    prompt token (already ``-100``), and it is what masks the causal-LM shift
    across a sequence boundary.
    """

    def __init__(self):
        self._flatten = DataCollatorWithFlattening(
            return_position_ids=True,
            return_flash_attn_kwargs=True,
            separator_id=-100,
        )

    def __call__(self, features: List[Mapping[str, Any]]) -> Dict[str, Any]:
        rows = []
        for f in features:
            keep = _as_tensor(f["attention_mask"]).bool()
            rows.append({k: _as_tensor(f[k])[keep].tolist() for k in ("input_ids", "labels")})
        return self._flatten(rows)


def build_collator(mode: str):
    """The collator for a :func:`resolve_padding_mode` result."""
    if mode == PADDING_FREE:
        return FlatteningCollator()
    if mode == PADDING_DYNAMIC:
        return DynamicPadCollator()
    raise ValueError(f"Unknown padding mode: {mode!r}")


def group_by_length_enabled(training_config: Mapping[str, Any], mode: str) -> bool:
    """Length grouping only pays in dynamic mode: padding-free has no padding to save."""
    return mode == PADDING_DYNAMIC and bool(training_config.get("group_by_length", True))


class LengthGroupedTrainer(Trainer):
    """``Trainer`` whose train sampler groups rows of similar length.

    HF's own ``group_by_length`` finds lengths only on a ``datasets.Dataset``
    with a length column, and otherwise iterates the whole dataset measuring
    ``len(input_ids)`` — which is ``max_length`` for every stored row here.
    This takes the real lengths up front instead.

    The sampler gets an explicit generator seeded from ``args.seed``. HF's path
    draws from the global RNG, but under data parallelism every rank iterates
    the *same* sampler and Accelerate's ``BatchSamplerShard`` keeps every
    ``world_size``-th batch, so the ranks must agree on the permutation or the
    shards overlap. A seeded generator advances identically on every rank.
    """

    def __init__(self, *args, train_lengths: Optional[Sequence[int]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_lengths = None if train_lengths is None else [int(x) for x in train_lengths]
        self._length_generator: Optional[torch.Generator] = None

    def _get_train_sampler(self, train_dataset=None):
        if self._train_lengths is None:
            return super()._get_train_sampler(train_dataset)
        if self._length_generator is None:
            self._length_generator = torch.Generator().manual_seed(int(self.args.seed))
        return LengthGroupedSampler(
            self.args.train_batch_size * self.args.gradient_accumulation_steps,
            lengths=self._train_lengths,
            generator=self._length_generator,
        )
