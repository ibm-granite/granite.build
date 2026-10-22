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

"""Ray Data tokenization shared by the HF multi-GPU SFT drivers.

``driver_multi_hf_fsdp.py`` and ``driver_multi_hf_ds.py`` both build their
``data_backend="ray_data"`` datasets through :func:`build_tokenized_datasets`,
so the read/fan-out/tokenize/materialize pipeline lives in one place.

Two properties matter for time-to-first-GPU-step, because all of this runs in
the trial process *before* the ``TorchTrainer`` (and its GPU workers) exists:

* **Tokenize exactly once.** The trainer needs the tokenized row count to size
  ``max_steps`` (the sliding window in ``tokenize_batch_arrays`` can expand one
  record into several rows, so the raw count is not it). ``count()`` on a lazy
  pipeline executes the whole tokenization and discards the result; the
  pipeline then re-ran during training, once per epoch. ``materialize()`` keeps
  the blocks in the object store, so the count is metadata and every epoch
  reads the same blocks.
* **Cheap blocks.** :class:`TokenizeBatch` returns fixed-shape ``int32``
  arrays, which Ray stores as tensor columns. The old per-token Python lists
  had to be converted to Arrow element by element, which cost more than the
  tokenizer itself. Only ``input_ids`` is stored, plus two ``int32`` scalars
  per row (``pad_len``, ``target_len``): ``attention_mask`` and ``labels`` are
  pure functions of those, so :func:`expand_rows` rebuilds them on the GPU
  worker instead of carrying 3x the bytes through the object store.

Records are shuffled once, seeded, before tokenization — the materialized
blocks are otherwise read in file order every epoch, and the per-rank local
shuffle buffer alone cannot decorrelate a file sorted by source or label (the
Arrow backend gets a global shuffle from ``DistributedSampler``).
"""

import logging
import os
import random
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import ray
import torch
from torch.utils.data import IterableDataset

from autotune.cluster import compute_ray_data_sizing, ray_data_block_target
from autotune.utils import get_tokenizer, tokenize_batch_arrays

logger = logging.getLogger(__name__)

# Rows per map_batches call. Large enough to amortize the batched Rust
# tokenizer call, small enough that a 1024-token int32 batch stays ~12 MB.
TOKENIZE_BATCH_SIZE = 1024


def read_dataset(file_path: str) -> "ray.data.Dataset":
    """Read a dataset file into a Ray Dataset, dispatching on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".parquet":
        return ray.data.read_parquet(file_path)
    if ext == ".csv":
        return ray.data.read_csv(file_path)
    if ext == ".jsonl":
        return ray.data.read_json(file_path)
    if ext == ".json":
        # Ray Data's read_json expects line-delimited JSON; fall back to
        # pandas for a single JSON array and wrap as a Ray Dataset.
        df = pd.read_json(file_path, lines=False)
        return ray.data.from_pandas(df)
    raise ValueError(f"Unsupported dataset extension: {ext} ({file_path})")


def _to_python(value: Any) -> Any:
    """Recursively turn Ray's numpy views of nested columns back into lists/dicts."""
    if isinstance(value, np.ndarray):
        return [_to_python(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_python(v) for k, v in value.items()}
    return value


def batch_to_dataframe(batch: Dict[str, Any]) -> pd.DataFrame:
    """``pd.DataFrame`` of a Ray numpy batch, with nested columns as Python lists.

    ``batch_format="numpy"`` hands a list-valued column (chat ``messages``,
    ``documents``, ``tools``) over as an object array whose cells are
    ``np.ndarray`` of dicts, not ``list``. ``_apply_chat_template_to_df``
    detects chat rows with ``isinstance(first, list)``, so without this every
    chat-format row skipped the template and ``str(ndarray)`` was tokenized
    as the prompt, silently. Flat columns pass through untouched.
    """
    columns = {}
    for name, col in batch.items():
        if isinstance(col, np.ndarray) and col.dtype == object and len(col) and isinstance(col[0], (np.ndarray, dict)):
            columns[name] = [_to_python(v) for v in col]
        else:
            columns[name] = col
    return pd.DataFrame(columns)


def compact_rows(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Reduce ``tokenize_batch_arrays`` output to ``input_ids`` + two per-row lengths.

    Rows are left-padded ``[pad | prompt chunk | target + EOS]``, so the mask is
    ``pos >= pad_len`` and the labels are ``input_ids`` from ``max_length -
    target_len`` on. Both are read off the arrays rather than recomputed, so
    this cannot drift from the windowing in ``tokenize_batch_arrays``. Not
    inferred from ``input_ids == pad_token_id``: ``get_tokenizer`` defaults
    the pad token to EOS, which also ends every target.
    """
    max_length = arrays["input_ids"].shape[1]
    return {
        "input_ids": arrays["input_ids"],
        "pad_len": (max_length - arrays["attention_mask"].sum(axis=1)).astype(np.int32),
        "target_len": (arrays["labels"] != -100).sum(axis=1).astype(np.int32),
    }


def expand_rows(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Inverse of :func:`compact_rows`, on a torch batch: rebuild mask and labels."""
    input_ids = batch["input_ids"].long()
    pos = torch.arange(input_ids.shape[1], device=input_ids.device)
    attention_mask = (pos >= batch["pad_len"].long().unsqueeze(1)).long()
    target_start = input_ids.shape[1] - batch["target_len"].long().unsqueeze(1)
    labels = torch.where(pos >= target_start, input_ids, torch.full_like(input_ids, -100))
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class RayShardIterableDataset(IterableDataset):
    """Wraps a Ray Data shard so HF Trainer can consume it as an IterableDataset.

    Ray Train splits the parent Dataset across workers, so each worker only
    iterates over its own rank-specific slice — no DistributedSampler needed.
    Batches arrive in the :func:`compact_rows` layout and are expanded to
    ``input_ids`` / ``attention_mask`` / ``labels`` here.

    When ``total_samples`` is set, iteration produces exactly that many
    samples per rank, cycling through the shard as needed. This is critical
    under distributed training: unequal iteration lengths across ranks cause
    NCCL watchdog hangs because fast ranks reach end-of-epoch collectives
    while slower ranks are still producing batches. Ray Train splits with
    ``equal=True`` today, but the cap makes equal lengths hold by construction
    rather than by a Ray default.

    ``group_by_length`` is the iterable counterpart of HF's
    ``LengthGroupedSampler`` (see :mod:`autotune.trainers._collate`): samples
    are buffered ``batch_size * GROUP_BUCKET_BATCHES`` at a time, sorted by
    real length, cut into ``batch_size`` runs and yielded in shuffled run
    order. The DataLoader batches consecutive samples, so each batch is
    length-homogeneous and the dynamic-padding collator can crop it short.
    Each rank owns its shard, so unlike the map-style sampler the ranks need
    not agree on an order.
    """

    GROUP_BUCKET_BATCHES = 64

    def __init__(
        self,
        shard,
        batch_size: int,
        total_samples: Optional[int] = None,
        shuffle_buffer: Optional[int] = None,
        group_by_length: bool = False,
        seed: int = 42,
    ):
        self.shard = shard
        self.batch_size = batch_size
        self.total_samples = total_samples
        self.shuffle_buffer = shuffle_buffer
        self.group_by_length = group_by_length
        self._rng = random.Random(seed)

    def _iter_rows(self):
        for batch in self.shard.iter_torch_batches(
            batch_size=self.batch_size,
            local_shuffle_buffer_size=self.shuffle_buffer,
        ):
            batch = expand_rows(batch)
            for i in range(batch["input_ids"].shape[0]):
                yield {k: v[i] for k, v in batch.items()}

    def _grouped(self, bucket):
        bucket.sort(key=lambda s: int(s["attention_mask"].sum()))
        runs = [bucket[i : i + self.batch_size] for i in range(0, len(bucket), self.batch_size)]
        self._rng.shuffle(runs)
        for run in runs:
            yield from run

    def _iter_once(self):
        if not self.group_by_length:
            yield from self._iter_rows()
            return
        bucket_size = max(1, self.batch_size) * self.GROUP_BUCKET_BATCHES
        bucket = []
        for sample in self._iter_rows():
            bucket.append(sample)
            if len(bucket) >= bucket_size:
                yield from self._grouped(bucket)
                bucket = []
        if bucket:
            yield from self._grouped(bucket)

    def __iter__(self):
        if self.total_samples is None:
            yield from self._iter_once()
            return

        yielded = 0
        while yielded < self.total_samples:
            for sample in self._iter_once():
                yield sample
                yielded += 1
                if yielded >= self.total_samples:
                    return


class TokenizeBatch:
    """``map_batches`` callable: chat template (if needed) + windowed tokenization.

    A class so Ray Data runs it on an actor pool: each actor loads the
    tokenizer once in ``__init__`` instead of once per task.
    ``apply_chat_template`` is the driver's own ``_apply_chat_template_to_df``,
    passed in so the driver keeps owning chat-template handling.
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_kwargs: Dict[str, Any],
        input_col: str,
        output_col: str,
        max_length: int,
        thinking: bool,
        apply_chat_template: Callable[..., Any],
    ):
        # Each actor reserves one CPU; a fast tokenizer's batched call would
        # otherwise spawn a Rayon thread per core in every actor, on a node
        # already running one actor per core.
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer, _ = get_tokenizer(model_name_or_path, **tokenizer_kwargs)
        self.input_col = input_col
        self.output_col = output_col
        self.max_length = max_length
        self.thinking = thinking
        self.apply_chat_template = apply_chat_template

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # Ray Data passes numpy-dict batches (batch_format="numpy"); convert
        # to a DataFrame locally so the pandas-based chat-template logic can
        # stay as-is. We avoid batch_format="pandas" because Ray Data's pandas
        # BlockAccessor references pandas.core.common.SettingWithCopyWarning,
        # which was removed in pandas 3.0.
        df = self.apply_chat_template(batch_to_dataframe(batch), self.tokenizer, self.input_col, self.thinking)
        return compact_rows(
            tokenize_batch_arrays(
                batch=df,
                tokenizer=self.tokenizer,
                input_col=self.input_col,
                output_col=self.output_col,
                max_length=self.max_length,
            )
        )


def build_tokenized_datasets(
    train_file: str,
    eval_file: str,
    *,
    percentage: float,
    seed: int,
    model_name_or_path: str,
    tokenizer_kwargs: Dict[str, Any],
    input_col: str,
    output_col: str,
    max_length: int,
    thinking: bool,
    apply_chat_template: Callable[..., Any],
    concurrency_override: Optional[int] = None,
    num_cpus_override: Optional[float] = None,
) -> Tuple["ray.data.Dataset", "ray.data.Dataset", int]:
    """Read, shuffle, tokenize and materialize the train/eval splits.

    Returns ``(train_ds, eval_ds, train_ds_size)``; both datasets are
    materialized in the :func:`compact_rows` layout, so ``train_ds_size``
    (tokenized rows, post-windowing) is free.
    """
    # Materialize the raw records: the count is then metadata, where on a lazy
    # jsonl/csv read it executed the whole read only for the pipeline to read
    # the file again. Raw text is small next to the tokenized blocks.
    train_ds = read_dataset(train_file).materialize()
    eval_ds = read_dataset(eval_file).materialize()

    train_rows = train_ds.count()
    eval_rows = eval_ds.count()
    if percentage < 1.0:
        # The leading rows, like the Arrow backend's df.head(n), so both
        # backends train an HPO trial on the same subset.
        train_rows = max(1, int(train_rows * percentage))
        eval_rows = max(1, int(eval_rows * percentage))
        train_ds = train_ds.limit(train_rows)
        eval_ds = eval_ds.limit(eval_rows)
    # Seeded, and on the raw text where an all-to-all shuffle is cheap; eval
    # order does not matter.
    train_ds = train_ds.random_shuffle(seed=seed)

    # Fan tokenization out across the whole cluster. Two levers, both required:
    #   1. Repartition into >= `concurrency` blocks. A single source file often
    #      reads as 1 block, and a block is the unit of work handed to one
    #      actor — so without this, tokenization runs on a single CPU regardless
    #      of cluster size. repartition(n, shuffle=False) is a cheap
    #      split/combine, not a full shuffle.
    #   2. concurrency / num_cpus size the actor pool (see _tokenize below).
    # concurrency defaults to every CPU free right now (see
    # compute_ray_data_sizing); overridable via config/CLI. num_cpus is the
    # per-actor CPU reservation (fractional allows oversubscribe).
    # batch_format="numpy" (default) avoids pandas-block compat issues on
    # pandas 3.x.
    concurrency, num_cpus = compute_ray_data_sizing(concurrency_override, num_cpus_override)
    train_blocks = ray_data_block_target(concurrency, train_rows)
    eval_blocks = ray_data_block_target(concurrency, eval_rows)
    # Skip repartition for trivially small datasets (single block already).
    if train_rows > 1:
        train_ds = train_ds.repartition(train_blocks, shuffle=False)
    if eval_rows > 1:
        eval_ds = eval_ds.repartition(eval_blocks, shuffle=False)
    # Materialize the prepared text before tokenizing, so the actor pool has no
    # upstream operator left to run. The pool is sized to every free CPU, and
    # Ray Data reserves CPUs for it up front: with the shuffle/repartition
    # tasks still upstream they got a CPU budget of 0 and the pipeline
    # deadlocked (reproduced on a 4-CPU local cluster, 2 runs out of 3).
    train_ds = train_ds.materialize()
    eval_ds = eval_ds.materialize()
    logger.info(
        f"[AutoTune] Ray Data tokenize: concurrency={concurrency}, num_cpus={num_cpus}, "
        f"repartition target train={train_blocks}/eval={eval_blocks} blocks "
        f"(batch_size={TOKENIZE_BATCH_SIZE})"
    )

    tokenize_kwargs = {
        "model_name_or_path": model_name_or_path,
        "tokenizer_kwargs": tokenizer_kwargs,
        "input_col": input_col,
        "output_col": output_col,
        "max_length": max_length,
        "thinking": thinking,
        "apply_chat_template": apply_chat_template,
    }

    def _tokenize(ds: "ray.data.Dataset", blocks: int) -> "ray.data.Dataset":
        # Fixed-size pool, capped at the block count (an actor with no block
        # would only pay the startup). A fixed pool beat an autoscaling (1..n)
        # one by ~30% here: autoscaling ramps up only after the first actors
        # report backpressure, and the whole job is a few minutes long.
        return ds.map_batches(
            TokenizeBatch,
            fn_constructor_kwargs=tokenize_kwargs,
            batch_size=TOKENIZE_BATCH_SIZE,
            compute=ray.data.ActorPoolStrategy(size=max(1, min(concurrency, blocks))),
            num_cpus=num_cpus,
        ).materialize()

    train_ds = _tokenize(train_ds, train_blocks)
    eval_ds = _tokenize(eval_ds, eval_blocks)

    train_ds_size = train_ds.count()
    logger.info(
        f"[AutoTune] Tokenized train={train_ds_size} rows ({train_rows} records), "
        f"eval={eval_ds.count()} rows ({eval_rows} records); "
        f"materialized {(train_ds.size_bytes() + eval_ds.size_bytes()) / 2**30:.1f} GiB"
    )
    return train_ds, eval_ds, train_ds_size
