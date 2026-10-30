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

"""Arrow backend shared by the two multi-GPU HF drivers.

The driver tokenizes once on the head node and writes an Arrow IPC file; every
worker memory-maps the same file (nothing goes through the Ray object store)
and HF Trainer's distributed sampler shards the rows. Both
``driver_multi_hf_fsdp.py`` and ``driver_multi_hf_ds.py`` use this, so change
the format here, not per driver.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import torch

from autotune.tools.chat_utils import apply_chat_template_to_df
from autotune.trainers._collate import row_lengths
from autotune.utils import tokenize_batch_arrays

logger = logging.getLogger(__name__)


class TokenizedArrowDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    """Map-style Dataset backed by a memory-mapped Arrow IPC file.

    Each worker mmaps the same file (no serialization through the Ray
    object store) and indexes into it. HF Trainer's default distributed
    sampler shards rows across ranks.
    """

    def __init__(self, arrow_path: str):
        import pyarrow as pa

        self.table = pa.ipc.open_file(pa.memory_map(arrow_path, "r")).read_all()
        self._len = len(self.table)
        # Real (unpadded) token count per row, written by load_tokenize_and_save;
        # the length-grouped sampler reads it once instead of scanning every row.
        self.lengths: Optional[np.ndarray] = (
            self.table.column("length").to_numpy() if "length" in self.table.column_names else None
        )

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.table.slice(idx, 1)
        return {
            "input_ids": torch.tensor(row.column("input_ids")[0].as_py(), dtype=torch.long),
            "attention_mask": torch.tensor(row.column("attention_mask")[0].as_py(), dtype=torch.long),
            "labels": torch.tensor(row.column("labels")[0].as_py(), dtype=torch.long),
        }


def _read_frame(file_path: str) -> Any:
    import pandas as pd

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(file_path)
    if ext == ".csv":
        return pd.read_csv(file_path)
    if ext == ".json":
        return pd.read_json(file_path, lines=False)
    # .jsonl (and anything else line-delimited)
    return pd.read_json(file_path, lines=ext == ".jsonl")


def load_tokenize_and_save(
    file_path: str,
    tokenizer: Any,
    input_col: str,
    output_col: str,
    max_length: int,
    output_arrow_path: str,
    percentage: float = 1.0,
    chunk_size: int = 50000,
    thinking: bool = False,
) -> int:
    """Tokenize a dataset file on the driver and write it as an Arrow IPC file.

    Tokenizes in chunks to keep peak memory bounded. Each row's real length is
    taken from the chunk's attention-mask array in one vectorized sum, before
    the columns are converted to the lists Arrow stores.

    Args:
        file_path: ``.parquet``, ``.csv``, ``.json`` or line-delimited JSON.
        tokenizer: The model's tokenizer.
        input_col: Prompt column (or the chat column, rendered first).
        output_col: Target column.
        max_length: Window length; long records expand to several rows.
        output_arrow_path: Where to write the Arrow file.
        percentage: Leading fraction of the rows to keep.
        chunk_size: Rows tokenized per chunk.
        thinking: Passed to the chat-template rendering.

    Returns:
        The number of tokenized rows written.
    """
    import pyarrow as pa
    from tqdm import tqdm

    dataset_name = os.path.basename(file_path)
    df = _read_frame(file_path)
    logger.info(f"[AutoTune] Loaded {len(df)} rows from {file_path}")

    if percentage < 1.0:
        n = max(1, int(len(df) * percentage))
        df = df.head(n)
        logger.info(f"[AutoTune] Subsampled to {len(df)} rows ({percentage * 100:.0f}%)")

    df = apply_chat_template_to_df(df, tokenizer, input_col, thinking)

    logger.info("[AutoTune] First instance (after formatting):")
    logger.info(f"  {input_col}: {df[input_col].iloc[0]}")
    logger.info(f"  {output_col}: {df[output_col].iloc[0]}")

    all_input_ids: list[list[int]] = []
    all_attention_mask: list[list[int]] = []
    all_labels: list[list[int]] = []
    all_lengths: list[np.ndarray] = []
    total_rows = len(df)

    logger.info(f"[AutoTune] Tokenizing dataset with max_length={max_length}...")
    with tqdm(total=total_rows, desc=f"Tokenizing {dataset_name}", unit="rows") as pbar:
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            arrays = tokenize_batch_arrays(
                batch=df.iloc[start:end],
                tokenizer=tokenizer,
                input_col=input_col,
                output_col=output_col,
                max_length=max_length,
            )
            all_input_ids.extend(arrays["input_ids"].tolist())
            all_attention_mask.extend(arrays["attention_mask"].tolist())
            all_labels.extend(arrays["labels"].tolist())
            all_lengths.append(row_lengths(arrays["attention_mask"]))
            pbar.update(end - start)

    lengths = np.concatenate(all_lengths) if all_lengths else np.zeros(0, dtype=np.int64)
    table = pa.table(
        {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
            "length": pa.array(lengths, type=pa.int32()),
        }
    )

    os.makedirs(os.path.dirname(output_arrow_path), exist_ok=True)
    writer = pa.ipc.new_file(output_arrow_path, table.schema)
    batch_rows = 100000
    with tqdm(total=len(table), desc=f"Writing {dataset_name}.arrow", unit="rows") as pbar:
        for i in range(0, len(table), batch_rows):
            chunk = min(batch_rows, len(table) - i)
            writer.write(table.slice(i, chunk))
            pbar.update(chunk)
    writer.close()

    num_samples = len(table)
    size_mb = os.path.getsize(output_arrow_path) / (1024 * 1024)
    logger.info(f"[AutoTune] Saved {num_samples} samples to {output_arrow_path} ({size_mb:.1f} MB)")
    return num_samples
