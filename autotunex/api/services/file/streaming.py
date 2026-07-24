# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Byte-level streaming I/O for datasets. Moved verbatim from file_service.py."""

from fastapi import UploadFile, HTTPException
import asyncio
import json
from typing import List, Optional, Dict
import pyarrow as pa
import pyarrow.parquet as pq
import os
import random
import logging

import paths

# Module logger. Root logging is configured once at app startup
# (do not call basicConfig/setLevel here — see CLAUDE.md logging conventions).
logger = logging.getLogger(__name__)

# Bounded buffer used when copying uploads to disk so memory stays flat
# regardless of file size (a 1GB upload must not become 1GB resident). 8MB keeps
# the number of read/write thread hops low without holding much memory.
STREAM_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB

AUTOTUNE_DATASETS_PATH = paths.datasets_path()
# Update CONFIG with absolute path
CONFIG = {
    "MAX_FILE_SIZE": 10 * 1024 * 1024 * 1024,  # 10GB
    "UPLOAD_DIR": os.path.abspath(
        os.path.join(AUTOTUNE_DATASETS_PATH)
    ),  # Absolute path
    "SUPPORTED_FORMATS": {
        "csv": {"mime_type": "text/csv", "extensions": [".csv"]},
        "json": {"mime_type": "application/json", "extensions": [".json"]},
        "jsonl": {"mime_type": "application/jsonl", "extensions": [".jsonl", ".jl"]},
        "text": {"mime_type": "text/plain", "extensions": [".txt", ".text", ".log"]},
        "parquet": {
            "mime_type": "application/octet-stream",
            "extensions": [".parquet"],
        },
    },
}


async def stream_to_disk(file: UploadFile, save_path: str, mode: str = "wb") -> str:
    """
    Copy an UploadFile to ``save_path`` in bounded chunks.

    Memory stays flat (~STREAM_CHUNK_SIZE) regardless of file size — this is the
    streaming replacement for ``content = await file.read()``. Starlette already
    spools large uploads to a temp file on disk, so this is a disk→disk copy.

    ``mode`` is ``"wb"`` for a fresh write or ``"ab"`` to append (chunked uploads).
    Returns the saved path.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    await file.seek(0)

    handle = await asyncio.to_thread(open, save_path, mode)
    try:
        while True:
            chunk = await file.read(STREAM_CHUNK_SIZE)
            if not chunk:
                break
            await asyncio.to_thread(handle.write, chunk)
    finally:
        await asyncio.to_thread(handle.close)

    await file.seek(0)
    return save_path


def _split_assignment(seed: str, ratio: int):
    """
    Return a zero-arg callable yielding True (→ train) for ~``ratio`` percent of
    rows. Seeded by dataset identity so a given upload splits reproducibly while
    still shuffling records across the train/validation boundary.
    """
    rng = random.Random(seed)
    return lambda: rng.random() * 100 < ratio


def _remap_record(record: dict, column_mapping: Dict[str, str]) -> dict:
    """
    Apply a {target_column: source_column} mapping to a single JSON record.

    Mirrors the frontend's previous ``applyColumnMapping``: keep only the mapped
    target columns, pulling each value from its source column when present. This
    moves JSONL column mapping server-side so the client can stream the raw file.
    """
    mapped = {}
    for target_col, source_col in column_mapping.items():
        if source_col and source_col in record:
            mapped[target_col] = record[source_col]
    return mapped


def _jsonl_out_line(line: str, column_mapping: Optional[Dict[str, str]]) -> str:
    """Return the JSONL line to persist, applying column mapping when provided."""
    if not column_mapping:
        return line if line.endswith("\n") else line + "\n"
    record = json.loads(line)
    return json.dumps(_remap_record(record, column_mapping)) + "\n"


async def stream_split_jsonl(
    src_path: str,
    train_path: str,
    val_path: str,
    train_set_percentage: int,
    seed: str,
    column_mapping: Optional[Dict[str, str]] = None,
) -> tuple:
    """
    Split a JSONL file into train/validation by streaming it line by line.

    Constant memory: never holds more than one line. Preserves the original
    "shuffle then split by ratio" semantics via a seeded per-row assignment, and
    applies ``column_mapping`` per record when provided (server-side equivalent
    of the previous client-side mapping). Returns (train_records, validation_records).
    """

    def _run():
        assign_to_train = _split_assignment(seed, train_set_percentage)
        train_n = val_n = 0
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(val_path), exist_ok=True)
        with (
            open(src_path, "r", encoding="utf-8") as src,
            open(train_path, "w", encoding="utf-8") as tr,
            open(val_path, "w", encoding="utf-8") as va,
        ):
            for line in src:
                if not line.strip():
                    continue
                out_line = _jsonl_out_line(line, column_mapping)
                if assign_to_train():
                    tr.write(out_line)
                    train_n += 1
                else:
                    va.write(out_line)
                    val_n += 1
        return train_n, val_n

    return await asyncio.to_thread(_run)


async def remap_jsonl_file(
    src_path: str,
    dest_path: str,
    column_mapping: Optional[Dict[str, str]] = None,
) -> int:
    """
    Stream a JSONL file applying ``column_mapping`` per record (or copy verbatim
    when no mapping). Used by the custom-validation upload path where train/val
    files arrive separately and must each be mapped without an in-memory load.
    Returns the record count written.
    """

    def _run():
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        count = 0
        with (
            open(src_path, "r", encoding="utf-8") as src,
            open(dest_path, "w", encoding="utf-8") as dst,
        ):
            for line in src:
                if not line.strip():
                    continue
                dst.write(_jsonl_out_line(line, column_mapping))
                count += 1
        return count

    return await asyncio.to_thread(_run)


async def stream_split_parquet(
    src_path: str,
    train_path: str,
    val_path: str,
    train_set_percentage: int,
    seed: str,
    column_mapping: Optional[Dict[str, str]] = None,
) -> tuple:
    """
    Split a Parquet file into train/validation by streaming row groups.

    Reads in batches via ``iter_batches`` and writes through two ParquetWriters,
    so peak memory is one batch — not the whole table. Applies the same column
    rename/select mapping used elsewhere. Returns (train_records, validation_records).
    """

    def _run():
        assign_to_train = _split_assignment(seed, train_set_percentage)
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(val_path), exist_ok=True)

        reader = pq.ParquetFile(src_path)

        def _remap(table: "pa.Table") -> "pa.Table":
            if not column_mapping:
                return table
            reverse_mapping = {v: k for k, v in column_mapping.items() if v}
            new_names = [reverse_mapping.get(name, name) for name in table.column_names]
            table = table.rename_columns(new_names)
            mapped_cols = [k for k in column_mapping.keys() if k in table.column_names]
            if mapped_cols:
                table = table.select(mapped_cols)
            return table

        train_writer = val_writer = None
        train_n = val_n = 0
        try:
            for batch in reader.iter_batches(batch_size=10_000):
                table = _remap(pa.Table.from_batches([batch]))
                # Per-row mask preserves shuffled assignment within each batch.
                mask = [assign_to_train() for _ in range(table.num_rows)]
                train_tbl = table.filter(pa.array(mask))
                val_tbl = table.filter(pa.array([not m for m in mask]))

                if train_tbl.num_rows:
                    if train_writer is None:
                        train_writer = pq.ParquetWriter(train_path, train_tbl.schema)
                    train_writer.write_table(train_tbl)
                    train_n += train_tbl.num_rows
                if val_tbl.num_rows:
                    if val_writer is None:
                        val_writer = pq.ParquetWriter(val_path, val_tbl.schema)
                    val_writer.write_table(val_tbl)
                    val_n += val_tbl.num_rows
        finally:
            if train_writer is not None:
                train_writer.close()
            if val_writer is not None:
                val_writer.close()
            reader.close()
        return train_n, val_n

    return await asyncio.to_thread(_run)


# Create a function to handle file saving
async def save_dataset_content(
    content: List[Dict],
    filename: str,
    dataset_id: str,
    dataset_name: str,
) -> str:
    """
    Save dataset content (already parsed) to a JSONL file
    Returns the saved file path
    """
    try:
        dir_path = os.path.join(CONFIG["UPLOAD_DIR"], dataset_id, dataset_name)
        save_path = os.path.join(dir_path, filename)

        def _write():
            os.makedirs(dir_path, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                for item in content:
                    f.write(json.dumps(item) + "\n")

        await asyncio.to_thread(_write)
        return save_path
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save dataset content: {str(e)}"
        )


async def save_raw_parquet_with_mapping(
    file: UploadFile,
    filename: str,
    dataset_id: str,
    dataset_name: str,
    column_mapping: Optional[Dict[str, str]] = None,
) -> tuple:
    """
    Stream a raw parquet UploadFile to disk, apply column mapping per row group,
    and write the mapped parquet out. Peak memory is one row-group batch rather
    than the whole file. Returns (save_path, record_count).
    """
    if not filename.endswith(".parquet"):
        filename = filename.rsplit(".", 1)[0] + ".parquet"

    dir_path = os.path.join(CONFIG["UPLOAD_DIR"], dataset_id, dataset_name)
    save_path = os.path.join(dir_path, filename)

    # Stream the upload to a temp file first so pyarrow can read it with random
    # access without buffering the raw bytes in memory.
    tmp_path = save_path + ".raw"
    await stream_to_disk(file, tmp_path)

    def _process_and_write():
        reader = pq.ParquetFile(tmp_path)
        reverse_mapping = (
            {v: k for k, v in column_mapping.items() if v} if column_mapping else None
        )

        writer = None
        num_rows = 0
        try:
            for batch in reader.iter_batches(batch_size=10_000):
                table = pa.Table.from_batches([batch])
                if reverse_mapping and column_mapping:
                    new_names = [
                        reverse_mapping.get(name, name) for name in table.column_names
                    ]
                    table = table.rename_columns(new_names)
                    present = [
                        c for c in column_mapping.keys() if c in table.column_names
                    ]
                    if present:
                        table = table.select(present)
                if writer is None:
                    writer = pq.ParquetWriter(save_path, table.schema)
                writer.write_table(table)
                num_rows += table.num_rows
        finally:
            if writer is not None:
                writer.close()
            reader.close()
        return num_rows

    try:
        num_rows = await asyncio.to_thread(_process_and_write)
    finally:
        await asyncio.to_thread(
            lambda: os.path.exists(tmp_path) and os.remove(tmp_path)
        )
    await file.seek(0)

    return save_path, num_rows


async def save_parquet_dataset_content(
    content: List[Dict],
    filename: str,
    dataset_id: str,
    dataset_name: str,
    column_mapping: Optional[Dict[str, str]] = None,
) -> str:
    """
    Save dataset content as a Parquet file, optionally applying column mapping.
    Returns the saved file path.
    """
    try:
        if not filename.endswith(".parquet"):
            filename = filename.rsplit(".", 1)[0] + ".parquet"

        dir_path = os.path.join(CONFIG["UPLOAD_DIR"], dataset_id, dataset_name)
        save_path = os.path.join(dir_path, filename)

        def _write():
            os.makedirs(dir_path, exist_ok=True)
            table = pa.Table.from_pylist(content)

            if column_mapping:
                reverse_mapping = {v: k for k, v in column_mapping.items() if v}
                new_names = [
                    reverse_mapping.get(name, name) for name in table.column_names
                ]
                table = table.rename_columns(new_names)
                mapped_cols = [
                    k for k in column_mapping.keys() if k in table.column_names
                ]
                if mapped_cols:
                    table = table.select(mapped_cols)

            pq.write_table(table, save_path)

        await asyncio.to_thread(_write)
        return save_path
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save parquet dataset content: {str(e)}"
        )
