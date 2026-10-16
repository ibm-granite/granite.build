# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Pure dataset file helpers: stream, sniff, count, read, split, remap.

No HTTP and no database — these functions take file paths and byte sources and
raise the domain exceptions in :mod:`autotunex.core.exceptions`. Every function
except :func:`stream_to_staging` is synchronous and CPU/IO-bound; the upload
runner calls them inside :func:`asyncio.to_thread` so they never block the event
loop. Formats supported: ``jsonl``, ``csv``, ``parquet``.
"""

from __future__ import annotations

import csv
import json
import random
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any, Protocol

import pyarrow as pa
import pyarrow.parquet as pq

from autotunex.core.exceptions import (
    DatasetTooLargeError,
    EmptySplitError,
    UnsupportedDatasetFormatError,
)

CHUNK_SIZE = 8 * 1024 * 1024
"""Bounded read size: uploads stream in 8 MiB chunks, never fully buffered."""

PARQUET_BATCH_SIZE = 10_000
"""Row-group streaming batch size: parquet is read one batch at a time, never whole."""

ALLOWED_FORMATS: tuple[str, ...] = ("jsonl", "csv", "parquet")

_GZIP_WBITS = 16 + zlib.MAX_WBITS
"""zlib window size that decodes the gzip container (not raw deflate)."""

_EXTENSION_TO_FORMAT = {
    ".jsonl": "jsonl",
    ".json": "jsonl",
    ".csv": "csv",
    ".parquet": "parquet",
    ".pq": "parquet",
}


class SupportsAsyncRead(Protocol):
    """A source with an async, size-bounded ``read`` — e.g. Starlette's ``UploadFile``."""

    async def read(self, size: int) -> bytes:
        """Return up to ``size`` bytes, or empty at EOF."""
        ...


async def stream_to_staging(
    source: SupportsAsyncRead, dest: Path, *, max_bytes: int, gzip_encoded: bool
) -> int:
    """Stream ``source`` to ``dest`` in bounded chunks, returning bytes written.

    Decodes gzip incrementally when ``gzip_encoded`` (the size cap then applies
    to the *decompressed* output, which is what defends against a zip bomb).
    Never holds more than one chunk plus zlib's window in memory.

    Raises:
        DatasetTooLargeError: written bytes would exceed ``max_bytes``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    decompressor = zlib.decompressobj(_GZIP_WBITS) if gzip_encoded else None
    written = 0

    with dest.open("wb") as handle:

        def _write_capped(data: bytes) -> None:
            """Cap-check then write ``data``, raising before it hits disk over budget."""
            nonlocal written
            written += len(data)
            if written > max_bytes:
                raise DatasetTooLargeError(f"Upload exceeds the {max_bytes}-byte limit.")
            handle.write(data)

        while True:
            chunk = await source.read(CHUNK_SIZE)
            if decompressor is None:
                if not chunk:
                    break
                _write_capped(chunk)
                continue
            if not chunk:
                break
            # Drain this chunk in bounded pieces, re-feeding whatever
            # decompress() could not consume within the CHUNK_SIZE output cap
            # (decompressor.unconsumed_tail) until it is exhausted: a highly
            # compressible input (a zip bomb) must not expand to many GB in
            # one decompress() call before the cap is ever checked.
            pending = chunk
            while pending:
                data = decompressor.decompress(pending, CHUNK_SIZE)
                pending = decompressor.unconsumed_tail
                if data:
                    _write_capped(data)
        if decompressor is not None:
            # Bounded for the same reason: an unbounded flush() could still
            # materialize a large final buffer in one call.
            tail = decompressor.flush(CHUNK_SIZE)
            while tail:
                _write_capped(tail)
                tail = decompressor.flush(CHUNK_SIZE)
    return written


def sniff_format(filename: str) -> str:
    """Return the dataset format implied by ``filename``'s extension.

    A trailing ``.gz`` (the file was gzip-transferred) is stripped first.

    Raises:
        UnsupportedDatasetFormatError: the extension is not one of the allowed formats.
    """
    name = filename[:-3] if filename.lower().endswith(".gz") else filename
    suffix = Path(name).suffix.lower()
    fmt = _EXTENSION_TO_FORMAT.get(suffix)
    if fmt is None:
        raise UnsupportedDatasetFormatError(
            f"Unsupported file extension {suffix or filename!r}; expected one of {ALLOWED_FORMATS}."
        )
    return fmt


class _JsonArrayReader:
    """Streams the top-level elements of a JSON array from a text stream.

    Bounded memory: the internal buffer holds at most one element plus one read
    chunk, never the whole file. :meth:`json.JSONDecoder.raw_decode` parses each
    element with the real JSON grammar, so nested objects/arrays and strings that
    contain brackets or commas are handled correctly rather than by scanning for
    delimiters by hand.
    """

    def __init__(self, src: IO[str]) -> None:
        self._src = src
        self._decoder = json.JSONDecoder()
        self._buffer = ""
        self._eof = False

    def _refill(self) -> bool:
        """Append one chunk to the buffer; return whether any characters were read."""
        if self._eof:
            return False
        chunk = self._src.read(CHUNK_SIZE)
        if not chunk:
            self._eof = True
            return False
        self._buffer += chunk
        return True

    def _skip_whitespace(self) -> None:
        """Drop leading whitespace, refilling while the buffer is all whitespace."""
        self._buffer = self._buffer.lstrip()
        while not self._buffer and self._refill():
            self._buffer = self._buffer.lstrip()

    def consume_array_open(self) -> bool:
        """Return ``True`` and consume a leading ``[``; ``False`` if not an array."""
        self._skip_whitespace()
        if self._buffer[:1] != "[":
            return False
        self._buffer = self._buffer[1:]
        return True

    def elements(self) -> Iterator[Any]:
        """Yield each top-level array element in order.

        Raises:
            UnsupportedDatasetFormatError: the array is truncated or malformed.
        """
        while True:
            self._skip_whitespace()
            if not self._buffer:
                raise UnsupportedDatasetFormatError(
                    "Truncated JSON array: reached end of file before ']'."
                )
            if self._buffer[0] == "]":
                return
            yield self._decode_element()
            self._skip_whitespace()
            nxt = self._buffer[:1]
            if nxt == ",":  # more elements may follow (a trailing comma is tolerated)
                self._buffer = self._buffer[1:]
            elif nxt == "]":
                return
            else:
                raise UnsupportedDatasetFormatError(
                    "Malformed JSON array: expected ',' or ']' between elements."
                )

    def _decode_element(self) -> Any:  # noqa: ANN401 — a JSON element has no fixed shape
        """Decode one JSON value at the buffer head, refilling until it completes."""
        while True:
            try:
                value, end = self._decoder.raw_decode(self._buffer)
            except json.JSONDecodeError as exc:
                # Either the element spans the chunk boundary (refill and retry)
                # or the file is genuinely malformed (no more data -> give up).
                if self._refill():
                    continue
                raise UnsupportedDatasetFormatError(
                    "Malformed JSON array: could not parse an element."
                ) from exc
            # A value that consumed the entire buffer may itself be truncated at
            # the boundary (e.g. a bare number split across chunks); pull more
            # before trusting it, unless we are already at EOF.
            if end == len(self._buffer) and self._refill():
                continue
            self._buffer = self._buffer[end:]
            return value


def normalize_json_array_to_jsonl(path: Path) -> None:
    r"""Rewrite ``path`` as JSONL if it holds a standard JSON *array*; else leave it.

    A ``.json`` upload is mapped to the ``jsonl`` format and its raw bytes are
    streamed to disk unchanged (the browser no longer re-serializes the file —
    that crashed the renderer on GB-scale inputs), so a standard, often
    pretty-printed JSON array ``[ {...}, {...} ]`` lands as a ``.jsonl``-named
    file the line-by-line JSONL helpers cannot read (``json.loads('[\n')`` raises
    ``Expecting value``). This converts such a file, in place, to one compact
    JSON object per line so :func:`count_records`, :func:`split_by_percentage`
    and :func:`remap_records` all work unchanged.

    A file that is already JSONL — its first non-whitespace character is not
    ``[`` — is left byte-for-byte untouched. The array is streamed via
    :class:`_JsonArrayReader` over a bounded buffer (peak memory: one element
    plus one read chunk), holding the same never-fully-buffered contract as the
    other helpers here.

    Raises:
        UnsupportedDatasetFormatError: the file opens as an array but is not
            valid JSON (truncated or otherwise malformed).
    """
    temp = path.with_suffix(path.suffix + ".normalize")
    with path.open("r", encoding="utf-8") as src:
        reader = _JsonArrayReader(src)
        if not reader.consume_array_open():
            return  # already JSONL (or empty/whitespace) — nothing to rewrite
        with temp.open("w", encoding="utf-8") as out:
            for element in reader.elements():
                # ensure_ascii=False keeps non-Latin text (e.g. Arabic) readable
                # and the file smaller; both forms round-trip identically.
                out.write(json.dumps(element, ensure_ascii=False) + "\n")
    temp.replace(path)


def count_records(path: Path, data_format: str) -> int:
    """Return the record count for ``path`` in ``data_format``."""
    if data_format == "jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    if data_format == "csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = sum(1 for _ in csv.reader(handle))
        return max(rows - 1, 0)
    if data_format == "parquet":
        return int(pq.ParquetFile(path).metadata.num_rows)
    raise UnsupportedDatasetFormatError(f"Unsupported format {data_format!r}.")


def read_records(path: Path, data_format: str, *, limit: int) -> list[dict[str, Any]]:
    """Return at most ``limit`` records from ``path`` as plain dicts."""
    if limit <= 0:
        return []
    if data_format == "jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                records.append(json.loads(line))
                if len(records) >= limit:
                    break
        return records
    if data_format == "csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [row for _, row in zip(range(limit), reader, strict=False)]
    if data_format == "parquet":
        records = []
        batch_size = max(1, min(limit, PARQUET_BATCH_SIZE))
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                records.append(row)
                if len(records) >= limit:
                    return records
        return records
    raise UnsupportedDatasetFormatError(f"Unsupported format {data_format!r}.")


def _validation_indices(total: int, validation_percentage: int, seed: int) -> set[int]:
    """Pick a reproducible set of validation row indices.

    Raises:
        EmptySplitError: the chosen count leaves an empty train or validation split.
    """
    validation_count = round(total * validation_percentage / 100)
    if validation_count <= 0 or validation_count >= total:
        raise EmptySplitError(
            f"validation_percentage={validation_percentage} over {total} records "
            "would leave an empty train or validation split."
        )
    return set(random.Random(seed).sample(range(total), validation_count))


def split_by_percentage(
    source: Path,
    train_out: Path,
    validation_out: Path,
    *,
    data_format: str,
    validation_percentage: int,
    seed: int,
) -> tuple[int, int]:
    """Split ``source`` into train/validation files, returning ``(train_n, val_n)``.

    Selection is seeded (by the dataset id, upstream) so a re-run is identical.
    All three formats stream: jsonl/csv row-by-row, parquet by row-group batches
    into two ``ParquetWriter``s. Peak memory is one batch plus the validation
    index set (``O(validation_count)``); the row *content* is never all held at
    once. Row order and assignment match a whole-table split.

    Raises:
        EmptySplitError: either resulting split is empty.
    """
    total = count_records(source, data_format)
    validation_rows = _validation_indices(total, validation_percentage, seed)

    if data_format == "jsonl":
        with (
            source.open("r", encoding="utf-8") as src,
            train_out.open("w", encoding="utf-8") as train,
            validation_out.open("w", encoding="utf-8") as validation,
        ):
            index = 0
            for line in src:
                if not line.strip():
                    continue
                target = validation if index in validation_rows else train
                target.write(line if line.endswith("\n") else line + "\n")
                index += 1
    elif data_format == "csv":
        with (
            source.open("r", encoding="utf-8", newline="") as src,
            train_out.open("w", encoding="utf-8", newline="") as train,
            validation_out.open("w", encoding="utf-8", newline="") as validation,
        ):
            reader = csv.reader(src)
            header = next(reader)
            train_writer, validation_writer = csv.writer(train), csv.writer(validation)
            train_writer.writerow(header)
            validation_writer.writerow(header)
            for index, row in enumerate(reader):
                (validation_writer if index in validation_rows else train_writer).writerow(row)
    elif data_format == "parquet":
        reader = pq.ParquetFile(source)
        schema = reader.schema_arrow
        train_parquet_writer: pq.ParquetWriter | None = None
        validation_parquet_writer: pq.ParquetWriter | None = None
        try:
            index = 0
            for batch in reader.iter_batches(batch_size=PARQUET_BATCH_SIZE):
                chunk = pa.Table.from_batches([batch], schema=schema)
                to_validation = [(index + i) in validation_rows for i in range(chunk.num_rows)]
                index += chunk.num_rows
                validation_tbl = chunk.filter(pa.array(to_validation))
                train_tbl = chunk.filter(pa.array([not v for v in to_validation]))
                if train_tbl.num_rows:
                    if train_parquet_writer is None:
                        train_parquet_writer = pq.ParquetWriter(train_out, schema)
                    train_parquet_writer.write_table(train_tbl)
                if validation_tbl.num_rows:
                    if validation_parquet_writer is None:
                        validation_parquet_writer = pq.ParquetWriter(validation_out, schema)
                    validation_parquet_writer.write_table(validation_tbl)
        finally:
            if train_parquet_writer is not None:
                train_parquet_writer.close()
            if validation_parquet_writer is not None:
                validation_parquet_writer.close()
    else:
        raise UnsupportedDatasetFormatError(f"Unsupported format {data_format!r}.")

    return total - len(validation_rows), len(validation_rows)


def remap_records(path: Path, mapping: dict[str, str], *, data_format: str) -> None:
    """Rewrite ``path`` in place, projecting columns per ``{target: source}``.

    A target whose source is **blank** (an unassigned field, e.g. ``""``) or
    **absent from the data** is silently skipped rather than raising — matching
    the 2025 uploader's tolerant ``_remap_record`` (``if source_col and source_col
    in record``). This keeps a partial or blank mapping from failing the whole
    upload with a ``KeyError``; only sources that actually exist are projected.
    For jsonl the check is per-record (ragged rows are fine); for csv/parquet it
    is against the header / schema.
    """
    temp = path.with_suffix(path.suffix + ".remap")

    if data_format == "jsonl":
        with (
            path.open("r", encoding="utf-8") as src,
            temp.open("w", encoding="utf-8") as out,
        ):
            for line in src:
                if not line.strip():
                    continue
                row = json.loads(line)
                projected = {t: row[s] for t, s in mapping.items() if s and s in row}
                out.write(json.dumps(projected) + "\n")
    elif data_format == "csv":
        with (
            path.open("r", encoding="utf-8", newline="") as src,
            temp.open("w", encoding="utf-8", newline="") as out,
        ):
            reader = csv.DictReader(src)
            header = reader.fieldnames or []
            kept = [(t, s) for t, s in mapping.items() if s and s in header]
            writer = csv.DictWriter(out, fieldnames=[t for t, _ in kept])
            writer.writeheader()
            for row in reader:
                writer.writerow({t: row[s] for t, s in kept})
    elif data_format == "parquet":
        reader = pq.ParquetFile(path)
        schema = reader.schema_arrow
        kept = [(t, s) for t, s in mapping.items() if s and s in schema.names]
        sources = [s for _, s in kept]
        targets = [t for t, _ in kept]
        parquet_writer: pq.ParquetWriter | None = None
        try:
            for batch in reader.iter_batches(batch_size=PARQUET_BATCH_SIZE):
                chunk = pa.Table.from_batches([batch], schema=schema)
                projected = chunk.select(sources).rename_columns(targets)
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(temp, projected.schema)
                parquet_writer.write_table(projected)
            if parquet_writer is None:
                # A file with zero data rows still needs a valid, projected empty file.
                empty = (
                    pa.Table.from_batches([], schema=schema).select(sources).rename_columns(targets)
                )
                pq.write_table(empty, temp)
        finally:
            if parquet_writer is not None:
                parquet_writer.close()
    else:
        raise UnsupportedDatasetFormatError(f"Unsupported format {data_format!r}.")

    temp.replace(path)
