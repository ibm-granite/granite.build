# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Tests for the streaming-ZIP archive helper."""

from __future__ import annotations

import io
import zipfile
from collections.abc import AsyncIterator, Awaitable, Callable

from autotunex.services.storage.archive import stream_zip
from autotunex.services.storage.artifacts import OpenedArtifact


def _opener_from(files: dict[str, bytes]) -> Callable[[str], Awaitable[OpenedArtifact]]:
    async def opener(path: str) -> OpenedArtifact:
        data = files[path]

        async def _stream() -> AsyncIterator[bytes]:
            # Deliberately chunked to exercise multi-chunk entries.
            yield data[: len(data) // 2]
            yield data[len(data) // 2 :]

        return OpenedArtifact(
            filename=path.rsplit("/", 1)[-1],
            media_type="application/octet-stream",
            size=len(data),
            stream=_stream(),
        )

    return opener


async def _collect(paths: list[str], files: dict[str, bytes]) -> bytes:
    return b"".join([chunk async for chunk in stream_zip(paths, _opener_from(files))])


async def test_stream_zip_roundtrips_entries_and_content() -> None:
    files = {"final_config.json": b'{"a": 1}', "job.log": b"hello world log\n"}

    blob = await _collect(list(files), files)

    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        assert sorted(archive.namelist()) == sorted(files)
        for path, content in files.items():
            assert archive.read(path) == content


async def test_stream_zip_preserves_directory_structure_for_same_basenames() -> None:
    # The Results panel shows many files all named 'adapters.safetensors' at
    # different paths; the archive must keep them as distinct entries.
    files = {
        "0000069/adapters.safetensors": b"AAAA",
        "0000138/adapters.safetensors": b"BBBBBB",
        "adapters.safetensors": b"CC",
    }

    blob = await _collect(list(files), files)

    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        assert sorted(archive.namelist()) == sorted(files)
        assert archive.read("0000069/adapters.safetensors") == b"AAAA"
        assert archive.read("0000138/adapters.safetensors") == b"BBBBBB"
        assert archive.read("adapters.safetensors") == b"CC"


async def test_stream_zip_empty_produces_valid_empty_archive() -> None:
    blob = await _collect([], {})

    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        assert archive.namelist() == []
