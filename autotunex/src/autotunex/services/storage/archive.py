# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Build a ZIP archive of a job's output assets as a byte stream.

Used by the "Download all assets" endpoint. The archive is produced lazily and
never held whole in memory or on disk: each source file is opened through the
:class:`~autotunex.services.storage.artifacts.ArtifactLister` seam, written into
a ``ZIP_STORED`` (uncompressed — model weights are incompressible, and this
keeps CPU off the event loop) entry, and the growing archive is drained chunk by
chunk to the caller. Entry names are the assets' *relative paths*, so files that
share a basename across directories stay distinct.
"""

from __future__ import annotations

import zipfile
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

from autotunex.services.storage.artifacts import OpenedArtifact


class _StreamBuffer:
    """A write-only, non-seekable sink `zipfile` writes into; drained per chunk.

    Deliberately implements neither ``tell`` nor ``seek``: that makes
    :class:`zipfile.ZipFile` treat the stream as non-seekable and emit data
    descriptors instead of rewinding to patch each entry's header, which is what
    allows the archive to be produced as a forward-only byte stream.
    """

    def __init__(self) -> None:
        self._chunks: list[bytes] = []

    def write(self, data: bytes) -> int:
        self._chunks.append(bytes(data))
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    def drain(self) -> bytes:
        """Return everything written since the last drain and reset the buffer."""
        if not self._chunks:
            return b""
        data = b"".join(self._chunks)
        self._chunks.clear()
        return data


async def stream_zip(
    paths: Sequence[str],
    opener: Callable[[str], Awaitable[OpenedArtifact]],
) -> AsyncIterator[bytes]:
    """Yield the bytes of a ZIP archive containing every file in ``paths``.

    Args:
        paths: relative paths (used verbatim as archive entry names).
        opener: resolves a path to an :class:`OpenedArtifact` whose ``stream`` is
            consumed exactly once into the corresponding entry.

    Raises:
        Whatever ``opener`` raises for a given path (e.g.
        ``JobArtifactsNotFoundError``) propagates out of the generator; because a
        few entries may already have been yielded, a failure mid-archive yields a
        truncated download rather than a clean error.
    """
    buffer = _StreamBuffer()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in paths:
            opened = await opener(path)
            info = zipfile.ZipInfo(path)
            # force_zip64: entry size is unknown up front and may exceed 4 GiB.
            with archive.open(info, mode="w", force_zip64=True) as entry:
                async for chunk in opened.stream:
                    entry.write(chunk)
                    if drained := buffer.drain():
                        yield drained
            if drained := buffer.drain():
                yield drained
    if trailing := buffer.drain():
        yield trailing
