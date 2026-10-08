#!/usr/bin/env python3

# Copyright LLM.build Authors
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

"""Zip/tar archive safety checks, shared across distributions.

This lives in ``gbcommon`` rather than ``gbserver.utils.archive`` because
``granite-build-analytics`` ships ``gb_ui_backend`` and ``gbcommon`` **without**
``gbserver``, and ``gb_ui_backend`` needs this guard when it decodes build
archives out of the database.

Importing it from ``gbserver`` worked in-tree and failed in that distribution with
``No module named 'gbserver'``. The failure was invisible for a long time because
the call site wrapped the import in ``except Exception``, so the archive simply
never decoded and the Data Processing page reported no datasets rather than an
error.

The function is pure stdlib, so it has no business living in the server package.
"""

import io
import tarfile
import zipfile

# Caps for any uploaded build archive (zip or tar), despite the historical names.
MAX_ZIP_ENTRIES = 1000
MAX_ZIP_UNCOMPRESSED_BYTES = 50 * 1024 * 1024  # 50 MB


class ArchiveLimitError(ValueError):
    """An archive has more entries, or more uncompressed bytes, than allowed."""


def check_zip_safe(
    zf: zipfile.ZipFile,
    max_entries: int = MAX_ZIP_ENTRIES,
    max_uncompressed_bytes: int = MAX_ZIP_UNCOMPRESSED_BYTES,
) -> None:
    """Guard against zip-bomb archives before reading any entry.

    Args:
        zf: An open zip archive.
        max_entries: Maximum number of entries allowed.
        max_uncompressed_bytes: Maximum total declared size of all entries.

    Raises:
        ArchiveLimitError: (a ValueError) if the archive has more entries, or
            more total uncompressed size, than the given caps.
    """
    infos = zf.infolist()
    if len(infos) > max_entries:
        raise ArchiveLimitError(
            f"archive has too many entries ({len(infos)} > {max_entries})"
        )
    total_size = sum(info.file_size for info in infos)
    if total_size > max_uncompressed_bytes:
        raise ArchiveLimitError(
            f"archive uncompressed size too large "
            f"({total_size} > {max_uncompressed_bytes} bytes)"
        )


def check_tar_safe(
    tar: tarfile.TarFile,
    max_entries: int = MAX_ZIP_ENTRIES,
    max_uncompressed_bytes: int = MAX_ZIP_UNCOMPRESSED_BYTES,
) -> None:
    """Guard against tar-bomb archives before extracting any member.

    The tar counterpart of check_zip_safe, using the same default caps. Members
    are iterated rather than read with getmembers(): a tar has no central index,
    so headers are read one by one, and stopping at the first member past a cap
    keeps an oversized archive from loading (and, if compressed, decompressing)
    everything before it is rejected.

    Args:
        tar: An open, seekable (non-stream mode) tar archive.
        max_entries: Maximum number of members allowed.
        max_uncompressed_bytes: Maximum total declared size of all members.

    Raises:
        ArchiveLimitError: (a ValueError) if the archive has more members, or
            more total uncompressed size, than the given caps.
    """
    count = 0
    total_size = 0
    for member in tar:
        count += 1
        total_size += member.size
        if count > max_entries:
            raise ArchiveLimitError(
                f"archive has too many entries (more than {max_entries})"
            )
        if total_size > max_uncompressed_bytes:
            raise ArchiveLimitError(
                f"archive uncompressed size too large "
                f"(more than {max_uncompressed_bytes} bytes)"
            )


def check_archive_bytes_safe(
    archive_binary: bytes,
    max_entries: int = MAX_ZIP_ENTRIES,
    max_uncompressed_bytes: int = MAX_ZIP_UNCOMPRESSED_BYTES,
) -> None:
    """Apply the archive caps to raw zip or tar bytes without extracting them.

    Lets an upload be refused when it is submitted rather than when a runner
    later extracts it. Bytes that are neither a zip nor a tar are not checked
    (there is nothing to cap); extraction rejects them as before.

    Args:
        archive_binary: The archive as bytes.
        max_entries: Maximum number of entries allowed.
        max_uncompressed_bytes: Maximum total declared size of all entries.

    Raises:
        ArchiveLimitError: (a ValueError) if a zip or tar exceeds the caps.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(archive_binary)) as zf:
            check_zip_safe(zf, max_entries, max_uncompressed_bytes)
            return
    except (zipfile.BadZipFile, zipfile.LargeZipFile):
        pass
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_binary), mode="r:*") as tar:
            check_tar_safe(tar, max_entries, max_uncompressed_bytes)
    except tarfile.ReadError:
        pass
