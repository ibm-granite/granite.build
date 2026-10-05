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

import io
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

from gbcommon.utils.archive_safety import (
    MAX_ZIP_ENTRIES,
    MAX_ZIP_UNCOMPRESSED_BYTES,
    check_zip_safe,
)
from gbserver.types.constants import DEFAULT_DIR_PERMS
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# check_zip_safe and its two limits now live in gbcommon, which is shipped by
# distributions that do not include gbserver (granite-build-analytics packages
# gb_ui_backend + gbcommon). The import above is what keeps them importable from
# here, so existing `from gbserver.utils.archive import check_zip_safe` call
# sites are unaffected -- no re-assignment is needed for that, and the two that
# used to sit here were self-assignments that did nothing.


def check_tar_safe(
    tar: tarfile.TarFile,
    max_entries: int = MAX_ZIP_ENTRIES,
    max_uncompressed_bytes: int = MAX_ZIP_UNCOMPRESSED_BYTES,
) -> None:
    """Guard against tar-bomb archives before extracting any member.

    The tar counterpart of ``check_zip_safe``, using the same default caps.

    Args:
        tar: An open, seekable (non-stream mode) tar archive.
        max_entries: Maximum number of members allowed.
        max_uncompressed_bytes: Maximum total declared size of all members.

    Raises:
        ValueError: If the archive has more members, or more total
            uncompressed size, than the given caps.
    """
    members = tar.getmembers()
    if len(members) > max_entries:
        raise ValueError(
            f"archive has too many entries ({len(members)} > {max_entries})"
        )
    total_size = sum(m.size for m in members)
    if total_size > max_uncompressed_bytes:
        raise ValueError(
            f"archive uncompressed size too large "
            f"({total_size} > {max_uncompressed_bytes} bytes)"
        )


def _extract_zip(
    archive_binary: bytes, output_dir: Path, max_entries: int, max_bytes: int
) -> None:
    """Size-check and extract a zip archive into ``output_dir``.

    ``ZipFile.extractall`` already strips absolute paths and ``..`` components
    and never creates symlinks, so only the size caps need adding here.

    Raises:
        zipfile.BadZipFile, zipfile.LargeZipFile: If the bytes are not a zip.
        ValueError: If the archive exceeds the entry or size caps.
    """
    with zipfile.ZipFile(io.BytesIO(archive_binary), "r") as zip_file:
        check_zip_safe(zip_file, max_entries, max_bytes)
        zip_file.extractall(output_dir)


def _extract_tar(
    archive_binary: bytes, output_dir: Path, max_entries: int, max_bytes: int
) -> None:
    """Size-check and extract a tar archive into ``output_dir``.

    Extraction uses the ``"data"`` filter (PEP 706), which rejects absolute
    paths, ``..`` traversal, links pointing outside ``output_dir`` and device
    files, and clears setuid/setgid bits. Python 3.11-3.13 apply no filter by
    default, so it must be passed explicitly.

    Raises:
        tarfile.ReadError: If the bytes are not a tar archive.
        ValueError: If the archive exceeds the caps, contains a member the
            data filter rejects, or the interpreter predates PEP 706.
    """
    if not hasattr(tarfile, "data_filter"):
        # Python < 3.11.4 has no extraction filters; refuse rather than unpack
        # an untrusted archive unfiltered.
        raise ValueError("tar extraction requires Python >= 3.11.4")
    # Seekable "r:*" (not stream "r|*") so members can be size-checked first.
    with tarfile.open(fileobj=io.BytesIO(archive_binary), mode="r:*") as tar:
        check_tar_safe(tar, max_entries, max_bytes)
        try:
            tar.extractall(output_dir, filter="data")
        except tarfile.FilterError as e:
            raise ValueError(f"unsafe tar member rejected: {e}") from e


def extract_archive(
    archive_binary: bytes,
    output_dir: Path,
    archive_format: str = "",
    max_entries: int = MAX_ZIP_ENTRIES,
    max_uncompressed_bytes: int = MAX_ZIP_UNCOMPRESSED_BYTES,
) -> bool:
    """Extracts an untrusted archive to the filesystem, attempting auto-detection.

    Archives arrive from API clients (e.g. ``POST /builds/validate``), so
    extraction is confined to ``output_dir`` and capped in size.

    Args:
        archive_binary: The archive as a bytes object.
        output_dir: The directory to extract the archive to.
        archive_format: Optional. If provided ("zip" or "tar"), forces the format.
        max_entries: Maximum number of entries the archive may contain.
        max_uncompressed_bytes: Maximum total uncompressed size of all entries.

    Returns:
        True on success, False if the bytes are not a readable archive.

    Raises:
        ValueError: If the archive is readable but unsafe (too large, too many
            entries, or a tar member escaping ``output_dir``). There is no
            fallback to another format in that case.
    """
    output_dir.mkdir(mode=DEFAULT_DIR_PERMS, parents=True, exist_ok=True)

    if archive_format == "zip" or archive_format == "":
        try:
            _extract_zip(
                archive_binary, output_dir, max_entries, max_uncompressed_bytes
            )
            return True
        except (zipfile.BadZipFile, zipfile.LargeZipFile) as e:
            logger.error("failed to extract as zip, error: %s", e)
            if archive_format == "zip":
                return False

    if archive_format == "tar" or archive_format == "":
        try:
            _extract_tar(
                archive_binary, output_dir, max_entries, max_uncompressed_bytes
            )
            return True
        except tarfile.ReadError as e:
            logger.error("failed to extract as tar, error: %s", e)
            if archive_format == "tar":
                return False

    logger.error("failed to extract build binary")
    return False


def create_archive(
    dir: Path, output_path: Optional[Path] = None, format: str = "zip"
) -> Path:
    """Create a archive from the given directory.
    The return path will be different because the archive file
    extension with be added to the output path.

    Args:
        dir (_type_): _description_
        output_path (_type_, optional): _description_. Defaults to None.

    Returns:
        path: Path to the zip archive we just created.
    """
    if output_path is None:
        output_path = Path(tempfile.gettempdir()) / dir.name
    output_path = Path(shutil.make_archive(str(output_path), format, dir))
    assert output_path.is_file(), f"the output path {output_path} is not a file"
    return output_path


def create_archive_bytes(
    dir: Path, output_path: Optional[Path] = None, format: str = "zip"
) -> bytes:
    """Create the archive of the given directory and return as bytes.

    Args:
        dir (_type_): _description_
        archive_path (_type_, optional): Filename to write the archive to. Defaults to None
            so that we then create and delete the file internally.
            if Provided, then the archive will be left in the file system and should
            be removed/managed by the caller.
        format(str): one of the formats supported by shutil.make_archive.

    Returns:
        bytes: _description_
    """
    archive_path = create_archive(dir=dir, output_path=output_path, format=format)
    with open(archive_path, "rb") as f:
        archive_bytes = f.read()
    if output_path is not None:
        # Only remove the file if we created it internally here.
        shutil.rmtree(archive_path, ignore_errors=True)
    return archive_bytes


def cleanup_archive_dir(dir: Path) -> None:
    # If necessary, add additional operations (e.g. clear the readonly bit) to ensure that the directory is removed
    shutil.rmtree(dir)
