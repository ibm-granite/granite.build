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

"""Archive module."""

import io
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

from gbserver.types.constants import DEFAULT_DIR_PERMS
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


def extract_archive(archive_binary: bytes, output_dir: Path, archive_format: str = "") -> bool:
    """Extracts an archive to the filesystem, attempting auto-detection.

    Args:
        archive_binary: The archive as a bytes object.
        output_dir: The directory to extract the archive to.
        archive_format: Optional. If provided, forces the format.

    Returns:
        True on success, False on failure.
    """

    output_dir.mkdir(mode=DEFAULT_DIR_PERMS, parents=True, exist_ok=True)

    if archive_format == "zip" or archive_format == "":
        try:
            # alternative https://docs.python.org/3/library/shutil.html#shutil.unpack_archive
            zip_buffer = io.BytesIO(archive_binary)
            with zipfile.ZipFile(zip_buffer, "r") as zip_file:
                zip_file.extractall(output_dir)
                return True
        except (zipfile.BadZipFile, zipfile.LargeZipFile) as e:
            logger.error("failed to extract as zip, error: %s", e)
            if archive_format == "zip":
                return False

    if archive_format == "tar" or archive_format == "":
        try:
            tar_buffer = io.BytesIO(archive_binary)
            with tarfile.open(fileobj=tar_buffer, mode="r|*") as tar:
                tar.extractall(output_dir)
                return True
        except tarfile.ReadError as e:
            logger.error("failed to extract as tar, error: %s", e)
            if archive_format == "tar":
                return False

    logger.error("failed to extract build binary")
    return False


def create_archive(dir: Path, output_path: Optional[Path] = None, format: str = "zip") -> Path:
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
    """Cleanup archive dir."""
    shutil.rmtree(dir)
