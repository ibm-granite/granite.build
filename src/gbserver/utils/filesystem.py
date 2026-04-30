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

"""
Functions for copying files/folders in/between the local and remote filesystems.
"""

import atexit
import os
import re
import shutil
import subprocess
import tempfile
import traceback
from glob import glob
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import yaml

from gbserver.types.constants import DEFAULT_DIR_PERMS
from gbserver.utils.logger import get_logger
from gbserver.utils.template import fill_template, fill_template_in_file

logger = get_logger(__name__)

_TEMP_DIRS = []


def sync_or_copy_helper(
    src: str, dest: str, delete: bool = True, raise_errors: bool = False
) -> bool:
    """Same as sync_or_copy but with shutil."""
    cwd = os.getcwd()
    logger.info(
        "trying with shutil.copytree/copy2"
        + " src: %s dest: %s cwd: %s delete: %s raise_errors: %s",
        src,
        dest,
        cwd,
        delete,
        raise_errors,
    )
    # FIXME: the analysis shows this path isn't really working as intended
    # - make sure to have "rsync" available to make the code work for the time being.
    try:
        if delete and Path(dest).exists():
            if os.path.isdir(dest):
                shutil.rmtree(dest, ignore_errors=True)
            else:
                os.remove(dest)
        if os.path.isdir(src):
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest)
        return True
    except Exception as e:
        if raise_errors:
            raise ValueError(f"failed to copy asset from '{src}' to '{dest}' cwd: '{cwd}'") from e
        logger.error("shutil.copy failed, error: %s", e)
        logger.error("%s", traceback.format_exc())
    return False


def sync_or_copy(
    src: Union[str, Path],
    dest: Union[str, Path],
    delete=True,
    raise_errors: bool = False,
) -> bool:
    """Sync or copy the source file/folder to the destination."""
    src = str(src)
    dest = str(dest)
    if os.path.isdir(src):
        os.makedirs(dest, exist_ok=True)
    cwd = os.getcwd()
    try:
        command = ["rsync", "-avz", src, dest]
        if delete:
            command = ["rsync", "-avz", "--delete", src, dest]
        logger.info("command: %s", command)
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode == 0:
            return True
        logger.error("rsync failed, returncode: %s", result.returncode)
        logger.error("stdout: %s", result.stdout)
        logger.error("stderr: %s", result.stderr)
        if raise_errors:
            raise ValueError(f"failed to rsync asset from '{src}' to '{dest}' cwd: '{cwd}'")
        logger.error("%s", traceback.format_exc())
        return False
    except FileNotFoundError as nfe:
        logger.error("rsync may not available, error: %s", nfe)
        return sync_or_copy_helper(src=src, dest=dest, delete=delete, raise_errors=raise_errors)


def merge_config_dirs(base_dir: Path, overlay_dir: Path, merged_dir: Path):
    """Merge the overlay_dir on top of the base_dir by copying both to merged_dir"""
    sync_or_copy(base_dir, merged_dir)
    if overlay_dir is not None and overlay_dir.exists():
        sync_or_copy(overlay_dir, merged_dir, delete=False)
        find_and_merge_files(
            base_dir=base_dir,
            overlay_dir=overlay_dir,
            merged_dir=merged_dir,
            filepathpattern="*.yaml",
            encoder=yaml.safe_dump,
            decoder=yaml.safe_load,
        )


def find_and_merge_files(
    base_dir: Path,
    overlay_dir: Path,
    merged_dir: Path,
    filepathpattern: str,
    encoder: Callable[[dict], str],
    decoder: Callable[[str], dict],
) -> None:
    """Find and merge files of a particular type."""
    base_paths = glob(str(base_dir / "**" / filepathpattern), recursive=True)
    for _base_path in base_paths:
        base_path = Path(_base_path).resolve()
        rel_base_path = base_path.relative_to(base_dir)
        overlay_path = overlay_dir / rel_base_path
        merged_path = merged_dir / rel_base_path
        if not overlay_path.is_file():
            continue
        merge_file(base_path, overlay_path, merged_path, encoder, decoder)


def merge_file(
    base_path: Path,
    overlay_path: Path,
    merged_path: Path,
    encoder: Callable[[dict], str],
    decoder: Callable[[str], dict],
) -> None:
    """
    Merge the contents of the file at overlay_path
    on top of the file at base_path and write the
    result to merged_path
    """
    try:
        with open(base_path, "r", encoding="utf-8") as f:
            raw_base_data = f.read()
        base = decoder(raw_base_data)
        with open(overlay_path, "r", encoding="utf-8") as f:
            raw_overlay_data = f.read()
        overlay = decoder(raw_overlay_data)
        merged = merge_dicts(base, overlay)
        merged_data = encoder(merged)
        with open(merged_path, "w", encoding="utf-8") as f:
            f.write(merged_data)
    except Exception as e:
        logger.error(
            "failed to merge %s and %s , skipping, error: %s",
            base_path,
            overlay_path,
            e,
        )


def merge_dicts(base: Any, overlay: Any) -> Any:
    """Merge the 2 dictionaries recursively into one."""
    if isinstance(base, dict) and isinstance(overlay, dict):
        updated = {}
        for k in base:
            updated[k] = base[k]

        for k in overlay:
            # Preserve None instead of deleting it
            if k not in updated:
                updated[k] = overlay[k]
                continue

            # pylint: disable=unidiomatic-typecheck
            if type(base[k]) != type(overlay[k]):
                logger.warning(
                    "overwriting value of key %s because the values have different types %s and %s",
                    k,
                    type(base[k]),
                    type(overlay[k]),
                )
                updated[k] = overlay[k]
                continue
            updated[k] = merge_dicts(base[k], overlay[k])
        return updated
    return overlay


def _get_matching_ignored_files(ignore_file: str) -> list[str]:
    """Get the list of matching ignored files within the directory containing the given ignore file

    Args:
        ignore_file (str): path and file name of ignore file.  For example, .gitignore

    Returns:
        list[str]: list of existing files (includes files within subdirectories)
        relative to the directory containing the ignore file matching
        the regex/glob patterns found in the ignore file.
    """
    parent_dir = Path(ignore_file).parent
    ignored_files = []
    with open(ignore_file, "r", encoding="utf-8") as file:
        for match in file:
            match = re.sub(r"#.*", "", match)
            match = match.strip()
            if len(match) > 0:
                filepaths = glob(str(parent_dir / match), recursive=True)
                ignored_files.extend(filepaths)
    return ignored_files


def _get_ignored_files(d: Path, ignore_file_name: str) -> List[str]:
    """Get the list of all ignored files within the given directory tree implied by all the **/ignore_file_name files in the dir tree.

    Args:
        d (Path): directory in which to search for matching ignored files.
        ignore_file_name (str): .gitignore-like file specifying regex/glob matching patterns within the given directory

    Returns:
        list[str]: list of existing files in the given directory matching the regex/glob patterns found in the ignore file.
    """
    ignore_files = glob(str(d / "**" / ignore_file_name), recursive=True)
    ignored_files = []
    for ignore_file in ignore_files:
        file_ignores = _get_matching_ignored_files(ignore_file)
        ignored_files.extend(file_ignores)
    return ignored_files


def is_text_file(filepath: Path) -> bool:
    """Reads a few bytes from the file and decode as utf-8 text."""
    if not filepath.is_file():
        return False
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            f.read(4096)
        return True
    except UnicodeDecodeError:
        return False
    except Exception as e:
        logger.debug("An error occurred: %s", e)
        return False


def fill_templates_in_dir(
    d: Path,
    data: dict,
    ignore_paths: Optional[List[Path]] = None,
    strict: bool = False,
    fill_paths: bool = True,
    fill_files: bool = True,
) -> None:
    """
    Finds files in the directory and treats them as templates,
    filling them with the given data.
    """
    gbtemplateignorepaths = glob(str(d / "**" / ".gbtemplateignore"), recursive=True)
    ignore_dirs_for_templating = [Path(p).parent for p in gbtemplateignorepaths]
    filepaths = glob(str(d / "**" / "*"), recursive=True)
    ignored_files = _get_ignored_files(d=d, ignore_file_name=".gbignore")
    paths_to_remove: List[Path] = []
    for _filepath in filepaths:
        filepath = Path(_filepath)

        if (
            (filepath.parent in ignore_dirs_for_templating)
            or ((ignore_paths is not None) and (filepath in ignore_paths))
            or (str(filepath) in ignored_files)
        ):
            continue
        # --- path filling ---
        if fill_paths:
            filepath_str = str(filepath)
            newfilepath_str = fill_template(templ=str(filepath), data=data, strict=strict)
            newfilepath = Path(newfilepath_str)
            if not newfilepath.exists() or not os.path.samefile(filepath, newfilepath):
                if filepath.is_dir():
                    filepath_str = str(filepath) + "/"
                sync_or_copy(filepath_str, newfilepath_str, delete=False)
                paths_to_remove.append(filepath)
                filepath = newfilepath

        # --- file content filling ---
        if is_text_file(filepath) and fill_files:
            fill_template_in_file(filepath=filepath, data=data, strict=strict)

    for path_to_remove in paths_to_remove:
        if path_to_remove.exists():
            shutil.rmtree(path_to_remove, ignore_errors=True)


def create_temp_subdir(parent_dir: Union[str, Path], autodelete: bool = True) -> Path:
    """Create a subdirectory and optionally have it removed on exit
    Return a Path object to the new path of the directory.
    """
    if isinstance(parent_dir, str):
        parent_dir = Path(parent_dir)
    parent_dir.mkdir(mode=DEFAULT_DIR_PERMS, parents=True, exist_ok=True)
    temp_dir_path = tempfile.mkdtemp(dir=parent_dir)
    if autodelete:
        _TEMP_DIRS.append(temp_dir_path)
    return Path(temp_dir_path)


def _cleanup_temp_dirs():
    for tmpdir in _TEMP_DIRS:
        shutil.rmtree(tmpdir, ignore_errors=True)


atexit.register(_cleanup_temp_dirs)
