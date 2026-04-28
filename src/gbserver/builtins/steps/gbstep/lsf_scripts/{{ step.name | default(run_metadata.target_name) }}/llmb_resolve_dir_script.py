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
Find the given venv/conda env by searching the list of base dirs.
"""

import base64
import json
import os
from pathlib import Path


def main():
    """
    A python script to go through the list of directories,
    looking for a particular relative path, and print the first one that exists
    """
    rel_target_dir_str = os.getenv("INPUT_TARGET_DIR")
    assert isinstance(rel_target_dir_str, str), "INPUT_TARGET_DIR wasn't provided"
    rel_target_dir = Path(rel_target_dir_str)
    if rel_target_dir.is_absolute():
        if not rel_target_dir.is_dir():
            assert False, f"{rel_target_dir} is absolute, but not a directory"
        print(str(rel_target_dir), end=None)
        return

    base_dirs_json_b64 = os.getenv("INPUT_BASE_DIRS")
    assert isinstance(base_dirs_json_b64, str), "INPUT_BASE_DIRS wasn't provided"
    base_dirs = []
    try:
        base_dirs_json = base64.b64decode(base_dirs_json_b64).decode("utf-8")
        base_dirs = json.loads(base_dirs_json)
    except Exception as e:
        assert False, f"failed to decode and parse as json: {e}"
    assert isinstance(base_dirs, list), f"invalid base_dirs: {base_dirs}"
    assert len(base_dirs) > 0, "no base_dirs were provided"
    for base_dir_str in base_dirs:
        assert isinstance(base_dir_str, str)
        base_dir = Path(base_dir_str)
        if not base_dir.is_dir():
            # invalid, ignoring...
            continue
        venv_path = base_dir / rel_target_dir
        if venv_path.is_dir():
            print(str(venv_path), end=None)
            return
    assert False, f"{rel_target_dir} was not found in any of the base dirs {base_dirs}"


if __name__ == "__main__":
    main()
