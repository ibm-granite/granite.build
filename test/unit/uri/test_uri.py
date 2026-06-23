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

from pathlib import Path

import pytest

from gbcommon.uri.env import EnvURI
from gbcommon.uri.file import FileURI
from gbcommon.uri.uri import URI


def test_empty_uri():
    """Check if empty URI is rejected."""
    with pytest.raises(ValueError, match="uri cannot be None"):
        URI.get_uri(None)

    with pytest.raises(ValueError, match="uri cannot be empty"):
        URI.get_uri("")


def test_file_uri():
    expected_uri_str = "file:///path/to/folder"
    fileuri = URI.get_uri(uri=expected_uri_str)
    print("fileuri", fileuri)
    assert isinstance(fileuri, FileURI), f"invalid fileuri: {fileuri}"
    actual_uri_str = str(fileuri)
    assert (
        actual_uri_str == expected_uri_str
    ), f"expected: {expected_uri_str} actual: {actual_uri_str}"
    actual_uri_str = URI.get_uristr(fileuri)
    assert (
        actual_uri_str == expected_uri_str
    ), f"expected: {expected_uri_str} actual: {actual_uri_str}"


def test_env_uri():
    test_cases = [
        {
            "input": "env:///path/to/folder/in/environment",
            "expected": "env:///path/to/folder/in/environment",
        },
        {
            "input": "env://two/slashes",
            "expected": "env:///two/slashes",  # turns into three slashes
        },
    ]
    for test_case in test_cases:
        input_uri_str = test_case["input"]
        expected_uri_str = test_case["expected"]
        envuri = URI.get_uri(uri=input_uri_str)
        print("envuri", envuri)
        assert isinstance(envuri, EnvURI), f"invalid envuri: {envuri}"
        print("envuri.uri.path", envuri.uri.path)
        actual_uri_str = str(envuri)
        assert (
            actual_uri_str == expected_uri_str
        ), f"expected: {expected_uri_str} actual: {actual_uri_str}"
        actual_uri_str = URI.get_uristr(envuri)
        assert (
            actual_uri_str == expected_uri_str
        ), f"expected: {expected_uri_str} actual: {actual_uri_str}"


def test_file_uri_pull_dir_default_nests_source(tmp_path):
    """By default, pulling a DIRECTORY nests the source dir under dest
    (dest/<basename>/...). This is the long-standing behavior other callers
    (step/space asset materialization) rely on and must not change."""
    src = tmp_path / "adapter"
    src.mkdir()
    (src / "adapter_config.json").write_text("{}")
    dest = tmp_path / "dest"

    fileuri = URI.get_uri(f"file://{src}")
    assert isinstance(fileuri, FileURI)
    assert fileuri.pull(dest) is True

    # Source dir nested under dest.
    assert (dest / "adapter" / "adapter_config.json").read_text() == "{}"


def test_file_uri_pull_dir_copy_contents_opt_in(tmp_path):
    """copy_dir_contents=True copies a directory's CONTENTS into dest without the
    extra nesting level (used by the bash filestore push)."""
    src = tmp_path / "adapter"
    src.mkdir()
    (src / "adapter_config.json").write_text("{}")
    (src / "weights.safetensors").write_text("w")
    dest = tmp_path / "out" / "adapter_hash"

    fileuri = URI.get_uri(f"file://{src}")
    assert isinstance(fileuri, FileURI)
    assert fileuri.pull(dest, copy_dir_contents=True) is True

    # Contents landed directly in dest; the source dir was not nested under it.
    assert (dest / "adapter_config.json").read_text() == "{}"
    assert (dest / "weights.safetensors").read_text() == "w"
    assert not (dest / "adapter").exists()


def test_file_uri_pull_file_copies_file(tmp_path):
    """Pulling a single FILE copies the file to dest (dest parent exists)."""
    src = tmp_path / "result.json"
    src.write_text("data")
    dest = tmp_path / "result_copy.json"

    fileuri = URI.get_uri(f"file://{src}")
    assert isinstance(fileuri, FileURI)
    assert fileuri.pull(dest) is True
    assert Path(dest).read_text() == "data"
