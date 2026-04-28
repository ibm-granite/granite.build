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
