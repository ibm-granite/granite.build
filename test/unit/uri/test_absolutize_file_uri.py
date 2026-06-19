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

"""Unit tests for absolutize_file_uri.

A relative ``file:`` URI is physically read/written relative to the process cwd,
so a registered artifact URI must be absolutized to match the on-disk location.
Absolute ``file:`` URIs and other schemes (hf:, lh:, cos:, env:) are unchanged.
"""

import pytest

from gbcommon.uri.file import absolutize_file_uri
from gbcommon.uri.uri import URI

FIXED_CWD = "/work/space"


@pytest.fixture(autouse=True)
def _fixed_cwd(monkeypatch):
    monkeypatch.setattr("gbcommon.uri.file.os.getcwd", lambda: FIXED_CWD)


@pytest.mark.standalone
def test_relative_file_dir_uri_is_absolutized():
    uri = absolutize_file_uri(URI.get_uri("file:outputs/lora-finetune/adapter_x/"))
    assert URI.get_uristr(uri) == f"file://{FIXED_CWD}/outputs/lora-finetune/adapter_x/"


@pytest.mark.standalone
def test_relative_file_uri_without_trailing_slash():
    uri = absolutize_file_uri(URI.get_uri("file:outputs/result.txt"))
    # No spurious trailing slash added for a non-directory URI.
    assert URI.get_uristr(uri) == f"file://{FIXED_CWD}/outputs/result.txt"


@pytest.mark.standalone
def test_absolute_file_uri_unchanged():
    original = "file:///abs/outputs/adapter_x/"
    uri = absolutize_file_uri(URI.get_uri(original))
    assert URI.get_uristr(uri) == original


@pytest.mark.standalone
def test_non_file_scheme_unchanged():
    # hf: is normalized by its own URI handler in get_uri; absolutize_file_uri
    # must not further alter it. Compare against get_uri's own output.
    parsed = URI.get_uri("hf:///ibm-granite/granite-4.0-h-350m")
    result = absolutize_file_uri(parsed)
    assert result is parsed
    assert URI.get_uristr(result) == URI.get_uristr(parsed)
