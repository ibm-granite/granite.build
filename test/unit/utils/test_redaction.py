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

"""Unit tests for gbserver.utils.redaction.redact_sensitive.

Guards the key-name based masking used before step config/metadata is emitted into
the member-readable build-lineage facet.
"""

import pytest

from gbserver.utils.redaction import REDACTED, redact_sensitive


@pytest.mark.standalone
def test_masks_common_secret_key_names():
    """Keys whose names look secret are masked; ``-``/``_`` and case tolerant."""
    result = redact_sensitive(
        {
            "password": "p",
            "PASSWD": "p",
            "pwd": "p",
            "db_pwd": "p",
            "api_key": "k",
            "api-key": "k",
            "apiKey": "k",
            "access_key": "a",
            "private-key": "pk",
            "token": "t",
            "credential": "c",
            "SECRET": "s",
        }
    )
    assert set(result.values()) == {REDACTED}


@pytest.mark.standalone
def test_non_secret_keys_pass_through():
    """Operational keys (e.g. commit_hash) are returned unchanged."""
    data = {"commit_hash": "deadbeef", "uri": "space://steps/byoc", "count": 3}
    assert redact_sensitive(data) == data


@pytest.mark.standalone
def test_recurses_into_nested_dicts_and_lists():
    """Nested dicts and dicts inside lists are redacted in place."""
    result = redact_sensitive(
        {
            "outer": {"token": "t", "ok": 1},
            "items": [{"secret": "s"}, {"name": "n"}],
        }
    )
    assert result == {
        "outer": {"token": REDACTED, "ok": 1},
        "items": [{"secret": REDACTED}, {"name": "n"}],
    }


@pytest.mark.standalone
def test_does_not_mutate_input():
    """The original mapping is left untouched (a copy is returned)."""
    original = {"token": "t", "nested": {"password": "p"}}
    redact_sensitive(original)
    assert original == {"token": "t", "nested": {"password": "p"}}


@pytest.mark.standalone
def test_scalars_returned_unchanged():
    """Non-container values pass through as-is."""
    assert redact_sensitive("plain") == "plain"
    assert redact_sensitive(42) == 42
    assert redact_sensitive(None) is None
