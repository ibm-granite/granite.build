# Copyright Granite.Build Authors
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

"""Tests for autotune.bridge_setup — the opt-in gate for AutoTuneX bridge logging."""

from autotune.bridge_setup import resolve_bridge_settings


def test_disabled_when_url_absent():
    enabled, base_url = resolve_bridge_settings(None)
    assert enabled is False
    assert base_url is None


def test_enabled_when_url_present():
    enabled, base_url = resolve_bridge_settings("https://my-bridge.example.com")
    assert enabled is True
    assert base_url == "https://my-bridge.example.com"


def test_trailing_slash_preserved_for_caller():
    # The helper does not normalize; AutoTuneXAPI handles rstrip internally.
    enabled, base_url = resolve_bridge_settings("https://x/")
    assert enabled is True
    assert base_url == "https://x/"


def test_empty_string_is_disabled():
    # An empty --autotunex_server_url="" is treated as "not provided".
    enabled, base_url = resolve_bridge_settings("")
    assert enabled is False
    assert base_url is None
