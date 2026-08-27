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

"""Tests for buildevent types."""

from gbserver.types.buildevent import EntityRunMetadata


class TestEntityRunMetadata:
    def test_from_dict_reads_build_name(self):
        meta = EntityRunMetadata.from_dict({"build_id": "abc", "build_name": "foo"})
        assert meta.build_name == "foo"

    def test_from_dict_defaults_build_name_empty(self):
        meta = EntityRunMetadata.from_dict({"build_id": "abc"})
        assert meta.build_name == ""

    def test_to_dict_includes_build_name(self):
        meta = EntityRunMetadata(build_id="abc", build_name="foo")
        d = meta.to_dict()
        assert d["build_name"] == "foo"

    def test_round_trip_preserves_build_name(self):
        meta = EntityRunMetadata(build_id="abc", build_name="foo")
        assert EntityRunMetadata.from_dict(meta.to_dict()).build_name == "foo"
