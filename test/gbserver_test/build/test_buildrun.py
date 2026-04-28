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
Tests for build run.
The directory cannot be named 'build' because pytest ignores it.
See https://docs.pytest.org/en/stable/reference/reference.html#confval-norecursedirs
"""

from dataclasses import asdict

from gbserver.build.buildrun import (
    get_key_from_dict,
    match_event_selectors_against_payload,
)
from gbserver.types.buildconfig import (
    BuildTargetOutputConfig,
    BuildTargetOutputEventSelectorsConfig,
)
from gbserver.types.buildevent import ArtifactEventPayload


class TestBuildRun:
    """
    Tests for build run.
    """

    def test_get_key_from_dict(self):
        test_cases = [
            {"key": "a.b.c", "value": None, "data": {}},
            {"key": "a.b.c", "value": "foo", "data": {"a": {"b": {"c": "foo"}}}},
            {"key": "a.b.c", "value": 42, "data": {"a": {"b": {"c": 42}}}},
            {"key": "a.b.d", "value": None, "data": {"a": {"b": {"c": 42}}}},
            {"key": "a.b.0.c", "value": 42, "data": {"a": {"b": [{"c": 42}]}}},
            {"key": "a.b.10.c", "value": None, "data": {"a": {"b": [{"c": 42}]}}},
        ]
        for test_case in test_cases:
            k = test_case["key"]
            d = test_case["data"]
            expected = test_case["value"]
            actual = get_key_from_dict(k=k, d=d)
            assert actual == expected, f"failed on test_case {test_case}"

    def test_match_event_selectors_against_payload(self):
        test_cases = [
            {
                "output": BuildTargetOutputConfig(
                    event_selectors=[
                        BuildTargetOutputEventSelectorsConfig(
                            field_name="binding.path",
                            field_value="/path/to/outputs/step_1",
                        )
                    ]
                ),
                "payload": ArtifactEventPayload(
                    binding={"path": "/path/to/outputs/step_1"},
                ),
                "matched": True,
            },
            {
                "output": BuildTargetOutputConfig(
                    event_selectors=[
                        BuildTargetOutputEventSelectorsConfig(
                            field_name="binding.path",
                            field_value_regex="step_.*",
                        )
                    ]
                ),
                "payload": ArtifactEventPayload(
                    binding={"path": "/path/to/outputs/step_1234"},
                ),
                "matched": True,
            },
            {
                "output": BuildTargetOutputConfig(
                    event_selectors=[
                        BuildTargetOutputEventSelectorsConfig(
                            field_name="binding.path",
                            field_value_regex="step_.*",
                        )
                    ]
                ),
                "payload": ArtifactEventPayload(
                    binding={"path": "/path/to/outputs/checkpoint_10"},
                ),
                "matched": False,
            },
            {
                "output": BuildTargetOutputConfig(
                    event_selectors=[
                        BuildTargetOutputEventSelectorsConfig(
                            field_name="binding.path",
                            field_value_regex="^/proj/data-eng/step_.*$",
                        )
                    ]
                ),
                "payload": ArtifactEventPayload(
                    binding={"path": "/path/to/outputs/step_1234"},
                ),
                "matched": False,
            },
            {
                "output": BuildTargetOutputConfig(
                    event_selectors=[
                        BuildTargetOutputEventSelectorsConfig(
                            field_name="binding.path",
                            field_value_regex="^/proj/data-eng/step_.*$",
                        )
                    ]
                ),
                "payload": ArtifactEventPayload(
                    binding={"path": "/proj/data-eng/step_1234"},
                ),
                "matched": True,
            },
            {
                "output": BuildTargetOutputConfig(
                    event_selectors=[
                        BuildTargetOutputEventSelectorsConfig(
                            field_name="binding.path",
                            field_value_regex="step_.*",
                        ),
                        BuildTargetOutputEventSelectorsConfig(
                            field_name="binding_id",
                            field_value="final_checkpoint",
                        ),
                    ]
                ),
                "payload": ArtifactEventPayload(
                    binding_id="final_checkpoint",
                    binding={"path": "/proj/data-eng/checkpoint-10"},
                ),
                "matched": True,
            },
            {
                "output": BuildTargetOutputConfig(
                    event_selectors=[
                        BuildTargetOutputEventSelectorsConfig(
                            field_name="binding.path",
                            field_value_regex="step_.*",
                        ),
                        BuildTargetOutputEventSelectorsConfig(
                            field_name="binding_id",
                            field_value="intermediate_checkpoint",
                        ),
                    ]
                ),
                "payload": ArtifactEventPayload(
                    binding_id="final_checkpoint",
                    binding={"path": "/proj/data-eng/checkpoint-10"},
                ),
                "matched": False,
            },
        ]
        for test_case in test_cases:
            o = test_case["output"]
            payload = test_case["payload"]
            expected = test_case["matched"]
            payload_dict = asdict(payload)
            actual = match_event_selectors_against_payload(o, payload_dict)
            assert actual == expected, f"failed on test_case {test_case}"
