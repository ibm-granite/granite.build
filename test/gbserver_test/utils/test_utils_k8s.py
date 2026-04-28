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


from gbserver.utils.utils_k8s import is_helm_v4_or_higher


class TestHelmVersion:
    def test_is_helm_v4_or_higher(self):
        test_cases = [
            {
                "input": "v3.19.0",
                "expected": False,
            },
            {
                "input": "v4.2.0",
                "expected": True,
            },
            {
                "input": "v4.1.1+g5caf004",
                "expected": True,
            },
        ]
        for test_case in test_cases:
            test_input = test_case["input"]
            expected = test_case["expected"]
            actual = is_helm_v4_or_higher(test_input)
            assert (
                actual == expected
            ), f"test_input: {test_input} actual: {actual} expected: {expected}"
