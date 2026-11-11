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

"""StepEnvironmentTypeConfig fields a step author sets and gbserver enforces."""

from gbserver.types.stepconfig import StepEnvironmentTypeConfig


class TestSingleNode:
    def test_defaults_to_false(self):
        assert (
            StepEnvironmentTypeConfig.model_validate({"launchers": {}}).single_node
            is False
        )

    def test_parses_true(self):
        env = StepEnvironmentTypeConfig.model_validate(
            {"launchers": {}, "single_node": True}
        )
        assert env.single_node is True
