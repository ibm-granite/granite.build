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

"""``gb step describe``: how a step's environment config is shown."""

from gbcli.services.service_step import parse_env_config
from gbcommon.types.stepconfig import StepEnvironmentTypeConfig

_LAUNCHERS = {"run": {"type": "skypilot", "config": {"run": "echo hi"}}}


def _describe(**env_config):
    return parse_env_config(
        StepEnvironmentTypeConfig.model_validate(
            {"launchers": _LAUNCHERS, **env_config}
        )
    )


def test_shows_a_single_node_step():
    """Without it, users learn the step is single-node only when a build fails."""
    assert _describe(single_node=True)["single_node"] is True


def test_omits_single_node_when_unset():
    assert "single_node" not in _describe()
    assert "single_node" not in _describe(single_node=False)
