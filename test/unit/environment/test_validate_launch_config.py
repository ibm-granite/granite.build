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

"""Environment.validate_launch_config: a step's launch constraints, checked at
step creation (TargetStepRun.__init__), before anything is provisioned."""

import asyncio
from types import SimpleNamespace

import pytest

from gbserver.environment.environment import Environment
from gbserver.types.stepconfig import StepEnvironmentTypeConfig


def _step_env(single_node: bool) -> StepEnvironmentTypeConfig:
    return StepEnvironmentTypeConfig.model_validate(
        {"launchers": {}, "single_node": single_node}
    )


class TestBaseEnvironment:
    """Environment is abstract; the base method only reads ``self.type``."""

    _SELF = SimpleNamespace(type="K8s")

    def test_without_the_flag_is_a_no_op(self):
        assert (
            Environment.validate_launch_config(
                self._SELF, _step_env(False), {}, {"compute_config": {"num_nodes": 4}}
            )
            is None
        )

    def test_refuses_the_flag_it_cannot_enforce(self):
        with pytest.raises(ValueError) as exc:
            Environment.validate_launch_config(self._SELF, _step_env(True), {}, {})
        assert str(exc.value) == (
            "The step sets single_node, but the K8s environment does not enforce "
            "it; only the SkyPilot environments do."
        )


@pytest.fixture(params=["Skypilot", "Skypilot_managed"])
def skypilot_env(request):
    if request.param == "Skypilot":
        from gbserver.environment.skypilot import Skypilot
        from gbserver.types.environmentconfig import EnvironmentConfig

        return Skypilot(
            event_q=asyncio.Queue(),
            environment_config=EnvironmentConfig(
                name="test-skypilot", type="Skypilot", config={"default_cloud": "k8s"}
            ),
        )
    from gbserver.environment.skypilot_managed import Skypilot_managed

    return Skypilot_managed(event_q=asyncio.Queue())


class TestSkypilotEnvironments:
    @pytest.mark.parametrize(
        "launcher_config,config",
        [
            ({}, {"compute_config": {"num_nodes": 2}}),
            ({"num_nodes": 2}, {"compute_config": {"num_nodes": 1}}),
            (
                {},
                {
                    "compute_config": {"num_nodes": 1},
                    "launcher_config": {"num_nodes": 2},
                },
            ),
        ],
    )
    def test_single_node_refuses_more_than_one(
        self, skypilot_env, launcher_config, config
    ):
        with pytest.raises(ValueError, match="this step is single-node"):
            skypilot_env.validate_launch_config(
                _step_env(True), launcher_config, config
            )

    def test_single_node_accepts_one(self, skypilot_env):
        assert (
            skypilot_env.validate_launch_config(
                _step_env(True), {}, {"compute_config": {"num_nodes": "01"}}
            )
            is None
        )

    def test_multi_node_without_the_flag_passes(self, skypilot_env):
        assert (
            skypilot_env.validate_launch_config(
                _step_env(False), {}, {"compute_config": {"num_nodes": 4}}
            )
            is None
        )

    def test_a_malformed_count_fails_at_creation(self, skypilot_env):
        with pytest.raises(ValueError, match="is not an integer"):
            skypilot_env.validate_launch_config(
                _step_env(False), {}, {"compute_config": {"num_nodes": "many"}}
            )
