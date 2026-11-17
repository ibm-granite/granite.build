# SPDX-License-Identifier: Apache-2.0

"""The legacy hand-authored ``space://steps/openinstruct-sft`` on SkyPilot/AWS refuses
multi-node builds (#683).

It trains on one machine (``accelerate launch --num_machines 1``), but with
``num_nodes: 2`` SkyPilot would provision 2 nodes and each would train on its own,
the #609 bug that #625 fixed for the framework step. The copy declares
``environment_configs.Skypilot.single_node``, so gbserver refuses, at step creation,
any build whose resolved ``num_nodes`` is not 1. The ``lsf/ibm-bluevela`` copy is
left as is: LSF is no longer supported.
"""

import pathlib

import pytest
import yaml

from gbserver.environment.skypilot import validate_launch_num_nodes
from gbserver.types.stepconfig import StepEnvironmentTypeConfig

AWS_STEP = (
    pathlib.Path(__file__).resolve().parents[3]
    / "configurations"
    / "assets"
    / "environments"
    / "skypilot"
    / "aws"
    / "steps"
    / "openinstruct-sft"
    / "step.yaml"
)
_SKY = yaml.safe_load(AWS_STEP.read_text())["environment_configs"]["Skypilot"]
_ENV = StepEnvironmentTypeConfig.model_validate(_SKY)
_LAUNCHER = _SKY["launchers"][_SKY["default_launcher"]]["config"]


def test_declares_single_node():
    assert _ENV.single_node is True


@pytest.mark.parametrize(
    "config",
    [
        {"compute_config": {"num_nodes": 2}},
        {"compute_config": {"num_nodes": 1}, "launcher_config": {"num_nodes": 2}},
    ],
)
def test_gbserver_refuses_more_than_one_node(config):
    with pytest.raises(ValueError, match="this step is single-node"):
        validate_launch_num_nodes(_ENV, _LAUNCHER, config)


@pytest.mark.parametrize("config", [{}, {"compute_config": {"num_nodes": 1}}])
def test_gbserver_accepts_one_node(config):
    """What the granite4-350m aws recipes send: num_nodes 1, or nothing."""
    assert validate_launch_num_nodes(_ENV, _LAUNCHER, config) == 1
