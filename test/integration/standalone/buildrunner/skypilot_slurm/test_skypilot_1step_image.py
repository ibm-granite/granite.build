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

"""Integration test for the command step WITH an image on Skypilot/slurm.

Validates the full pipeline when command_config.image is set, so the command
runs inside a container image (image_id: docker:<image>) rather than on the bare
node:
  HF URI input  ->  command step in a container on Skypilot/slurm  ->  HF output

Running an image on SLURM requires the Pyxis SPANK plugin. The local Docker SLURM
fixture (make slurm-setup) has no Pyxis, so this test auto-skips there (and on any
unreachable cluster) and only runs against a Pyxis-enabled cluster. The bare-node
command path is covered by test_skypilot_1step.py; the multi-target/mem:// path by
test_skypilot_slurm_2target.py.

The fixture's build.yaml and buildtest.yaml live in the directory returned by
_get_yaml_spec_dir below.
"""

from pathlib import Path

import pytest
from integration.environment.test_skypilot_slurm_e2e import (
    _slurm_cluster_reachable,
    _slurm_cluster_supports_containers,
)
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

pytestmark = pytest.mark.skypilot_integration


# Real-infra build test (launches a SLURM job via Skypilot) — only run in the
# extended suite (make extended-tests), not the fast quick-tests suite.
@extended_testing_only
@pytest.mark.skipif(
    not _slurm_cluster_reachable(),
    reason="Docker SLURM cluster not reachable (run: make slurm-setup)",
)
@pytest.mark.skipif(
    not _slurm_cluster_supports_containers(),
    reason="SLURM cluster has no Pyxis/container support — image-based steps "
    "require the Pyxis SPANK plugin (the local make slurm-setup fixture lacks it)",
)
class TestSkypilotSlurm1StepImage(AbstractYamlBuildRunnerTest):
    """HF input -> command step in a container on slurm via Skypilot -> HF output."""

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "1step-image"
