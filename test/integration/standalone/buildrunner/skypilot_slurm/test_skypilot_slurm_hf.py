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

"""Integration test for the image step on Skypilot/slurm with HF URI input/output.

Validates the full pipeline:
  HF URI input  ->  image step on Skypilot/slurm  ->  HF URI output

Requires a running Docker SLURM cluster (see scripts/slurm/setup-slurm.sh).
Auto-skips when the cluster is not reachable via SSH.

The fixture's build.yaml and buildtest.yaml live in the directory returned by
_get_yaml_spec_dir below.
"""

from pathlib import Path

import pytest

from integration.environment.test_skypilot_slurm_e2e import (
    _slurm_cluster_reachable,
)
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)

pytestmark = pytest.mark.skypilot_integration


@pytest.mark.skipif(
    not _slurm_cluster_reachable(),
    reason="Docker SLURM cluster not reachable (run: make slurm-setup)",
)
class TestSkypilotSlurmNativeHF(AbstractYamlBuildRunnerTest):
    """HF input -> native step on slurm via Skypilot -> HF output."""

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "native"
