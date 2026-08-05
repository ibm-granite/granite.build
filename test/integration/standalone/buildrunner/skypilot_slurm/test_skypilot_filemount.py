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

"""Integration test: Skypilot/slurm file_mounts copies a step-relative directory.

Almost identical to the sibling ``test_skypilot_1step`` test, but the target
runs a custom step (defined in a co-located test space) that declares a
``file_mounts`` key copying the ``payload/`` directory shipped next to its
``step.yaml`` onto the cluster. The step's ``run`` command asserts the directory
is present (failing the step, and thus the build, if it is missing), exercising
the Skypilot launcher's relative-source resolution against the step.yaml dir.

Requires a running Docker SLURM cluster (see scripts/slurm/setup-slurm.sh).
Auto-skips when the cluster is not reachable via SSH.

The fixture's build.yaml, buildtest.yaml, and test space live in the directory
returned by _get_yaml_spec_dir below.
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
from libgbtest.constants import extended_testing_only

pytestmark = pytest.mark.skypilot_integration


# Real-infra build test (launches a SLURM job via Skypilot) — only run in the
# extended suite (make extended-tests), not the fast quick-tests suite.
@extended_testing_only
@pytest.mark.skipif(
    not _slurm_cluster_reachable(),
    reason="Docker SLURM cluster not reachable (run: make slurm-setup)",
)
class TestSkypilotSlurmFileMount(AbstractYamlBuildRunnerTest):
    """Custom step copies a step-relative dir via file_mounts on slurm."""

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "filemount"
