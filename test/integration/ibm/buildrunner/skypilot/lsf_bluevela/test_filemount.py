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

"""Integration test: Skypilot/BlueVela-LSF file_mounts copies a step-relative dir.

The BlueVela (LSF-over-Skypilot) sibling of the slurm ``test_skypilot_filemount``
test. The target runs a custom step (defined in a co-located test space) that
declares a ``file_mounts`` key copying the ``payload/`` directory shipped next to
its ``step.yaml`` onto the cluster. The step's ``run`` command asserts the
directory is present (failing the step, and thus the build, if it is missing),
exercising the Skypilot launcher's relative-source resolution against the
step.yaml dir on the LSF/bluevela path.

The ``file_mounts`` relative-source resolution is cloud-agnostic (same
``Skypilot`` environment class / ``type: skypilot`` launcher as slurm), so this
gives coverage that the feature works when the underlying cloud is BlueVela LSF.

Runs against the real BlueVela cluster (like the sibling bluevela build tests):
no local cluster probe. For this test to run in IBM SPS build tests it needs an
``environments/skypilot/lsf/bluevela/environment.yaml`` referencing the
``BV_SSH_PRIVATE_KEY`` secret; without that it uses the local space and expects a
local ``~/.ssh/ibm-bluevela.key`` (so it can be run locally with cluster access).

The fixture's build.yaml, buildtest.yaml, and test space live in the directory
returned by _get_yaml_spec_dir below.
"""

import os
from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

pytestmark = pytest.mark.ibm


# Real-infra build test (launches a BlueVela LSF job via Skypilot) — only run in
# the extended suite (make extended-tests), grouped with the other bluevela build
# tests so they don't launch concurrent jobs, and skipped in SPS CI/CD until the
# shared spaces carry a bluevela environment.yaml with the SSH-key secret ref.
@extended_testing_only
@pytest.mark.xdist_group(name="buildtest_bv")
@pytest.mark.skipif(
    os.environ.get("RUNNING_IN_CICD", "False").lower() == "true",
    reason="Skip in SPS CI/CD until we have environments/skypilot/lsf/bluevela/environment.yaml with key reference in gb-test and other space repos",
)
class TestSkypilotBlueVelaFileMount(AbstractYamlBuildRunnerTest):
    """Custom step copies a step-relative dir via file_mounts on BlueVela LSF."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "filemount"
