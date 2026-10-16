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

"""Integration test: Skypilot/BlueVela-SLURM file_mounts copies a step-relative dir into a container.

The BlueVela SLURM sibling of the bluevela-LSF ``test_filemount`` and the local
Docker ``test_skypilot_filemount`` tests. The target runs a custom step (defined
in a co-located test space) that declares a ``file_mounts`` key copying the
``payload/`` directory shipped next to its ``step.yaml`` onto the cluster. The
command runs INSIDE a container image (``command_config.image`` is set, so
image_id resolves to ``docker:<image>``), which on SLURM goes through the Pyxis
SPANK plugin BlueVela provides. The step's ``run`` command asserts the payload
directory is present *inside the container* (failing the step, and thus the
build, if it is missing), exercising both the Skypilot launcher's relative-source
resolution against the step.yaml dir AND the enroot/Pyxis container bind-mount of
the per-run workdir on the BlueVela SLURM path.

The ``skypilot/slurm/bluevela`` environment lives only in the remote gb-test
space, so the co-located test space chains to it via a ``base_uris`` git entry
(and uses the ``ibmcloud`` secret manager so the env's ``BV_SSH_PRIVATE_KEY``
reference resolves); the builtin ``monitors/skypilot`` is auto-resolved and
``assetstores/hf`` resolves from gb-test's own tree.

Runs against the real BlueVela SLURM cluster (like the sibling bluevela build
tests): no local cluster probe. For this test to run in IBM SPS build tests it
needs an ``environments/skypilot/slurm/bluevela/environment.yaml`` referencing
the ``BV_SSH_PRIVATE_KEY`` secret in the shared spaces; until then it is skipped
in CI/CD and run locally with ibmcloud secret access + BlueVela reachability.

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


# Real-infra build test (launches a BlueVela SLURM job via Skypilot) — only run
# in the extended suite (make extended-tests), grouped with the other bluevela
# build tests so they don't launch concurrent jobs, and skipped in SPS CI/CD
# until the shared spaces carry a bluevela SLURM environment.yaml with the
# SSH-key secret ref.
@extended_testing_only
@pytest.mark.xdist_group(name="buildtest_bv")
@pytest.mark.skipif(
    os.environ.get("RUNNING_IN_CICD", "False").lower() == "true",
    reason="Skip in SPS CI/CD until we have environments/skypilot/slurm/bluevela/environment.yaml with key reference in gb-test and other space repos",
)
class TestSkypilotBlueVelaSlurmFileMount(AbstractYamlBuildRunnerTest):
    """Custom step copies a step-relative dir via file_mounts into a container on BlueVela SLURM."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "filemount"
