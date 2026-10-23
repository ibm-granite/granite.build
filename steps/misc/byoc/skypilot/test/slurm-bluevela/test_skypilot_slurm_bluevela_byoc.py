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

"""Integration test: the byoc step runs end to end on BlueVela SLURM (Skypilot).

The fixture dir ``test-data/slurm-bluevela/`` is a **custom test space**: it holds
its own ``space.yaml`` (``secret_manager: ibmcloud``, chaining to the gb-test space
for the ``skypilot/slurm/bluevela`` environment), and ``buildtest.yaml`` sets
``space_uri: .``. ``make space`` (a ``make test`` prerequisite) links the fixture's
``steps`` to the generated ``../../space/steps``, so the step is resolved from your
**local source**, not from gb-test. See "Custom test spaces" in steps/README.md.

Mode-1 only: ``make publish-step`` does not publish this test. Run it with::

    make -C steps/misc/byoc/skypilot test

Needs ``IBM_CLOUD_API_KEY`` (for the ibmcloud secret manager) and ssh access to
github.ibm.com (for the gb-test base). Set ``GBTEST_SPS_IBMCLOUD_API_KEY`` to load
it from SPS (steps/conftest.py brings in test/conftest.py, which does that).
Skipped in SPS CI/CD, like the other BlueVela build tests.
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


@extended_testing_only
@pytest.mark.xdist_group(name="buildtest_bv")
@pytest.mark.skipif(
    os.environ.get("RUNNING_IN_CICD", "False").lower() == "true",
    reason="Skip in SPS CI/CD until the bluevela environment.yaml references BV_SSH_PRIVATE_KEY in the space repos",
)
@pytest.mark.skipif(
    not os.environ.get("IBM_CLOUD_API_KEY"),
    reason="IBM_CLOUD_API_KEY (set GBTEST_SPS_IBMCLOUD_API_KEY) is required by the custom test space's ibmcloud secret manager",
)
class TestSkypilotSlurmBlueVelaByoc(AbstractYamlBuildRunnerTest):
    """byoc clones a public repo and runs a command end to end on BlueVela SLURM."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding build.yaml, buildtest.yaml and space.yaml."""
        return get_test_data_dir_for(__file__)

    @pytest.mark.timeout(2100)  # above buildtest.yaml's timeout_minutes (30)
    def test_runner(self):
        """Run the happy-path build with room for the buildtest timeout."""
        super().test_runner()
