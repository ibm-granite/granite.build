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

"""Integration test: the officeqa eval step runs end to end on BlueVela SLURM (Skypilot).

Runs the recipes' two-target shape on the shared
``space://environments/skypilot/slurm/bluevela``: officeqa-pro generates on
the family's GPU image (granite-4.2-3b served by vLLM, answering from
gold-page contexts built out of a synthetic parsed corpus) and scores on its
CPU image, bound by the granite_generation artifact. The dataset is the
synthetic one shipped beside this fixture (invented questions and parsed
pages, license-clean by construction; no gated upstream data): a trusted
local root, so the run needs no HF gate and no token while the context
build, the model path, the CSV re-scoring and the reward port all run for
real. One build test covers both family images.

The step comes from the Space ``make space`` renders (``space_uri:
../../space``), i.e. from your local source.

Run it with::

    make -C steps/eval/officeqa/skypilot test TESTS=test/slurm-bluevela

**Real cluster.** Extended-suite only, marked ``ibm``, and skipped unless the
BlueVela key for the ``granitebuild`` functional ID is at the path the
environment names. Also skipped in SPS CI, which has no key on the host.
"""

import os
from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

pytestmark = [pytest.mark.skypilot_integration, pytest.mark.ibm]

# The key the shared environment reads (cluster_ssh_configs IdentityFile).
_BLUEVELA_KEY = Path("~/.ssh/ibm-bluevela.key").expanduser()


@extended_testing_only
# No xdist_group: skypilot_integration already puts it in the serialized
# "skypilot" group, and under --dist=loadgroup any other group is a UsageError.
@pytest.mark.skipif(
    not _BLUEVELA_KEY.is_file(),
    reason=f"BlueVela granitebuild SSH key not found at {_BLUEVELA_KEY}",
)
@pytest.mark.skipif(
    os.environ.get("RUNNING_IN_CICD", "False").lower() == "true",
    reason="No BlueVela SSH key on SPS CI/CD hosts",
)
class TestSkypilotSlurmBlueVelaEvalOfficeqa(AbstractYamlBuildRunnerTest):
    """The officeqa step runs officeqa-pro end to end on BlueVela SLURM."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)

    @pytest.mark.timeout(3900)  # above buildtest.yaml's timeout_minutes (60)
    def test_runner(self):
        """Run the happy-path build with room for the buildtest timeout."""
        super().test_runner()
