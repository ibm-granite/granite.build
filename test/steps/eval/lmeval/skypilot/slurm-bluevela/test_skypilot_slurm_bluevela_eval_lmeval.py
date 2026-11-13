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

"""Integration test: the lmeval eval step runs end to end on BlueVela SLURM (Skypilot).

Runs the model-free ``answers=gold`` mode of mmlu-prox-lite on the shared
``space://environments/skypilot/slurm/bluevela``: the GoldLM answers every
question with its reference letter, so the run serves no model and needs no
GPU, no LLM keys and no secrets. What it proves end to end: the step
dispatch, the family image (pinned tag from the shared registry), the pinned
dataset fetch, the lm-eval harness path inside the image, and the
``results.json`` artifact. The gold answers score 1.0.

The step comes from the Space ``make space`` renders (``space_uri:
../../space``), i.e. from your local source.

Run it with::

    make -C steps/eval/lmeval/skypilot test TESTS=test/slurm-bluevela

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
class TestSkypilotSlurmBlueVelaEvalLmeval(AbstractYamlBuildRunnerTest):
    """lmeval's gold mode runs mmlu-prox-lite end to end on BlueVela SLURM."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)

    @pytest.mark.timeout(2100)  # above buildtest.yaml's timeout_minutes (30)
    def test_runner(self):
        """Run the happy-path build with room for the buildtest timeout."""
        super().test_runner()
