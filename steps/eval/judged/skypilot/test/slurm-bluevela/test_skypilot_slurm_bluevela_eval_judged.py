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

"""Integration test: the judged eval step runs end to end on BlueVela SLURM (Skypilot).

Runs gdpval on the shared ``space://environments/skypilot/slurm/bluevela``
with granite-4.2-3b served by vLLM inside the job and ``judge_model=self``
-- the key-less smoke path that grades with the served model -- so no judge
endpoint and no ``EVAL_JUDGE_API_KEY`` are needed. Scoring needs the server
when the judge is the served model, so the run is phase all in one job on
the family's GPU image. (The split score phase on the CPU image is the
external-judge shape; a public fixture cannot name a site-specific judge
endpoint, which is why this test covers only the GPU image.)

The step comes from the Space ``make space`` renders (``space_uri:
../../space``), i.e. from your local source.

Run it with::

    make -C steps/eval/judged/skypilot test TESTS=test/slurm-bluevela

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
class TestSkypilotSlurmBlueVelaEvalJudged(AbstractYamlBuildRunnerTest):
    """The judged step runs gdpval (self-judged) end to end on BlueVela SLURM."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)

    @pytest.mark.timeout(3900)  # above buildtest.yaml's timeout_minutes (60)
    def test_runner(self):
        """Run the happy-path build with room for the buildtest timeout."""
        super().test_runner()
