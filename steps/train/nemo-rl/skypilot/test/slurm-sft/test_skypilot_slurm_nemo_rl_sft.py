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

"""NeMo-RL SFT end to end on skypilot/slurm against IBM BlueVela (real GPU node).

The SLURM counterpart of ``test/aws-sft``: the same short SFT run on a tiny base
model, on BlueVela's gpu-mid partition via
``space://environments/skypilot/slurm/ibm-bluevela``. Reaching SUCCESS proves the
step's SLURM path: the base model is staged by a separate hfpull step onto the
shared /proj filesystem, the NeMo-RL image runs as a Pyxis/enroot container from a
prebuilt ``.sqsh``, ``examples/run_sft.py`` trains, ``convert_dcp_to_hf.py``
produces a non-empty HF dir, and the ``model`` artifact is registered from the
step's marker.

**Real cluster — never runs by accident.** It is extended-suite only, marked
``ibm`` (it needs IBM infrastructure), and skips unless the BlueVela key for the
``granitebuild`` functional ID is present at the path the environment names. It
also skips in SPS CI, which has no key on the host.

Fixtures (build.yaml/buildtest.yaml) live in the ``test-data/`` dir mirroring this
file, resolved by the repo's ``test/`` <-> ``test-data/`` helper so the pairing
holds in both test modes (see steps/README.md).
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

# The key the ibm-bluevela SLURM environment reads (cluster_ssh_configs IdentityFile).
_BLUEVELA_KEY = Path("~/.ssh/ibm-bluevela.key").expanduser()


@extended_testing_only
@pytest.mark.skipif(
    not _BLUEVELA_KEY.is_file(),
    reason=f"BlueVela granitebuild SSH key not found at {_BLUEVELA_KEY}",
)
@pytest.mark.skipif(
    os.environ.get("RUNNING_IN_CICD", "False").lower() == "true",
    reason="No BlueVela SSH key on SPS CI/CD hosts",
)
class TestSkypilotSlurmNemoRlSft(AbstractYamlBuildRunnerTest):
    """nemo-rl step: single-node SFT end to end on BlueVela SLURM (gpu-mid)."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)
