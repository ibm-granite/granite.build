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

"""hf-peft-sft end to end on IBM BlueVela SLURM (gpu-mid, two H200s per target).

SUCCESS proves the SLURM path for the two sharded modes: LoRA under FSDP on a
bound dataset and QLoRA under DeepSpeed ZeRO-3, ClearML offline. hfpull steps
stage the inputs onto shared /proj, the image runs under Pyxis from the shared
.sqsh cache, and each `adapter` artifact plus the method/peft_version metadata
are recorded.

IBM infrastructure through the local space, so marked `ibm`. Skips without the
BlueVela granitebuild SSH key, and in SPS CI/CD. The image comes from the
environment's shared .sqsh cache (slurm_image_cache, #560): a cold cache imports
it once in a CPU job before the GPU job starts.
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
class TestSkypilotSlurmBluevelaTrainHfPeftSft(AbstractYamlBuildRunnerTest):
    """hf-peft-sft step: sharded LoRA/QLoRA SFT end to end on BlueVela SLURM."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)

    @pytest.mark.timeout(18300)  # above buildtest.yaml's timeout_minutes (300)
    def test_runner(self):
        """Run the happy-path build with room for the buildtest timeout."""
        super().test_runner()
