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

"""openinstruct-sft end to end on skypilot/aws (real EC2, one A10G).

SUCCESS proves the aws path: the quay.io image pulls, the inline `hf download`
stages SmolLM2-135M, finetune.py trains 3 steps under DeepSpeed ZeRO-3 on HF-id
data, and the `model` artifact plus the open_instruct_commit metadata are
recorded from the step's markers.

Real EC2 (costs money): extended AND skipped unless AWS credentials are in the
environment. The image must already be published (README.md, "Publishing the
image"). Fixtures live in the test-data/ dir mirroring this file
(get_test_data_dir_for).
"""

from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

from gbserver.environment.skypilot import aws_credentials_present

pytestmark = pytest.mark.skypilot_integration


@extended_testing_only
@pytest.mark.skipif(
    not aws_credentials_present(),
    reason="AWS credentials not in environment (set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE)",
)
class TestSkypilotAwsTrainOpeninstructSft(AbstractYamlBuildRunnerTest):
    """openinstruct-sft step: single-GPU SFT end to end on aws (EC2)."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)

    @pytest.mark.timeout(5700)  # above buildtest.yaml's timeout_minutes (90)
    def test_runner(self):
        """Run the happy-path build with room for the buildtest timeout."""
        super().test_runner()
