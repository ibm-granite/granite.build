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

"""NeMo-RL SFT end to end on skypilot/aws (real EC2, single multi-GPU node).

The smallest SFT config (``sft.yaml``) with a tiny base model, proven on a real
EC2 GPU instance. Reaching SUCCESS proves the whole step path on aws: the NGC
release image pulls, the launcher's inline ``hf download`` stages the base model,
``examples/run_sft.py`` trains, ``convert_dcp_to_hf.py`` produces a non-empty HF
dir, and the ``model`` artifact is registered from the step's marker.

**Real EC2 — never runs by accident.** It is extended-suite only AND skips unless
AWS credentials are present (:func:`gbserver.environment.skypilot.aws_credentials_present`),
so no instance is provisioned without credentials explicitly exported. ClearML is
left in ``auto``: with no CLEARML_* creds in CI it cleanly disables, exercising the
passthrough path.

Fixtures (build.yaml/buildtest.yaml) live in the ``test-data/`` dir mirroring this
file, resolved by the repo's ``test/`` <-> ``test-data/`` helper so the pairing
holds in both test modes (see steps/README.md).
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
    reason=(
        "AWS credentials not in environment "
        "(set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE)"
    ),
)
class TestSkypilotAwsNemoRlSft(AbstractYamlBuildRunnerTest):
    """nemo-rl step: single-node SFT end to end on aws (EC2)."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)
