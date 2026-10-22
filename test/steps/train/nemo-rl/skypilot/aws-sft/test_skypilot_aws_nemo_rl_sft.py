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

**Real EC2.** It is marked extended AND skips unless AWS credentials are present
(:func:`gbserver.environment.skypilot.aws_credentials_present`) and a Hub-writable
``HF_TOKEN`` will reach the build (:func:`hf_push_skip_reason`). The extended
marker is not deselected by a plain ``pytest test/`` or ``make test``, so in a
shell with ``AWS_PROFILE`` and a write token exported this DOES provision EC2;
run unit tests with ``-m "not extended"``. ClearML is left in ``auto``: with no
CLEARML_* creds it cleanly disables, exercising the passthrough path.

Fixtures (build.yaml/buildtest.yaml) live in the ``test-data/`` dir mirroring this
file, resolved by the repo's ``test/`` <-> ``test-data/`` helper so the pairing
holds in both test modes (see steps/README.md).
"""

import os
from collections.abc import Mapping
from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

from gbserver.environment.skypilot import aws_credentials_present

pytestmark = pytest.mark.skypilot_integration


def hf_push_skip_reason(env: Mapping[str, str]) -> str:
    """Why the Hub push in this fixture cannot succeed, or "" if it can.

    The fixture pushes to ``granite-build-test/nemo-rl-sft-smoke``, which needs a
    write-capable ``HF_TOKEN``. The test setup (test/conftest.py) replaces
    ``HF_TOKEN`` with the secret manager's read-only token unless
    ``GBTEST_SPS_ENABLE_ENV_VAR_OVERRIDE=true``, so both must be set. Write
    access itself is not checked here (that needs the network at collection
    time); a token without it still fails at the push.

    :param env: The environment to inspect, normally ``os.environ``.
    :returns: A skip reason, or an empty string when the push can proceed.
    """
    if not env.get("HF_TOKEN"):
        return "no HF_TOKEN: the fixture pushes to the Hub and needs a write token"
    if env.get("GBTEST_SPS_ENABLE_ENV_VAR_OVERRIDE", "").lower() != "true":
        return (
            "HF_TOKEN would be replaced by the read-only test token: set "
            "GBTEST_SPS_ENABLE_ENV_VAR_OVERRIDE=true to keep a write token"
        )
    return ""


@extended_testing_only
@pytest.mark.skipif(
    not aws_credentials_present(),
    reason=(
        "AWS credentials not in environment "
        "(set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE)"
    ),
)
@pytest.mark.skipif(
    bool(hf_push_skip_reason(os.environ)),
    reason=hf_push_skip_reason(os.environ) or "HF push available",
)
class TestSkypilotAwsNemoRlSft(AbstractYamlBuildRunnerTest):
    """nemo-rl step: single-node SFT end to end on aws (EC2)."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)
