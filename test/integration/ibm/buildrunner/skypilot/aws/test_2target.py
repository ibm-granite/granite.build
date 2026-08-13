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

"""Two command-step targets on AWS EC2 (via SkyPilot), with cross-target binding.

`first` runs the generic `command` step to emit an output path and register it as
artifact `out1`; `second` binds `first.out1` as an input, echoes the bound path,
and registers its own output `out2`. This exercises cross-target output -> input
binding over the env_local (env://) assetstore. It is the AWS analog of the
sibling ``test/integration/ibm/buildrunner/skypilot/bluevela/test_2target.py``:
bluevela runs on LSF, whereas SkyPilot on AWS provisions a small EC2 instance per
target (bare command, no container).

``env://`` (env_local) I/O is a no-op, so the test drives both command steps
end-to-end without HF/S3 credentials.

Like the sibling aws/1step-image test, this is intentionally NOT marked ``ibm``:
it needs AWS credentials + SkyPilot, not the IBM cloud secret bundle that the
``ibm`` marker's ``check_cloud_config()`` gate enforces. It is gated instead on
AWS credentials being present, so it auto-skips in CI and on machines without AWS
access.

Prerequisites to actually run (locally, in the extended suite):
  1. AWS credentials configured (env vars or ``~/.aws/credentials``).
  2. SkyPilot installed and ``sky check aws`` passing.

The fixture's build.yaml and buildtest.yaml live in the directory returned by
``_get_yaml_spec_dir`` below.
"""

import os
from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only


def _aws_credentials_available() -> bool:
    """True if AWS credentials look configured (env vars or ~/.aws/credentials)."""
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return True
    return (Path.home() / ".aws" / "credentials").is_file()


# Real-infra build test (SkyPilot provisions EC2 instances) — only run in the
# extended suite (make extended-tests), not the fast quick-tests suite. Shares the
# same xdist group as the sibling AWS test so concurrent AWS provisions don't race
# on SkyPilot's local state.
@extended_testing_only
@pytest.mark.xdist_group(name="buildtest_aws")
@pytest.mark.skipif(
    not _aws_credentials_available(),
    reason="AWS credentials not configured (set AWS_ACCESS_KEY_ID/"
    "AWS_SECRET_ACCESS_KEY or provide ~/.aws/credentials); SkyPilot cannot "
    "provision an EC2 instance. Also requires `sky check aws` to pass.",
)
class TestSkypilotAws2Target(AbstractYamlBuildRunnerTest):
    """Two command-step targets on AWS EC2 via SkyPilot; target 2 binds target 1's output."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "2target"
