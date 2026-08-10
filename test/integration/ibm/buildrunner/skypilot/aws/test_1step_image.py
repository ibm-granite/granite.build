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

"""Containerized command-step target on AWS EC2 (via SkyPilot).

Runs a single ``command`` step whose ``command_config.image`` is set, so the
command executes INSIDE a Docker container on the provisioned EC2 instance. This
is the AWS analog of the sibling
``test/integration/ibm/buildrunner/skypilot/bluevela/test_1step_image.py``:
bluevela runs the image through LSF's enroot, whereas SkyPilot on AWS renders
``image_id`` to ``docker:<image>`` and runs it with Docker on the VM.

``env://`` (env_local) I/O is a no-op, so the test drives the command step
end-to-end without HF/S3 credentials — the simplest "does AWS + SkyPilot work"
smoke test.

Unlike the bluevela sibling, this test is intentionally NOT marked ``ibm``: it
needs AWS credentials + SkyPilot, not the IBM cloud secret bundle that the
``ibm`` marker's ``check_cloud_config()`` gate enforces (Lakehouse token, IBM
Cloud API key, GitHub token, image tags). It is gated instead on AWS credentials
being present, so it auto-skips in CI and on machines without AWS access.

Prerequisites to actually run (locally, in the extended suite):
  1. AWS credentials configured (env vars or ``~/.aws/credentials``).
  2. SkyPilot installed and ``sky check aws`` passing.
  3. quay.io reachable from the launched instance (public image, no auth).

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


# Real-infra build test (SkyPilot provisions an EC2 instance) — only run in the
# extended suite (make extended-tests), not the fast quick-tests suite. Its own
# xdist group serializes concurrent AWS provisions so they don't race on
# SkyPilot's local state.
@extended_testing_only
@pytest.mark.xdist_group(name="buildtest_aws")
@pytest.mark.skipif(
    not _aws_credentials_available(),
    reason="AWS credentials not configured (set AWS_ACCESS_KEY_ID/"
    "AWS_SECRET_ACCESS_KEY or provide ~/.aws/credentials); SkyPilot cannot "
    "provision an EC2 instance. Also requires `sky check aws` to pass.",
)
class TestSkypilotAws1StepImage(AbstractYamlBuildRunnerTest):
    """Single command step running inside a Docker image on AWS EC2 via SkyPilot."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "1step-image"
