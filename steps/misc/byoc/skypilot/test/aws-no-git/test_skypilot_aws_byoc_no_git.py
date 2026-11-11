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

"""Integration test: byoc runs in an image without git on Skypilot/aws (#529).

The ``aws`` test's build, run in ``pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime``
instead of on the bare EC2 instance. That image has no git, and byoc clones the
repo before ``setup_command`` runs, so before #529 the step failed at setup with
``bash: git: command not found`` (build ``cb7a2e33``). byoc now installs git
before the clone (from conda-forge here, since the image has conda), and puts it
on ``PATH`` in ``run`` too. SUCCESS, plus the ``commit_hash`` that ``run`` records
with ``git rev-parse``, proves both.

Run it like the ``aws`` test, with AWS credentials exported::

    make -C steps/misc/byoc/skypilot test TESTS=test/aws-no-git

Real EC2, so it is in the extended suite only and skips unless AWS credentials
are in the environment (:func:`gbserver.environment.skypilot.aws_credentials_present`).
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
class TestSkypilotAwsByocNoGit(AbstractYamlBuildRunnerTest):
    """byoc installs git, clones and runs in an image without git, on aws (EC2)."""

    def _get_yaml_spec_dir(self) -> Path:
        # Mode 1: steps/misc/byoc/skypilot/test-data/aws-no-git/
        # Mode 2: test-data/steps/misc/byoc/skypilot/aws-no-git/
        return get_test_data_dir_for(__file__)
