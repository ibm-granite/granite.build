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

"""Integration test for s3:// inputs/outputs on Skypilot/slurm.

Validates s3:// input -> s3pull -> command step -> s3push -> s3:// output on the
local Docker SLURM cluster (``make slurm-setup``).

It always runs against the local SeaweedFS (``make s3-setup``) and checks the
pushed object round-trips the seeded input. The store is local infrastructure
like the cluster, so unlike hf:// there is nothing to mock; GBTEST_MOCK_COS is
pinned off for the run. It skips when the cluster or the store is unavailable.
"""

from pathlib import Path

import pytest
from integration.environment.test_skypilot_slurm_e2e import (
    _slurm_cluster_reachable,
)
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only
from libgbtest.s3 import S3BuildTestMixin, local_s3_unavailable_reason

pytestmark = pytest.mark.skypilot_integration

_S3_UNAVAILABLE = local_s3_unavailable_reason()


@extended_testing_only
@pytest.mark.skipif(
    not _slurm_cluster_reachable(),
    reason="Docker SLURM cluster not reachable (run: make slurm-setup)",
)
@pytest.mark.skipif(_S3_UNAVAILABLE is not None, reason=_S3_UNAVAILABLE or "")
class TestSkypilotSlurmLocalS3(S3BuildTestMixin, AbstractYamlBuildRunnerTest):
    """s3:// input -> command step on slurm via Skypilot -> s3:// output."""

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "s3"
