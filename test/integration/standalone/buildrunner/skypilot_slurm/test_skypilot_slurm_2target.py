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

"""Integration test for cross-target mem:// binding on SkyPilot SLURM.

Two `bash`-step targets on the SkyPilot SLURM backend: the first emits an
`GB_ARTIFACT_ID:… GB_ARTIFACT_STATE:…` marker (a mem:// output), the second
binds it and asserts the value arrived verbatim via the mem:// store — proving
the skypilot monitor now recognizes the STATE marker and mem:// bindings work on
the slurm backend. Mirrors the bash mem 2-target test. Requires a running Docker
SLURM cluster (see scripts/slurm/setup-slurm.sh / `make slurm-setup`).

The fixture's build.yaml and buildtest.yaml live in the directory returned by
_get_yaml_spec_dir below.
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

pytestmark = pytest.mark.skypilot_integration


# Real-infra build test (launches SLURM jobs via Skypilot) — extended suite only.
@extended_testing_only
@pytest.mark.skipif(
    not _slurm_cluster_reachable(),
    reason="Docker SLURM cluster not reachable (run: make slurm-setup)",
)
class TestSkypilotSlurm2Target(AbstractYamlBuildRunnerTest):
    """Cross-target mem:// output → input binding over two skypilot bash steps."""

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "2target"
