#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Integration test for the command step on Skypilot/slurm with HF URI input/output.

Validates the full pipeline:
  HF URI input  ->  command step on Skypilot/slurm  ->  HF URI output

Requires a running Docker SLURM cluster (see scripts/slurm/setup-slurm.sh).
Auto-skips when the cluster is not reachable via SSH.

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


# Real-infra build test (launches a SLURM job via Skypilot) — only run in the
# extended suite (make extended-tests), not the fast quick-tests suite.
@extended_testing_only
@pytest.mark.skipif(
    not _slurm_cluster_reachable(),
    reason="Docker SLURM cluster not reachable (run: make slurm-setup)",
)
class TestSkypilotSlurm1Step(AbstractYamlBuildRunnerTest):
    """HF input -> command step on slurm via Skypilot -> HF output."""

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "1step"
