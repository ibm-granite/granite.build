#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Integration test for the Docker command step with HF URI input/output.

Validates the full pipeline:
  HF URI input  →  Docker container (command step)  →  HF URI output

The Docker daemon must be available.

The fixture's build.yaml and buildtest.yaml live in the directory returned by
_get_yaml_spec_dir below.
"""

import os
from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

pytestmark = pytest.mark.docker_required


# Real-infra build test (launches a local Docker container) — only run in the
# extended suite (make extended-tests), not the fast quick-tests suite.
@extended_testing_only
# TODO: We need to disable this skip when image pulling is supported
@pytest.mark.skipif(
    os.environ.get("RUNNING_IN_CICD", "False").lower() == "true",
    reason="Skip in CI/CD until we have automatic image pulling during the build",
)
class TestDockerCommandBuild(AbstractYamlBuildRunnerTest):
    """Integration test: HF input → Docker command step → HF output.

    Runs a real local Docker container with HF I/O mocked so no actual
    HuggingFace network calls are made.
    """

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "1step"
