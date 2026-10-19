# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Containerized command-step target on BlueVela LSF (via Skypilot).

Runs a single `command` step whose `command_config.image` is set, so the command
executes INSIDE a container. On LSF this goes through enroot (the bluevela
environment's cloud_config.lsf.cluster_configs.bluevela.enroot section), which is
the LSF equivalent of slurm's Pyxis. The non-containerized command path is
covered by test_2target.py; this test is the only coverage for the enroot image
path.

env:// (env_local) I/O is a shared-FS no-op, so the test drives the command step
end-to-end without HF credentials or real pushes.

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

pytestmark = pytest.mark.ibm


@extended_testing_only
@pytest.mark.xdist_group(name="buildtest_bv")
# No separate container-support probe gate (unlike slurm's
# _slurm_cluster_supports_containers): enroot is declared enabled: true on the
# bluevela environment, so the cluster is expected to support containers.
#
# For this test to run in IBM SPS build tests, it needs to
# 1) have a environments/skypilot/lsf/bluevela/environment.yaml referencing BV_SSH_PRIVATE_KEY secret
#     IdentityKey: BV_SSH_PRIVATE_KEY
# 2) Change the test to use the public IBM space, which uses the ibm secret manager
# Without these changes, the test uses the local space and expects a local ~/.ssh/ibm-bluevela.key
# This allows it to be run locally.
@pytest.mark.skipif(
    os.environ.get("RUNNING_IN_CICD", "False").lower() == "true",
    reason="Skip in SPS CI/CD until we have environments/skypilot/lsf/bluevela/environment.yaml with key reference in gb-test and other space repos",
)
class TestSkypilotBlueVela1StepImage(AbstractYamlBuildRunnerTest):
    """Single command step running inside a container (enroot) on BlueVela LSF."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "1step-image"
