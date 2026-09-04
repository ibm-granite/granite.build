# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Bare-host command-step target on BlueVela SLURM (via Skypilot).

Runs a single `command` step against the sky-slurm-bluevela environment
(space://environments/skypilot/slurm/ibm-bluevela), which reaches BlueVela's
SLURM login node (login1) over SSH and submits to the `gpu-mid` partition (set
via the environment's `zone`). The command runs DIRECTLY on the allocated
compute node — no `command_config.image` is set, so no Pyxis SPANK plugin is
required (the SLURM equivalent of the enroot image path covered separately for
LSF in test_1step_image.py).

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

# NOTE: to validate SSH auth against a freshly edited key/credential, set
# GBTEST_SKY_SSH_RESET=true in gbserver's environment before running — SkyPilot
# otherwise reuses a persisted SSH ControlMaster socket keyed on
# (host, port, user), not the key, masking an edited cluster_ssh_config for the
# ControlPersist window. This is intentionally NOT an autouse fixture: the
# socket clear globs the whole per-user root (/tmp/skypilot_ssh_<hash>/*/*), so
# forcing it on every run could yank the ControlMaster socket of another
# skypilot build (e.g. an AWS test) running in parallel on a different xdist
# group. Leave it unset for normal runs (the key is unchanged); set it manually
# only when testing a credential change.


@extended_testing_only
@pytest.mark.xdist_group(name="buildtest_bv")
# For this test to run in IBM SPS build tests, it needs to
# 1) have an environments/skypilot/slurm/ibm-bluevela/environment.yaml referencing
#    the BV_SSH_PRIVATE_KEY secret (IdentityKey: BV_SSH_PRIVATE_KEY)
# 2) Change the test to use the public IBM space, which uses the ibm secret manager
# Without these changes, the test uses the local space and expects a local
# ~/.ssh/ibm-bluevela.key, allowing it to be run locally.
@pytest.mark.skipif(
    os.environ.get("RUNNING_IN_CICD", "False").lower() == "true",
    reason="Skip in SPS CI/CD until we have environments/skypilot/slurm/ibm-bluevela/environment.yaml with key reference in gb-test and other space repos",
)
class TestSkypilotBlueVelaSlurm1Step(AbstractYamlBuildRunnerTest):
    """Single bare-host command step on BlueVela SLURM (gpu-mid partition)."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "1step"
