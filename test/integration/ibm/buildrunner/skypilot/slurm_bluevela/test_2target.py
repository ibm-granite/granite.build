# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Two bash-step targets on BlueVela SLURM (via Skypilot) with hf:// I/O.

`first` runs the generic `command` step against the sky-slurm-bluevela
environment (space://environments/skypilot/slurm/ibm-bluevela): it pulls an hf://
dataset input, writes a real output file on the allocated compute node, and
registers it as artifact `out1` — which is pushed to an hf:// dataset repo.
`second` binds `first.out1` as an input (so buildrunner hf-pulls first's output),
reads the bound path, and registers its own hf:// output `out2`. This exercises
cross-target output -> input binding over REAL HuggingFace pulls/pushes on the
SLURM bare-host path (no `command_config.image`, so no Pyxis SPANK plugin — the
SLURM equivalent of the enroot image path covered for LSF in test_1step_image.py).

Because the hf:// URIs drive live HuggingFace pulls/pushes (real files, real
uploads to ibm-research), this is @extended_testing_only and needs HF_TOKEN with
write access to the output repos.

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
class TestSkypilotBlueVelaSlurm2Target(AbstractYamlBuildRunnerTest):
    """Two hf:// command targets on BlueVela SLURM; target 2 binds target 1's output."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "2target"
