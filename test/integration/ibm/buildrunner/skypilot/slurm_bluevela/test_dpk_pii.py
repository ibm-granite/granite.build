# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""The shared `dpk` step on BlueVela SLURM (via Skypilot), running a real transform.

Runs the `dpk` step's ``pii_redactor`` transform against the BlueVela SLURM
environment (space://environments/skypilot/slurm/bluevela), which reaches
BlueVela's SLURM login node (login1) over SSH and submits to the `gpu-mid`
partition (set via the environment's `zone`). This is the real-cluster
counterpart of the step's own local fixture at
``steps/dpk/skypilot/test/slurm-pii`` (Docker SLURM) and of its
``test/aws-pii`` (EC2): same step, same transform, same two derivations —

* module — ``transform: pii_redactor`` → ``python -m dpk_pii_redactor.runtime``
* pip    — → ``data-prep-toolkit-transforms[pii-redactor]==<dpk_version>``

so what this fixture adds is the CLUSTER, not the transform.

Both assets resolve from the one gb-test space the sibling fixtures already use:
the environment from gb-test itself, and ``space://steps/dpk`` from gb-test's own
base_uri (the assets repo, at ``steps/skypilot/dpk``). The step admits this env
because its ``environment_configs`` declares ``Skypilot`` with no ``subtypes:``
restriction.

**No container, so no Pyxis.** The sibling ``test_1step.py`` / ``test_2target.py``
set ``command_config.image`` and therefore exercise BlueVela's Pyxis SPANK plugin.
This target sets no ``dpk_config.dpk_image``, so DPK pip-installs onto the bare
launcher node. A failure here alongside a passing 1step points at the step or DPK,
not at containerization.

**Read-only HF.** The hf:// input is pulled (its own step on slurm — the pull is
not inline as it is on aws) and the output is a terminal ``env://``, so buildrunner
queues an hfpull and NO hfpush. Unlike 1step/2target, which push to an
ibm-research dataset repo, this needs only a READ-scoped HF_TOKEN.

**It is slow.** The ``[pii-redactor]`` extra resolves to 125 packages including
torch, flair, and presidio, and the transform downloads a flair NER model on first
use; the fixture allows 60 minutes, where the sibling bluevela fixtures use 20-30.

The fixture's build.yaml and buildtest.yaml live in the directory returned by
_get_yaml_spec_dir below.

REQUIRES the SlurmCommandRunner wrap-exemption fix in the SkyPilot fork this repo
pins (``git+https://github.com/cmadam/skypilot.git@gb-sky-v1-stable``). Green on
BlueVela SLURM 2026-09-28 against the patched fork; without it the test fails in
SkyPilot's file-mount stage, before the transform runs::

    sudo: a password is required
    srun: error: p1-r08-n1: task 0: Exited with exit code 1
    Failed to create symlinks. The target destination may already exist.

Why, because it applies to any bare-host step that ships files, not just dpk:

1. The dpk step ships its scripts via ``file_mounts: {src: src}``, a RELATIVE
   destination that gbserver's ``_remap_relative_dest`` (environment/skypilot.py)
   rewrites onto the shared ``${build_workdir}/src`` — an absolute ``/proj/...``
   path on BlueVela.
2. SkyPilot's ``_execute_file_mounts`` sudo-symlink-wraps every absolute
   destination not under ``~/``/``/tmp/`` unless the runner's
   ``get_unwrapped_mount_prefixes()`` exempts it. On SLURM every runner command is
   ``srun``'d to the compute node, so on the bare path that wrap's ``sudo mkdir``
   runs as ``granitebuild``, which has no passwordless sudo.
3. The fix gives ``SlurmCommandRunner`` the exemption ``LsfCommandRunner`` already
   had, rooted at the SLURM ``workdir`` (``cloud_config.slurm.cluster_configs.
   bluevela.workdir`` in the gb-test env, an ancestor of ``shared_workdir``).

A containerized step never hit this — its file_mounts run as root inside the
enroot container, where SkyPilot aliases ``sudo`` away — which is why the sibling
image-mode fixtures, and ``test_dpk_tok_image.py``, passed without the fix.

**Memory is requested explicitly.** BlueVela enforces per-job memory, but gbserver
drops ``compute_config.total_memory_per_node`` on slurm/lsf, so the build sets
``launcher_config.resources.memory``. Without it the job got the partition default
(2 CPUs / 2G) and was OOM-killed loading ``flair/ner-english-large``.
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
# ControlPersist window. Intentionally NOT an autouse fixture; see the longer
# note in test_1step.py for why forcing it on every run is unsafe under xdist.


@extended_testing_only
# Shares the bluevela xdist group with the sibling fixtures so the BlueVela tests
# serialize rather than contending for the same login node / SSH ControlMaster
# socket.
@pytest.mark.xdist_group(name="buildtest_bv")
# For this test to run in IBM SPS build tests, it needs to
# 1) have an environments/skypilot/slurm/bluevela/environment.yaml referencing
#    the BV_SSH_PRIVATE_KEY secret (IdentityKey: BV_SSH_PRIVATE_KEY)
# 2) Change the test to use the public IBM space, which uses the ibm secret manager
# The fixture resolves the `bluevela` env from the remote gb-test space
# (buildtest.yaml space_uri: git+ssh://.../gb-test.git@gbspace-config).
@pytest.mark.skipif(
    os.environ.get("RUNNING_IN_CICD", "False").lower() == "true",
    reason="Skip in SPS CI/CD until we have environments/skypilot/slurm/bluevela/environment.yaml with key reference in gb-test and other space repos",
)
class TestDPKBlueVelaSlurm(AbstractYamlBuildRunnerTest):
    """dpk step running pii_redactor on BlueVela SLURM (gpu-mid), bare launcher node."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "dpk-pii"

    @pytest.mark.timeout(
        3900
    )  # above buildtest.yaml's timeout_minutes (60); pyproject's global 1200s would kill it first
    def test_runner(self):
        super().test_runner()
