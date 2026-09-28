# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DPK tokenization on BlueVela SLURM in IMAGE MODE (containerized Pyxis path).

Runs the shared `dpk` step's ``tokenization2arrow`` transform against the BlueVela
SLURM environment (space://environments/skypilot/slurm/bluevela, gpu-mid partition)
with ``dpk_config.dpk_image`` set to a prebaked DPK image from IBM Container
Registry. This is the real-cluster counterpart of the step's own local fixture at
``steps/dpk/skypilot/test/slurm-tok`` — same transform, same args, same derivations;
what differs is the environment and the image.

**First image-mode coverage anywhere.** The dpk README records that ``dpk_image`` was
"exercised only by render tests", because image mode needs the Pyxis SPANK plugin that
``make slurm-setup``'s local Docker cluster does not install. BlueVela provides Pyxis
(proven by the sibling ``test_1step.py``), so this fixture closes that gap.

**Complementary to the pii fixture, not a replacement.** The two dpk fixtures here
cover the two SLURM execution paths, which fail differently:

* ``test_dpk_pii.py`` — bare path. Its ``file_mounts`` run on the host as
  ``granitebuild``, so it needs the SlurmCommandRunner wrap-exemption fix in the
  SkyPilot fork. Do not "fix" a regression there by containerizing it.
* this test — containerized path. Its ``file_mounts`` run as root inside the enroot
  container, where SkyPilot aliases ``sudo`` away, so it passes with or without that
  fix. Full analysis in ``docs/environments/skypilot-slurm-file-mounts-sudo.md``.

**What image mode changes.** ``dpk_version`` is ignored and the venv + ``uv pip
install`` are skipped entirely, so the image must already provide DPK; the build
therefore omits ``dpk_version`` and ``pip_index_url`` rather than setting values that
would not be used. ``file_mounts`` still happens (only the install is skipped), so
``./src`` is still staged and ``bash ./src/dpk_run.sh`` still drives the run — which
is precisely why this fixture also exercises the containerized file_mounts path.

**The image must provide PUBLIC DPK.** The step runs ``python -m
dpk_<transform>.runtime``, public DPK's layout. The ``dpk-1.1.8`` tag is built from the
Dockerfile beside this fixture's build.yaml; the earlier ``gb_v1`` was built from
IBM-internal DPK, whose ``tokenization2arrow_transform_python`` module name does not
match, and failed with "No module named 'dpk_tokenization2arrow'".

**Credentials.** The image is private, but granite.build supplies no registry
credentials on this path (SkyPilot's ``_docker_login_config`` is not wired into
``provision/slurm``, and the pull happens at provision time). The BlueVela nodes'
enroot already holds credentials for the ``cil15-shared-registry`` namespace —
confirmed by the green runs. A registry auth error would mean that changed
cluster-side. A ``ResourcesUnavailableError`` instead most likely means the image is
not Debian/apt-based (SkyPilot bootstraps its in-container SSH shim with
``apt-get``); confirm with ``sacct -j <job_id> --format=JobID,State,ExitCode,Reason``.

**Troubleshooting a ``Usage:: command not found`` / exit 127 before "Job started".**
That is SkyPilot's own container command, not this step: a SkyPilot API server
started from a venv with a pre-``5f18669`` fork generates ``$(which env ...)``, which an
exported host ``which`` function poisons inside the container. Check
``sky api info`` reports the fork commit this checkout pins; ``sky api stop`` and
restart it from this checkout's venv if not.

The input dataset is public, so unlike the sibling 1step/2target fixtures this needs
no HF_TOKEN at all.

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
# ControlPersist window. Intentionally NOT an autouse fixture; see the longer
# note in test_1step.py for why forcing it on every run is unsafe under xdist.


@extended_testing_only
# Shares the bluevela xdist group with the sibling fixtures so the BlueVela tests
# serialize rather than contending for the same login node / SSH ControlMaster socket.
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
class TestDPKImageBlueVelaSlurm(AbstractYamlBuildRunnerTest):
    """dpk step in image mode: tokenization2arrow from a prebaked ICR image on BlueVela SLURM."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "dpk-tok-image"

    @pytest.mark.timeout(
        3000
    )  # above buildtest.yaml's timeout_minutes (45); pyproject's global 1200s would kill it first
    def test_runner(self):
        super().test_runner()
