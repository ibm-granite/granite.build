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

"""Gated real-infra E2E: shared_filesystem (EFS) producer -> consumer (#378/#393).

The OPT-IN, skip-GATED real-AWS proof for the ``shared_filesystem`` EFS provider.
It provisions real EC2 instances (via SkyPilot) and talks to a real,
pre-provisioned EFS filesystem, so it COSTS MONEY and MUST NEVER run in normal
CI. It self-skips unless every one of these holds:

* ``GB_RUN_SHARED_FS_E2E=1``   -- the explicit opt-in gate (spins up EC2/EFS; $).
* AWS credentials in the environment (``aws_credentials_present()``), matching the
  gate the shipped byoc/dpk skypilot-aws step tests use so no instance is ever
  provisioned by accident.
* the committed fixture ``environment.yaml`` points at a REAL BYO EFS (its efs
  ``file_system_id`` is no longer the shipped placeholder) -- see below.

**EFS coordinates live in the committed environment.yaml, not env vars.** Unlike
an earlier scaffold that materialized specs from ``GB_TEST_EFS_FS_ID`` /
``GB_TEST_EFS_REGION`` at runtime, this test uses a static, committed fixture
tree under ``shared_fs_efs/`` and drives it through the SAME build-submission
harness the shipped skypilot-aws step tests use
(:class:`libgbtest.buildrunner.buildtest.AbstractYamlBuildRunnerTest`). The EFS
coordinates are the ``shared_filesystem`` efs block of
``shared_fs_efs/space/environments/skypilot/aws-shared-fs/environment.yaml`` (the
single source of truth, also read by ``test_shared_fs_efs_concurrent_lock.py``).
To run for real, edit that file's efs ``file_system_id`` / ``region`` to your BYO
EFS (mount targets per worker AZ, an SG allowing NFS 2049, root chmod 1777 -- see
docs/environments/skypilot-aws.md); until then it ships the placeholder and the
tests self-skip so no EC2/EFS is provisioned against a bogus id.

**What it proves.** Two byoc steps in ONE target both start with CWD
``$GB_BUILD_WORKDIR`` (the per-target-run prefix ``${mount_point}/builds/<build_id>/
runs/<targetrun_id>/`` the ``shared_filesystem`` provider mounts on EFS and
``chmod 1777``s). SkyPilot allocates a SEPARATE EC2 instance per step, so:

* :class:`TestSharedFsEfsProducerConsumerBare` -- step 1 writes a sentinel under
  ``$GB_BUILD_WORKDIR``; step 2, on a different instance, reads it back. SUCCESS
  proves the EFS mount carried step 1's bytes across instances.
* :class:`TestSharedFsEfsProducerConsumerContainerized` -- the same 2-step flow
  but each step sets a container image (byoc renders ``image_id: docker:<image>``),
  exercising the in-container NFS mount and the 1777/uid path from inside the
  container.

The mount + cross-node coherence mechanism is ALSO validated on real AWS by the
sibling ``test_shared_fs_efs_concurrent_lock.py`` (2-node hfpull ``mkdir``-lock
over EFS); this module tracks the two-step build-submission path.

Run it (opt-in, real AWS) like the other skypilot-aws step tests -- ``-s`` is
required (else click sees a bad fd), after editing the fixture environment.yaml::

    GB_RUN_SHARED_FS_E2E=1 \\
    AWS_PROFILE=gb-skypilot \\
    PYTEST_ADDOPTS=-s \\
    .venv/bin/python -m pytest \\
      test/integration/environment/test_shared_fs_efs_e2e.py -q

Confirm the gate skips in a normal run (what CI does)::

    .venv/bin/python -m pytest \\
      test/integration/environment/test_shared_fs_efs_e2e.py -q
    # -> 2 skipped
"""

import os
from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import AbstractYamlBuildRunnerTest
from libgbtest.constants import extended_testing_only
from libgbtest.shared_fs import efs_coords_from_environment_yaml

from gbserver.environment.skypilot import aws_credentials_present

_GATE_ENV = "GB_RUN_SHARED_FS_E2E"

# The committed fixture tree: a Space carrying the aws-shared-fs environment (the
# efs block) and two spec dirs (bare/ and containerized/) each with a 2-step
# producer->consumer build.yaml + buildtest.yaml.
_SPEC_ROOT = Path(__file__).parent / "shared_fs_efs"
_ENV_YAML = (
    _SPEC_ROOT
    / "space"
    / "environments"
    / "skypilot"
    / "aws-shared-fs"
    / "environment.yaml"
)

# EFS coordinates from the committed environment.yaml (parsed at collection; a
# pure, cloud-free read guarded by test_shared_fs_efs_config.py). Skip when the
# fixture still ships the placeholder id so a gated run never provisions EC2/EFS
# against a bogus filesystem.
_EFS = efs_coords_from_environment_yaml(_ENV_YAML)

pytestmark = [
    pytest.mark.skypilot_integration,
    # Real EC2/EFS: extended suite only, and never by accident.
    extended_testing_only,
    pytest.mark.skipif(
        os.environ.get(_GATE_ENV) != "1",
        reason=(
            f"real-infra E2E: set {_GATE_ENV}=1 (spins up EC2/EFS; costs $). "
            "Also requires AWS creds and a real BYO EFS in the fixture environment.yaml."
        ),
    ),
    pytest.mark.skipif(
        not aws_credentials_present(),
        reason=(
            "AWS credentials not in environment "
            "(set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE)."
        ),
    ),
    pytest.mark.skipif(
        _EFS.is_placeholder,
        reason=(
            "shared-fs E2E opted in but the fixture still ships the placeholder EFS id "
            f"({_EFS.file_system_id}). Edit {_ENV_YAML} shared_filesystem.efs with your "
            "validated BYO EFS file_system_id/region to run for real."
        ),
    ),
]


class TestSharedFsEfsProducerConsumerBare(AbstractYamlBuildRunnerTest):
    """Bare EC2 (byoc ``image: ""``): step 1 writes a sentinel under
    ``$GB_BUILD_WORKDIR``, step 2 on a SEPARATE instance reads it back. SUCCESS
    proves EFS carried the bytes across instances.
    """

    def _get_yaml_spec_dir(self) -> Path:
        return _SPEC_ROOT / "bare"


class TestSharedFsEfsProducerConsumerContainerized(AbstractYamlBuildRunnerTest):
    """Containerized (byoc ``image: buildpack-deps:bookworm-scm`` ->
    ``image_id: docker:...``): the same 2-step flow, but the commands run INSIDE
    the container against the in-container NFS mount, exercising the
    SYS_ADMIN/--net=host/fuse + 1777/uid path. The image is Debian-based with git
    (byoc clones the repo inside the container) and apt (the mount prologue installs
    nfs-common).
    """

    def _get_yaml_spec_dir(self) -> Path:
        return _SPEC_ROOT / "containerized"
