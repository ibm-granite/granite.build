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

"""SkyPilot-on-AWS shared_filesystem (EFS) cross-instance data flow (#378).

The AWS analog of the sibling ``skypilot/aws`` build tests (2target, 1step-image,
filemount): a two-step byoc producer -> consumer build in one target. SkyPilot
allocates a SEPARATE EC2 instance per step and AWS has no networked filesystem,
so step 2 can only read what step 1 wrote if the ``shared_filesystem`` EFS
provider mounts a shared ``$GB_BUILD_WORKDIR`` on both instances. Step 1 writes a
sentinel under ``$GB_BUILD_WORKDIR``; step 2, on a different instance, reads it
back and asserts its value (``set -eu`` fails the step, and thus the build,
otherwise). Reaching SUCCESS proves EFS carried step 1's bytes across instances.

Two fixtures exercise the two mount paths:
  * :class:`TestSkypilotAwsSharedFsBare` — byoc ``image: ""`` runs on the bare
    EC2 instance; the mount happens on the host.
  * :class:`TestSkypilotAwsSharedFsContainerized` — byoc sets an image
    (``image_id: docker:<image>``) so the commands run INSIDE the container
    against the in-container NFS mount (SkyPilot's default SYS_ADMIN/--net=host/
    fuse) and the 1777/uid path.

``env://`` (env_local) I/O is a no-op, so the build drives the byoc steps
end-to-end without HF/S3 credentials — only AWS access plus the BYO EFS.

Like the sibling aws build tests this is intentionally NOT marked ``ibm``: it
needs AWS credentials + SkyPilot, not the IBM cloud secret bundle the ``ibm``
marker's ``check_cloud_config()`` gate enforces. It is gated on AWS credentials
being present, so it auto-skips in CI and on machines without AWS access.

It ADDITIONALLY needs a real, pre-provisioned BYO EFS (gbserver never creates or
destroys the filesystem). The fixture Space ships the documented PLACEHOLDER EFS
``file_system_id``, so the test self-skips until an operator points it at a real
one — this reads that committed ``environment.yaml`` and skips while it is still
the placeholder, so no EC2/EFS is provisioned against a bogus id.

Prerequisites to actually run (locally, in the extended suite):
  1. AWS credentials configured (env vars or ``~/.aws/credentials``).
  2. SkyPilot installed and ``sky check aws`` passing.
  3. A pre-provisioned BYO EFS (mount targets per worker AZ, an SG allowing NFS
     2049, root chmod 1777 — see docs/environments/skypilot-aws.md), with its
     ``file_system_id``/``region`` written into the fixture Space's
     ``environments/skypilot/aws-shared-fs/environment.yaml``.

Each fixture's build.yaml, buildtest.yaml, and the shared test Space live under
the directory returned by ``_get_yaml_spec_dir`` below.
"""

import os
from pathlib import Path

import pytest
import yaml
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

# The documented placeholder file_system_id the fixture Space ships (mirrors the
# commented example in configurations/assets/environments/skypilot/aws/
# environment.yaml). A real run replaces it with a validated BYO EFS id; until
# then the tests self-skip.
_PLACEHOLDER_EFS_FS_ID = "fs-0abc123"

# The committed fixture Space's aws-shared-fs environment (its shared_filesystem
# efs block); the build.yaml fixtures reference it as
# space://environments/skypilot/aws-shared-fs.
_ENV_YAML = (
    get_test_data_dir_for(__file__)
    / "shared-fs"
    / "space"
    / "environments"
    / "skypilot"
    / "aws-shared-fs"
    / "environment.yaml"
)


def _aws_credentials_available() -> bool:
    """True if AWS credentials look configured (env vars or ~/.aws/credentials)."""
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return True
    return (Path.home() / ".aws" / "credentials").is_file()


def _fixture_ships_placeholder_efs() -> bool:
    """True while the fixture environment.yaml still ships the placeholder EFS id.

    A cloud-free read of the committed shared_filesystem.efs block. Returns True
    (=> skip) whenever the file_system_id is the placeholder or cannot be read,
    so a run never provisions EC2/EFS against a bogus filesystem.
    """
    try:
        data = yaml.safe_load(_ENV_YAML.read_text(encoding="utf-8")) or {}
    except OSError:
        return True
    efs = (((data.get("config") or {}).get("shared_filesystem") or {}).get("efs")) or {}
    return efs.get("file_system_id", _PLACEHOLDER_EFS_FS_ID) == _PLACEHOLDER_EFS_FS_ID


# Real-infra build test (SkyPilot provisions EC2 instances against a BYO EFS) —
# only run in the extended suite (make extended-tests), and only once an operator
# has pointed the fixture at a real EFS. Shares the same xdist group as the other
# AWS tests so concurrent AWS provisions don't race on SkyPilot's local state.
pytestmark = [
    extended_testing_only,
    pytest.mark.xdist_group(name="buildtest_aws"),
    pytest.mark.skipif(
        not _aws_credentials_available(),
        reason="AWS credentials not configured (set AWS_ACCESS_KEY_ID/"
        "AWS_SECRET_ACCESS_KEY or provide ~/.aws/credentials); SkyPilot cannot "
        "provision an EC2 instance. Also requires `sky check aws` to pass.",
    ),
    pytest.mark.skipif(
        _fixture_ships_placeholder_efs(),
        reason=(
            "fixture Space still ships the placeholder EFS id "
            f"({_PLACEHOLDER_EFS_FS_ID}); set shared_filesystem.efs "
            f"file_system_id/region in {_ENV_YAML} to a validated BYO EFS to run "
            "(see docs/environments/skypilot-aws.md)."
        ),
    ),
]


class TestSkypilotAwsSharedFsBare(AbstractYamlBuildRunnerTest):
    """Bare EC2: step 1 writes under $GB_BUILD_WORKDIR, step 2 reads it back on a
    separate instance. SUCCESS proves EFS carried the bytes across instances."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "shared-fs" / "bare"


class TestSkypilotAwsSharedFsContainerized(AbstractYamlBuildRunnerTest):
    """Containerized: the same 2-step producer -> consumer flow, but the commands
    run inside the container against the in-container NFS mount + 1777/uid path."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "shared-fs" / "containerized"
