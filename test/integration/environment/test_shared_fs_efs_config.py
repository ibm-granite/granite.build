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

"""Unit guard (no AWS) for the committed shared-fs EFS ``environment.yaml``.

The two gated real-infra tests (``test_shared_fs_efs_e2e.py`` and
``test_shared_fs_efs_concurrent_lock.py``) no longer read the EFS coordinates
from ``GB_TEST_*`` env vars — they read them from the committed
``shared_fs_efs`` fixture's ``environment.yaml`` via
:func:`libgbtest.shared_fs.efs_coords_from_environment_yaml`. This test runs in
the normal suite (no gate, no cloud) so that fixture + helper stay valid: the
efs block must parse, derive the NFS DNS name, and ship the documented
placeholder ``file_system_id`` (so a gated run skips until an operator points it
at a real BYO EFS).
"""

from pathlib import Path

from libgbtest.shared_fs import (
    PLACEHOLDER_FILE_SYSTEM_ID,
    efs_coords_from_environment_yaml,
)

# The single committed source of truth for the shared-fs test EFS coordinates:
# the fixture Space's aws-shared-fs environment. Both gated tests resolve their
# coordinates from this same file.
_ENV_YAML = (
    Path(__file__).parent
    / "shared_fs_efs"
    / "space"
    / "environments"
    / "skypilot"
    / "aws-shared-fs"
    / "environment.yaml"
)


def test_efs_coords_parse_and_derive_dns():
    """The committed environment.yaml's efs block parses and derives its DNS."""
    coords = efs_coords_from_environment_yaml(_ENV_YAML)
    assert coords.file_system_id
    assert coords.region
    assert (
        coords.dns_name == f"{coords.file_system_id}.efs.{coords.region}.amazonaws.com"
    )


def test_committed_fixture_ships_placeholder_fs_id():
    """The committed fixture ships the placeholder id, so gated runs self-skip."""
    coords = efs_coords_from_environment_yaml(_ENV_YAML)
    assert coords.file_system_id == PLACEHOLDER_FILE_SYSTEM_ID
    assert coords.is_placeholder is True
