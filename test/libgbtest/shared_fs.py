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

"""Read EFS coordinates from a committed ``shared_filesystem`` environment.yaml.

The gated shared-fs real-infra tests (``test_shared_fs_efs_e2e.py`` and
``test_shared_fs_efs_concurrent_lock.py``) take their EFS coordinates from the
committed fixture ``environment.yaml`` (the ``shared_filesystem`` efs block),
NOT from ``GB_TEST_*`` env vars — the fixture environment IS the single source
of truth. This helper parses that block through the same
:class:`~gbserver.environment.shared_fs.config.SharedFilesystemConfig` /
``EfsConfig`` the server validates it with, and flags the committed placeholder
``file_system_id`` so a gated run self-skips until an operator points the
fixture at a real BYO EFS (rather than provisioning EC2 against a bogus id).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from gbserver.environment.shared_fs.config import SharedFilesystemConfig

# The documented placeholder file_system_id shipped in the committed fixture's
# environment.yaml (mirrors the commented example in
# configurations/assets/environments/skypilot/aws/environment.yaml). A gated
# real run replaces it with a validated BYO EFS id; until then the tests skip.
PLACEHOLDER_FILE_SYSTEM_ID = "fs-0abc123"


@dataclass(frozen=True)
class EfsCoords:
    """The EFS coordinates a shared-fs test needs, parsed from environment.yaml."""

    file_system_id: str
    region: Optional[str]
    dns_name: str
    is_placeholder: bool


def efs_coords_from_environment_yaml(path: Path) -> EfsCoords:
    """Parse the ``config.shared_filesystem.efs`` block from an environment.yaml.

    Args:
        path: Path to a skypilot/aws ``environment.yaml`` carrying a
            ``shared_filesystem`` efs block.

    Returns:
        The file_system_id, region, derived NFS DNS name, and whether the
        file_system_id is still the committed placeholder.

    Raises:
        ValueError: if the shared_filesystem block is missing or its efs block
            cannot derive a DNS name.
        pydantic.ValidationError: if the efs block is otherwise invalid.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    shared = (data.get("config") or {}).get("shared_filesystem")
    if shared is None:
        raise ValueError(f"{path}: config.shared_filesystem block is missing")
    cfg = SharedFilesystemConfig.model_validate(shared)
    dns = cfg.efs.derived_dns_name()
    if dns is None:
        raise ValueError(f"{path}: efs block cannot derive a DNS name")
    return EfsCoords(
        file_system_id=cfg.efs.file_system_id,
        region=cfg.efs.region,
        dns_name=dns,
        is_placeholder=cfg.efs.file_system_id == PLACEHOLDER_FILE_SYSTEM_ID,
    )
