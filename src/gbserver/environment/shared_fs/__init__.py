# Copyright Granite.Build Authors
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

"""Shared-filesystem provider layer for environments without a networked FS (EFS)."""

from typing import TYPE_CHECKING, List, Optional

from gbserver.environment.shared_fs.base import (
    ProvisionedResources,
    SharedFilesystemProvider,
    resolve_local_scratch,
    resolve_shared_workdir,
    resolve_workdir_mount,
)
from gbserver.environment.shared_fs.config import parse_shared_filesystems
from gbserver.environment.shared_fs.efs import EfsProvider

if TYPE_CHECKING:
    from gbserver.types.environmentconfig import EnvironmentConfig

__all__ = [
    "SharedFilesystemProvider",
    "ProvisionedResources",
    "resolve_shared_workdir",
    "resolve_local_scratch",
    "resolve_workdir_mount",
    "build_providers",
]


def build_providers(
    config: Optional["EnvironmentConfig"],
) -> List[SharedFilesystemProvider]:
    """One provider per declared shared_filesystem mount (declared order)."""
    if config is None:
        return []
    mounts = parse_shared_filesystems((config.config or {}).get("shared_filesystem"))
    providers: List[SharedFilesystemProvider] = []
    for m in mounts:
        # SharedFilesystemConfig._check guarantees a non-None efs block for the
        # (only) "efs" provider — it raises during parsing otherwise — so m.efs
        # is never None here; assert narrows Optional[EfsConfig] for the ctor.
        assert m.efs is not None
        providers.append(EfsProvider(m.mount_point, m.efs))
    return providers
