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
    return [EfsProvider(m.mount_point, m.efs) for m in mounts]
