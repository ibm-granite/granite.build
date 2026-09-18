"""Shared-filesystem provider layer for environments without a networked FS (EFS)."""

from typing import TYPE_CHECKING, Optional

from gbserver.environment.shared_fs.base import (
    SharedFilesystemProvider,
    resolve_shared_workdir,
)
from gbserver.environment.shared_fs.config import SharedFilesystemConfig
from gbserver.environment.shared_fs.efs import EfsProvider

if TYPE_CHECKING:
    from gbserver.types.environmentconfig import EnvironmentConfig

__all__ = ["SharedFilesystemProvider", "resolve_shared_workdir", "build_provider"]


def build_provider(
    config: Optional["EnvironmentConfig"],
) -> Optional[SharedFilesystemProvider]:
    """Construct the provider for an EnvironmentConfig, or None when there is no
    ``shared_filesystem`` block."""
    if config is None:
        return None
    sf_raw = (config.config or {}).get("shared_filesystem")
    if not sf_raw:
        return None
    sf = SharedFilesystemConfig.model_validate(sf_raw)
    return EfsProvider(sf.mount_point, sf.efs)
