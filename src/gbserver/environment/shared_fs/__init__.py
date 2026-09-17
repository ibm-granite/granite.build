"""Shared-filesystem provider layer for environments without a networked FS (EFS)."""

from typing import Optional

from gbserver.environment.shared_fs.base import (
    SharedFilesystemProvider,
    resolve_shared_workdir,
)
from gbserver.environment.shared_fs.config import SharedFilesystemConfig
from gbserver.environment.shared_fs.efs import EfsProvider

__all__ = ["SharedFilesystemProvider", "resolve_shared_workdir", "build_provider"]


def build_provider(config) -> Optional[SharedFilesystemProvider]:
    """Construct the provider for an EnvironmentConfig, or None when there is no
    ``shared_filesystem`` block."""
    if config is None:
        return None
    sf_raw = (config.config or {}).get("shared_filesystem")
    if not sf_raw:
        return None
    sf = SharedFilesystemConfig.model_validate(sf_raw)
    return EfsProvider(sf.mount_point, sf.efs)
