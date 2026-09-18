"""Typed, validated schema for the environment.yaml `shared_filesystem` block (EFS)."""

import os
from typing import Literal, Optional

from pydantic import model_validator

from gbserver.types.config import Config


class EfsConfig(Config):
    """BYO, pre-provisioned EFS filesystem reference.

    Exactly one of ``file_system_id`` or ``dns_name`` is required. When only
    ``file_system_id`` is given, ``region`` is required so the container-safe
    ``nfs4`` fallback DNS name can be derived. ``cleanup_zone`` optionally pins
    the teardown VM to an AZ that has a mount target; if unset, the teardown VM
    lands in the cloud's default AZ, which may lack a mount target and fail the
    cleanup (surfaced as an orphan WARNING) -- provision a mount target in every
    worker AZ, or set ``cleanup_zone``.
    """

    file_system_id: Optional[str] = None
    dns_name: Optional[str] = None
    region: Optional[str] = None
    tls: bool = True
    cleanup_zone: Optional[str] = None

    @model_validator(mode="after")
    def _require_target(self) -> "EfsConfig":
        if not self.file_system_id and not self.dns_name:
            raise ValueError("efs: one of file_system_id or dns_name is required")
        if self.file_system_id and not self.dns_name and not self.region:
            raise ValueError(
                "efs: 'region' is required with 'file_system_id' (to derive the "
                "nfs4-fallback DNS name); or set 'dns_name' explicitly"
            )
        if (
            self.cleanup_zone
            and self.region
            and not self.cleanup_zone.startswith(self.region)
        ):
            raise ValueError(
                f"efs: cleanup_zone {self.cleanup_zone!r} is not in region "
                f"{self.region!r} (an AWS AZ name is its region plus a letter, "
                "e.g. us-east-1a)"
            )
        return self

    def derived_dns_name(self) -> Optional[str]:
        if self.dns_name:
            return self.dns_name
        if self.file_system_id and self.region:
            return f"{self.file_system_id}.efs.{self.region}.amazonaws.com"
        return None


class SharedFilesystemConfig(Config):
    """The `shared_filesystem` block. Produces the `shared_workdir` root."""

    provider: Literal["efs"]
    mount_point: str
    efs: Optional[EfsConfig] = None

    @model_validator(mode="after")
    def _check(self) -> "SharedFilesystemConfig":
        if not os.path.isabs(self.mount_point):
            raise ValueError(
                f"shared_filesystem.mount_point must be absolute, got {self.mount_point!r}"
            )
        if self.efs is None:
            raise ValueError("provider 'efs' requires an 'efs' block")
        return self
