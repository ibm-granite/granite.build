"""Typed, validated schema for the environment.yaml `shared_filesystem` block (EFS)."""

import os
from typing import List, Literal, Optional

from pydantic import model_validator

from gbserver.types.config import Config


class EfsConfig(Config):
    """BYO, pre-provisioned EFS filesystem reference.

    Exactly one of ``file_system_id`` or ``dns_name`` is required. When only
    ``file_system_id`` is given, ``region`` is required so the container-safe
    ``nfs4`` fallback DNS name can be derived. ``cleanup_zone`` (BYO only)
    optionally pins the teardown VM to an AZ that has a mount target; if unset,
    the teardown VM lands in the cloud's default AZ, which may lack a mount target
    and fail the cleanup (surfaced as an orphan WARNING) -- provision a mount
    target in every worker AZ, or set ``cleanup_zone``. It is rejected for
    ``provision: ephemeral`` (that teardown deletes via boto3, no cleanup VM).
    """

    provision: Literal["byo", "ephemeral"] = "byo"
    file_system_id: Optional[str] = None
    dns_name: Optional[str] = None
    region: Optional[str] = None
    tls: bool = True
    cleanup_zone: Optional[str] = None
    vpc_id: Optional[str] = None
    subnets: Optional[List[str]] = None
    security_group_id: Optional[str] = None

    @model_validator(mode="after")
    def _require_target(self) -> "EfsConfig":
        if self.provision == "ephemeral":
            if self.file_system_id or self.dns_name:
                raise ValueError(
                    "efs: provision 'ephemeral' must not set file_system_id/"
                    "dns_name (the filesystem is created at runtime)"
                )
            if not self.region:
                raise ValueError("efs: provision 'ephemeral' requires 'region'")
            if self.cleanup_zone:
                # Ephemeral teardown deletes the filesystem via boto3 and never
                # launches the throwaway cleanup VM that consumes cleanup_zone, so
                # it would be a silent no-op. Reject it rather than mislead.
                raise ValueError(
                    "efs: cleanup_zone is not used for provision 'ephemeral' "
                    "(teardown deletes via boto3, no cleanup VM is launched); "
                    "remove it"
                )
        else:  # byo
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
    """The `shared_filesystem` block: defines the mount only (an EFS mounted at
    `mount_point`). The workdir location is the separate, explicit `shared_workdir`
    (validated by EnvironmentConfig to be under `mount_point`)."""

    provider: Literal["efs"]
    mount_point: str
    efs: Optional[EfsConfig] = None
    local_scratch: Optional[str] = None
    """Instance-local scratch dir exported as ``GB_LOCAL_SCRATCH`` (defaults to
    ``/tmp/gb-scratch`` when unset). Must be absolute. Point it at the image's
    instance-store NVMe mount (e.g. ``/opt/dlami/nvme/...``) for true local scratch;
    ``/tmp`` is the EBS root volume on stock AWS/DLAMI images."""

    @model_validator(mode="after")
    def _check(self) -> "SharedFilesystemConfig":
        # Normalize a trailing slash so the chmod-walk's mount-root sentinel
        # matches (else the walk climbs past the root to /).
        self.mount_point = self.mount_point.rstrip("/") or "/"
        if not os.path.isabs(self.mount_point):
            raise ValueError(
                f"shared_filesystem.mount_point must be absolute, got {self.mount_point!r}"
            )
        if self.efs is None:
            raise ValueError("provider 'efs' requires an 'efs' block")
        if self.local_scratch is not None and not os.path.isabs(self.local_scratch):
            raise ValueError(
                "shared_filesystem.local_scratch must be absolute, got "
                f"{self.local_scratch!r}"
            )
        return self


def parse_shared_filesystems(sf_raw) -> "List[SharedFilesystemConfig]":
    """Parse the environment `shared_filesystem` value into a validated list.

    A lone object is coerced to a 1-element list (back-compat with the single
    mount form). Cross-mount rules: mount_points must be unique and non-nested;
    at most one mount may set local_scratch (it is instance-local, not per-FS).
    """
    if not sf_raw:
        return []
    items = sf_raw if isinstance(sf_raw, list) else [sf_raw]
    mounts = [SharedFilesystemConfig.model_validate(i) for i in items]
    mps = [m.mount_point for m in mounts]
    for i, a in enumerate(mps):
        for j, b in enumerate(mps):
            if i == j:
                continue
            if a == b:
                raise ValueError(f"shared_filesystem: mount_point {a!r} is not unique")
            if b.startswith(a.rstrip("/") + "/"):
                raise ValueError(
                    f"shared_filesystem: mount_point {b!r} is nested under {a!r}"
                )
    if sum(1 for m in mounts if m.local_scratch) > 1:
        raise ValueError(
            "shared_filesystem: at most one mount may set local_scratch "
            "(it is instance-local, not per-filesystem)"
        )
    return mounts
