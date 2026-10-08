"""SharedFilesystemProvider contract + the shared_workdir resolver."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from gbserver.types.environmentconfig import EnvironmentConfig


@dataclass(frozen=True)
class ProvisionedResources:
    """Runtime identity of an ephemeral EFS created for one target-run.
    Opaque to skypilot.py; threaded from provision() to deprovision()."""

    region: str
    file_system_id: str
    dns_name: str
    mount_target_ids: list = field(default_factory=list)
    subnet_ids: list = field(default_factory=list)
    security_group_id: Optional[str] = None
    created_sg: bool = False


class SharedFilesystemProvider(ABC):
    """Emits the shell that backs a shared_workdir root. skypilot.py owns all
    `sky` orchestration; a provider only returns shell strings."""

    def __init__(self, mount_point: str) -> None:
        self.mount_point = mount_point

    @abstractmethod
    def mount_prologue(self, dns_override: Optional[str] = None) -> str:
        """Idempotent shell mounting the FS at ``mount_point`` (host or, for a
        containerized step, inside the container). ``dns_override`` supplies the
        runtime DNS for ephemeral mounts; BYO mounts use config. Must echo a clear
        message and exit non-zero on failure so the caller's ``set -eu`` aborts."""

    async def provision(  # pylint: disable=unused-argument
        self, tags: dict, aws_profile: Optional[str]
    ) -> Optional[ProvisionedResources]:
        """Create backing infra for an ephemeral mount and return its runtime
        identity. Default None (BYO / non-provisioning backends)."""
        return None

    async def deprovision(  # pylint: disable=unused-argument
        self, provisioned: Optional[ProvisionedResources], aws_profile: Optional[str]
    ) -> None:
        """Destroy infra created by :meth:`provision`. Default no-op."""
        return None

    def cleanup_run_script(  # pylint: disable=unused-argument
        self, per_run_workdir: str
    ) -> Optional[str]:
        """Shell run on a throwaway VM to reap the per-run workdir: mount,
        ``rm -rf`` the per-run workdir, then best-effort ``rmdir`` the now-empty
        ``runs/`` and ``builds/<id>/`` parents.

        Returns ``None`` when the backend needs no VM-side cleanup — e.g. a
        stage-in/stage-out or object-store backend that reaps server-side via
        :meth:`cleanup` and has no filesystem to mount from a VM. Mount backends
        (EFS) override this; the default is ``None`` so a non-mount backend need
        not carry an empty implementation."""
        return None

    async def cleanup(self) -> None:
        """Server-side per-target-run cleanup for backends that don't reap via a
        throwaway VM (see :meth:`cleanup_run_script`). Default no-op; a mount
        backend leaves this unimplemented and uses ``cleanup_run_script``."""
        return None

    def cleanup_zone(self) -> Optional[str]:
        """AZ to pin the cleanup VM to (must have a mount target), or None."""
        return None

    def cleanup_timeout_s(self) -> Optional[float]:
        """Seconds teardown waits for the cleanup VM's reap job to finish, or
        None for the caller's default."""
        return None

    def transit_encryption_note(self) -> Optional[str]:
        """A one-line note logged server-side (gbserver) at launch about transit
        encryption — e.g. that a fallback mount may be cleartext — so an operator
        watching gbserver logs (not just the per-step log) sees it. Default None."""
        return None


def resolve_shared_workdir(config: Optional["EnvironmentConfig"]) -> Optional[str]:
    """Resolve the shared_workdir root from an EnvironmentConfig (only ``.config``
    is read). ``shared_filesystem`` defines the mount; ``shared_workdir`` is the
    explicit workdir path (EnvironmentConfig validates it is under mount_point).
    Legacy environments set only ``shared_workdir``. Returns None when neither is
    set."""
    if config is None:
        return None
    return (config.config or {}).get("shared_workdir")


def resolve_workdir_mount(config: Optional["EnvironmentConfig"]):
    """Return the SharedFilesystemConfig whose mount_point prefixes shared_workdir
    (the mount that hosts the per-run workdir), or None."""
    if config is None:
        return None
    workdir = resolve_shared_workdir(config)
    if not workdir:
        return None
    from gbserver.environment.shared_fs.config import parse_shared_filesystems

    for m in parse_shared_filesystems((config.config or {}).get("shared_filesystem")):
        mp = m.mount_point
        if workdir == mp or workdir.startswith(mp.rstrip("/") + "/"):
            return m
    return None


def resolve_local_scratch(config: Optional["EnvironmentConfig"]) -> Optional[str]:
    """Return the single mount's local_scratch, or None (caller applies default)."""
    if config is None:
        return None
    from gbserver.environment.shared_fs.config import parse_shared_filesystems

    for m in parse_shared_filesystems((config.config or {}).get("shared_filesystem")):
        if m.local_scratch:
            return m.local_scratch
    return None
