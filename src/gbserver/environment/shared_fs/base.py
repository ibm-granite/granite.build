"""SharedFilesystemProvider contract + the shared_workdir resolver."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from gbserver.environment.shared_fs.config import SharedFilesystemConfig

if TYPE_CHECKING:
    from gbserver.types.environmentconfig import EnvironmentConfig


class SharedFilesystemProvider(ABC):
    """Emits the shell that backs a shared_workdir root. skypilot.py owns all
    `sky` orchestration; a provider only returns shell strings."""

    def __init__(self, mount_point: str) -> None:
        self.mount_point = mount_point

    @abstractmethod
    def mount_prologue(self) -> str:
        """Idempotent shell that mounts the FS at ``mount_point`` (host or, for a
        containerized step, inside the container). Must `echo` a clear message
        and exit non-zero on failure so the caller's `set -eu` aborts the step."""

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

    def transit_encryption_note(self) -> Optional[str]:
        """A one-line note logged server-side (gbserver) at launch about transit
        encryption — e.g. that a fallback mount may be cleartext — so an operator
        watching gbserver logs (not just the per-step log) sees it. Default None."""
        return None


def resolve_shared_workdir(config: Optional["EnvironmentConfig"]) -> Optional[str]:
    """Resolve the shared_workdir root from an EnvironmentConfig (only ``.config``
    is read). shared_filesystem -> mount_point; else shared_workdir; else None.
    Both set -> ValueError (defensive; EnvironmentConfig also rejects this)."""
    if config is None:
        return None
    cfg = config.config or {}
    sf_raw = cfg.get("shared_filesystem")
    workdir = cfg.get("shared_workdir")
    if sf_raw and workdir:
        raise ValueError(
            "environment config sets both 'shared_filesystem' and 'shared_workdir'"
        )
    if sf_raw:
        return SharedFilesystemConfig.model_validate(sf_raw).mount_point
    return workdir


def resolve_local_scratch(config: Optional["EnvironmentConfig"]) -> Optional[str]:
    """Return the ``shared_filesystem.local_scratch`` dir, or None when there is no
    ``shared_filesystem`` block (the caller applies the ``/tmp/gb-scratch`` default).
    Read through the typed config so an invalid (e.g. relative) value is rejected."""
    if config is None:
        return None
    sf_raw = (config.config or {}).get("shared_filesystem")
    if not sf_raw:
        return None
    return SharedFilesystemConfig.model_validate(sf_raw).local_scratch
