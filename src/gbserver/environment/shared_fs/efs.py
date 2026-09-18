"""EFS provider: BYO pre-provisioned NFS filesystem, mounted on host or inside
a containerized step (SkyPilot containers already grant SYS_ADMIN + /dev/fuse +
--net=host, so an in-container mount works)."""

import shlex
from typing import Optional

from gbserver.environment.shared_fs.base import SharedFilesystemProvider
from gbserver.environment.shared_fs.config import EfsConfig

_NFS_OPTS = (
    "nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport"
)
# Privileged commands need `sudo` on the bare host (SkyPilot runs steps as a
# non-root user with passwordless sudo), but a containerized step runs as root in
# a minimal image (e.g. debian:12-slim) that has NO `sudo` at all — calling it
# there dies with "sudo: not found". Gate on the effective uid: root -> no sudo.
_SUDO_SETUP = 'SUDO=""; [ "$(id -u)" -eq 0 ] || SUDO="sudo"\n'
# Best-effort NFS client install (SkyPilot's container setup already assumes debian).
_INSTALL_NFS = (
    "command -v mount.nfs4 >/dev/null 2>&1 || "
    "{ $SUDO apt-get update -qq && $SUDO apt-get install -y -qq nfs-common; } || "
    "{ command -v yum >/dev/null 2>&1 && $SUDO yum install -y -q nfs-utils; } || true"
)


class EfsProvider(SharedFilesystemProvider):
    def __init__(self, mount_point: str, cfg: EfsConfig) -> None:
        super().__init__(mount_point)
        self.cfg = cfg

    def _mount_line(self, mp_quoted: str) -> str:
        # Validation guarantees a derivable DNS name (fsid+region or dns_name).
        dns = self.cfg.derived_dns_name()
        tls = " -o tls" if self.cfg.tls else ""
        fsid = self.cfg.file_system_id
        efs_cmd = (
            f"$SUDO mount -t efs{tls} {shlex.quote(fsid + ':/')} {mp_quoted}"
            if fsid
            else None
        )
        nfs_cmd = (
            f"$SUDO mount -t nfs4 -o {_NFS_OPTS} {shlex.quote(dns + ':/')} {mp_quoted}"
        )
        if efs_cmd:
            # Prefer amazon-efs-utils on bare hosts; fall back to nfs4 (containers).
            return f"if command -v mount.efs >/dev/null 2>&1; then {efs_cmd}; else {nfs_cmd}; fi"
        return nfs_cmd

    def mount_prologue(self) -> str:
        mp = shlex.quote(self.mount_point)
        fail = f'echo "shared_filesystem: EFS mount at {self.mount_point} failed" >&2; exit 1'
        return (
            f"{_SUDO_SETUP}"
            f"{_INSTALL_NFS}\n"
            f"if ! mountpoint -q {mp}; then\n"
            f"  $SUDO mkdir -p {mp}\n"
            f"  {self._mount_line(mp)} || {{ {fail}; }}\n"
            f"fi\n"
        )

    def cleanup_run_script(self, per_run_workdir: str) -> str:
        # Always single-quote (unlike shlex.quote, which omits quotes for
        # shell-safe paths) so the emitted paths are unambiguous.
        pr = "'" + per_run_workdir.replace("'", "'\\''") + "'"
        return (
            # Fail-fast like the step prologue. The mount line already aborts
            # (|| exit 1) before any rm if the mount fails; `set -eu` is
            # defense-in-depth so nothing runs after a silent failure.
            "set -eu\n"
            + self.mount_prologue()
            + f"rm -rf {pr}\n"
            + f'rmdir --ignore-fail-on-non-empty "$(dirname {pr})" 2>/dev/null || true\n'
            + f'rmdir --ignore-fail-on-non-empty "$(dirname "$(dirname {pr})")" 2>/dev/null || true\n'
        )

    def cleanup_zone(self) -> Optional[str]:
        return self.cfg.cleanup_zone
