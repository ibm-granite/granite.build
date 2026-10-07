"""EFS provider: BYO pre-provisioned NFS filesystem, mounted on host or inside a
containerized step. SkyPilot's container run options already permit the in-container
mount -- ``--cap-add=SYS_ADMIN`` for mount(2) and ``--security-opt=apparmor:unconfined``
to clear AppArmor (plus host networking to reach the mount target)."""

import asyncio
import shlex
from typing import Optional

from gbserver.environment.shared_fs.base import SharedFilesystemProvider
from gbserver.environment.shared_fs.config import EfsConfig
from gbserver.environment.shared_fs.efs_provisioning import (
    EfsDeprovisionError,
    deprovision_efs,
    provision_efs,
)

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


# amazon-efs-utils release built on the teardown VM. Pinned to v1.35.2 because it
# is the last pure-Python/stunnel release; v2+ needs a Rust toolchain to build.
# Tags can be moved, so the clone must also resolve to this exact commit before
# build-deb.sh runs as root.
_EFS_UTILS_VERSION = "v1.35.2"
_EFS_UTILS_COMMIT = "0fdb5b0af469737f5730c5bdca91d9846f042247"
_VERIFY_EFS_UTILS_COMMIT = (
    f'if [ "$(git -C "$__gb_efs" rev-parse HEAD)" != "{_EFS_UTILS_COMMIT}" ]; then\n'
    f'  echo "shared_filesystem: efs-utils {_EFS_UTILS_VERSION} is not the pinned '
    f'commit {_EFS_UTILS_COMMIT}; refusing to build it" >&2; exit 1\n'
    "fi\n"
)
# Install amazon-efs-utils (mount.efs) if absent. For the gbserver-owned teardown
# cleanup VM ONLY: that throwaway VM runs SkyPilot's default image (no mount.efs),
# and an access-point mount cannot fall back to nfs4. Worker steps NEVER
# auto-install -- their access-point mount fails fast when mount.efs is missing.
# apt hosts build the .deb from github.com/aws/efs-utils (Ubuntu ships no
# amazon-efs-utils package); yum hosts (Amazon Linux) install the distro package.
# The installs wait up to 120s for the dpkg lock rather than abort the reap if
# something else on a freshly booted VM still holds it.
_APT_GET = "$SUDO env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=120"
_INSTALL_EFS_UTILS = (
    "if ! command -v mount.efs >/dev/null 2>&1; then\n"
    "  if command -v apt-get >/dev/null 2>&1; then\n"
    f"    {_APT_GET} update -qq\n"
    f"    {_APT_GET} install -y -qq git "
    "ca-certificates binutils build-essential debhelper dh-make nfs-common "
    "stunnel4 python3\n"
    '    __gb_efs="$(mktemp -d)"\n'
    f"    git clone --depth 1 --branch {_EFS_UTILS_VERSION} "
    'https://github.com/aws/efs-utils "$__gb_efs"\n'
    f"{_VERIFY_EFS_UTILS_COMMIT}"
    '    (cd "$__gb_efs" && $SUDO ./build-deb.sh)\n'
    f"    {_APT_GET} install -y -qq "
    '"$__gb_efs"/build/amazon-efs-utils*.deb\n'
    "  else\n"
    "    $SUDO yum install -y -q amazon-efs-utils\n"
    "  fi\n"
    "fi\n"
)
# Teardown wait budget for an access-point reap (see cleanup_timeout_s).
_AP_CLEANUP_TIMEOUT_S = 600.0


class EfsProvider(SharedFilesystemProvider):
    def __init__(self, mount_point: str, cfg: EfsConfig) -> None:
        super().__init__(mount_point)
        self.cfg = cfg

    def _mount_line(self, mp_quoted: str, dns_override: Optional[str] = None) -> str:
        # BYO: validation guarantees a derivable DNS name (fsid+region or dns_name)
        # and it wins. Ephemeral: config has no DNS, so the runtime dns_override
        # (the just-created filesystem's DNS) is used.
        dns = self.cfg.derived_dns_name() or dns_override
        if dns is None:
            # Only ephemeral reaches here (BYO validation guarantees a DNS): a
            # missing override means setup_config lost the provisioned filesystem's
            # dns_name (stale/replayed config or a broken retry). Fail with a clear
            # message rather than letting shlex.quote(None + ':/') raise TypeError.
            raise ValueError(
                f"shared_filesystem: ephemeral EFS mount at {self.mount_point} has "
                "no runtime DNS (setup_config is missing the provisioned "
                "filesystem's dns_name)"
            )
        tls = " -o tls" if self.cfg.tls else ""
        # BYO carries the fsid in config; ephemeral forbids it, but its runtime DNS
        # is the AWS-generated "<fsid>.efs.<region>.amazonaws.com", so recover the
        # fsid from it. Without this, ephemeral never takes the mount.efs path and
        # silently mounts cleartext nfs4 even where amazon-efs-utils is present,
        # defeating tls=true. (BYO with only dns_name keeps nfs4 -- no fsid known.)
        fsid = self.cfg.file_system_id
        if fsid is None and self.cfg.provision == "ephemeral":
            fsid = dns.split(".", 1)[0]
        if self.cfg.access_point_id:
            if fsid is None:
                # Validation guarantees file_system_id for a BYO access-point
                # mount; guard here so mypy can narrow and to fail clearly if a
                # future caller bypasses validation.
                raise ValueError(
                    "shared_filesystem: access_point_id at "
                    f"{self.mount_point} requires file_system_id"
                )
            # Access-point mount: mount.efs requires tls with accesspoint (it
            # refuses `-o accesspoint` alone), and validation enforces `tls: true`
            # for access-point configs, so tls is always emitted here.
            opts = f"accesspoint={self.cfg.access_point_id},tls"
            ap_cmd = (
                f"$SUDO mount -t efs -o {opts} {shlex.quote(fsid + ':/')} {mp_quoted}"
            )
            ap_fail = (
                f'echo "shared_filesystem: access_point_id set at '
                f"{self.mount_point} but amazon-efs-utils (mount.efs) is absent; "
                'cannot mount via an access point" >&2; exit 1'
            )
            # No nfs4 fallback: nfs4 cannot select an access point.
            return (
                f"if command -v mount.efs >/dev/null 2>&1; then {ap_cmd}; "
                f"else {ap_fail}; fi"
            )
        efs_cmd = (
            f"$SUDO mount -t efs{tls} {shlex.quote(fsid + ':/')} {mp_quoted}"
            if fsid
            else None
        )
        # Plain nfs4 cannot encrypt to EFS (that needs amazon-efs-utils' stunnel via
        # `mount -t efs -o tls`). When tls was requested and we fall back to nfs4,
        # warn loudly rather than mount in cleartext while the config says tls: true.
        tls_warn = (
            'echo "shared_filesystem: WARNING tls=true but amazon-efs-utils '
            "(mount.efs) is absent; mounting EFS over nfs4 WITHOUT encryption in "
            'transit" >&2; '
            if self.cfg.tls
            else ""
        )
        nfs_cmd = (
            f"{tls_warn}$SUDO mount -t nfs4 -o {_NFS_OPTS} "
            f"{shlex.quote(dns + ':/')} {mp_quoted}"
        )
        if efs_cmd:
            # Prefer amazon-efs-utils on bare hosts (honors -o tls); fall back to
            # nfs4 (containers / stock images), warning if tls was requested.
            return f"if command -v mount.efs >/dev/null 2>&1; then {efs_cmd}; else {nfs_cmd}; fi"
        return nfs_cmd

    def mount_prologue(self, dns_override: Optional[str] = None) -> str:
        mp = shlex.quote(self.mount_point)
        fail = f'echo "shared_filesystem: EFS mount at {self.mount_point} failed" >&2; exit 1'
        # Ephemeral EFS has a fresh root:root 0755 root; gbserver cannot mount it
        # from k8s, so the first step (sudo on bare / root in container) makes it
        # sticky world-writable so non-root steps can create the per-run workdir.
        chmod_root = (
            f"  $SUDO chmod 1777 {mp}\n" if self.cfg.provision == "ephemeral" else ""
        )
        return (
            f"{_SUDO_SETUP}"
            f"{_INSTALL_NFS}\n"
            # An existing mount at mount_point is trusted as-is, so the mount
            # stays idempotent across steps on a host. With access_point_id, a
            # pre-existing mount of the filesystem ROOT there (made out of band,
            # not by gbserver) would bypass the access point.
            f"if ! mountpoint -q {mp}; then\n"
            f"  $SUDO mkdir -p {mp}\n"
            f"  {self._mount_line(mp, dns_override)} || {{ {fail}; }}\n"
            f"{chmod_root}"
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
            # Access-point mounts need mount.efs (no nfs4 fallback), which the
            # default-image teardown VM lacks: install it first (teardown only).
            + (_SUDO_SETUP + _INSTALL_EFS_UTILS if self.cfg.access_point_id else "")
            + self.mount_prologue()
            + f"rm -rf {pr}\n"
            + f'rmdir --ignore-fail-on-non-empty "$(dirname {pr})" 2>/dev/null || true\n'
            + f'rmdir --ignore-fail-on-non-empty "$(dirname "$(dirname {pr})")" 2>/dev/null || true\n'
        )

    def _session(self, aws_profile):
        # Lazy: boto3 ships only with the skypilot/aws extra; BYO + non-AWS envs
        # and unit tests import this module without it (mirror _require_skypilot).
        import boto3

        return (
            boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
        )

    async def provision(self, tags, aws_profile):
        if self.cfg.provision != "ephemeral":
            return None
        session = self._session(aws_profile)
        return await asyncio.to_thread(
            provision_efs,
            session,
            self.cfg.region,
            tags,
            self.cfg.vpc_id,
            self.cfg.subnets,
            self.cfg.security_group_id,
            self.mount_point,
        )

    async def deprovision(self, provisioned, aws_profile):
        if provisioned is None:
            return
        session = self._session(aws_profile)
        failures = await asyncio.to_thread(deprovision_efs, session, provisioned)
        if failures:
            raise EfsDeprovisionError(provisioned, failures)

    def cleanup_zone(self) -> Optional[str]:
        return self.cfg.cleanup_zone

    def cleanup_timeout_s(self) -> Optional[float]:
        # An access-point reap first builds amazon-efs-utils on the cleanup VM
        # (apt installs that may wait on the dpkg lock, plus a .deb build), so
        # give it more than the teardown default before reporting it unconfirmed.
        return _AP_CLEANUP_TIMEOUT_S if self.cfg.access_point_id else None

    def transit_encryption_note(self) -> Optional[str]:
        if self.cfg.access_point_id:
            # No cleartext fallback under access-point mode (mount fails fast if
            # amazon-efs-utils is missing), so there is nothing to warn about.
            return None
        if not self.cfg.tls:
            return None
        return (
            "shared_filesystem: tls=true requested, but EFS TLS needs "
            "amazon-efs-utils (mount.efs); if it is absent on the worker/image the "
            "mount falls back to UNENCRYPTED nfs4 (the step log records which path ran)."
        )
