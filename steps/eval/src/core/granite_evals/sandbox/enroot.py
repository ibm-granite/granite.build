"""enroot backend.

Images are imported once into a squashfs cache (``GRANITE_EVALS_ENROOT_CACHE``, a
shared filesystem on the cluster so every job reuses them) and each sandbox is a
throwaway ``enroot create`` of that squashfs. Commands run with
``enroot start --root --rw``: SWE-bench images expect to be root in /testbed.

Every ``enroot start`` mounts a fresh tmpfs on /tmp (enroot's default
mounts), and each command is its own ``start``, so a sandbox gets a host
directory (under ``ENROOT_TEMP_PATH``) bound over /tmp instead: files written
there by one command are still there for the next.

Sandboxes share the host's network namespace. On shared clusters the host's
/etc/hosts maps ``localhost`` to ::1 as well as 127.0.0.1, so clients that
take the first address (Node) connect to ::1, while task servers that bind
IPv4 only (NodeBB's test forum) refuse them. Docker containers, where the
tasks were written, usually have no IPv6, so there ``localhost`` means
127.0.0.1. Each sandbox gets /etc/hosts bound to a copy of the host's with
``localhost`` dropped from the IPv6 lines.

Sandboxes get no GPUs: enroot passes our environment through, and in a GPU
job NVIDIA_VISIBLE_DEVICES would make its nvidia hook look for
nvidia-container-cli, which the image doesn't have.

Nor do they get our secrets: the model under test runs code in the sandbox,
so variables whose names look like credentials (judge and user-simulator
keys, HF_TOKEN, ...) are dropped from what ``enroot start`` passes through.
A benchmark that needs one in the sandbox passes it explicitly as ``env``.

Imports use the rootless helpers in :mod:`granite_evals.sandbox.ovlfs`, because
enroot's own need capabilities Some compute nodes don't grant. Set
``GRANITE_EVALS_ENROOT_ROOTLESS=0`` to use enroot's.
"""

from __future__ import annotations

import fcntl
import functools
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

from granite_evals.sandbox import Sandbox, ovlfs

log = logging.getLogger(__name__)

ENROOT = os.environ.get("GRANITE_EVALS_ENROOT", "enroot")
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL|AUTH", re.IGNORECASE)


def enroot_uri(image: str) -> str:
    """``docker.io/org/img:tag`` -> ``docker://org/img:tag``; other registries
    use enroot's ``registry#path`` form."""
    first, _, rest = image.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        if first in ("docker.io", "index.docker.io", "registry-1.docker.io"):
            return f"docker://{rest}"
        return f"docker://{first}#{rest}"
    return f"docker://{image}"


def cache_dir() -> Path:
    d = Path(os.environ.get("GRANITE_EVALS_ENROOT_CACHE", Path.home() / ".cache" / "granite" / "enroot"))
    d.mkdir(parents=True, exist_ok=True)
    return d


@functools.cache
def _import_env() -> dict[str, str] | None:
    if os.environ.get("GRANITE_EVALS_ENROOT_ROOTLESS", "1") == "0":
        return None
    return {**os.environ, "PATH": f"{ovlfs.helper_dir()}{os.pathsep}{os.environ.get('PATH', '')}"}


def ensure_squashfs(image: str) -> Path:
    """Import ``image`` into the cache unless present. Safe under concurrency:
    one importer per image, the rest wait on the lock."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", image)
    sqsh = cache_dir() / f"{safe}.sqsh"
    if sqsh.exists():
        return sqsh
    with open(sqsh.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not sqsh.exists():
            tmp = sqsh.with_suffix(".sqsh.partial")
            tmp.unlink(missing_ok=True)
            log.info("enroot import %s", image)
            subprocess.run([ENROOT, "import", "-o", str(tmp), enroot_uri(image)], check=True, env=_import_env())
            tmp.rename(sqsh)
    return sqsh


def _kill_rooted(rootfs: str) -> int:
    """SIGKILL every process of ours whose root directory is ``rootfs``: the
    daemons a task starts that drop an environment marker (nginx workers
    clear their environment, redis-server overwrites it), which would otherwise outlive the task and keep
    the batch job alive."""
    # Compare by inode: enroot pivots into the rootfs in its own mount
    # namespace, so readlink() of /proc/<pid>/root from outside it gives "/"
    # (checked on a SLURM cluster), while stat() follows it to the directory itself.
    try:
        st = os.stat(rootfs)
    except OSError:
        return 0
    want = (st.st_dev, st.st_ino)
    host = os.stat("/")
    if want == (host.st_dev, host.st_ino):  # never every process on the node
        return 0
    killed = 0
    try:
        entries = os.listdir("/proc")
    except OSError:  # not Linux
        return 0
    for entry in entries:
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            root = os.stat(f"/proc/{entry}/root")
            if (root.st_dev, root.st_ino) == want:
                os.kill(int(entry), signal.SIGKILL)
                killed += 1
        except (OSError, ProcessLookupError):
            continue
    return killed


def enroot_rootfs(name: str) -> str:
    """Where ``enroot create --name name`` puts the container's root filesystem."""
    data = os.environ.get("ENROOT_DATA_PATH") or os.path.join(
        os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"), "enroot"
    )
    return os.path.join(data, name)


def ipv4_localhost(hosts: str) -> str:
    """``hosts`` with the name ``localhost`` dropped from its IPv6 lines."""
    out = []
    for line in hosts.splitlines():
        fields = line.split("#", 1)[0].split()
        if len(fields) > 1 and ":" in fields[0] and "localhost" in fields[1:]:
            names = [n for n in fields[1:] if n != "localhost"]
            line = " ".join([fields[0], *names]) if names else ""
        if line:
            out.append(line)
    if not any(
        (f := ln.split("#", 1)[0].split()) and f[0] == "127.0.0.1" and "localhost" in f[1:] for ln in out
    ):
        out.insert(0, "127.0.0.1 localhost")
    return "\n".join(out) + "\n"


class EnrootSandbox(Sandbox):
    tmp: Path | None = None
    hosts: Path | None = None

    def start(self) -> None:
        sqsh = ensure_squashfs(self.image)
        base = os.environ.get("ENROOT_TEMP_PATH") or tempfile.gettempdir()
        os.makedirs(base, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(prefix=f"{self.name}-tmp-", dir=base))
        self.tmp.chmod(0o1777)
        try:
            host_hosts = Path("/etc/hosts").read_text()
        except OSError:
            host_hosts = ""
        self.hosts = Path(f"{self.tmp}.hosts")
        self.hosts.write_text(ipv4_localhost(host_hosts))
        self.hosts.chmod(0o644)
        subprocess.run(
            [ENROOT, "create", "--name", self.name, str(sqsh)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def _exec_argv(self, command: str, cwd: str, env: dict[str, str]) -> list[str]:
        argv = [ENROOT, "start", "--root", "--rw"]
        if self.tmp:
            argv += ["--mount", f"{self.tmp}:/tmp"]
        if self.hosts:
            argv += ["--mount", f"{self.hosts}:/etc/hosts"]
        for key, value in env.items():
            argv += ["--env", f"{key}={value}"]
        return argv + [self.name, "bash", "-c", f"cd {shlex.quote(cwd)} && {command}"]

    def _run_env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not SECRET_NAME.search(k)}
        return {**env, "NVIDIA_VISIBLE_DEVICES": "void"}

    def close(self) -> None:
        # enroot gives a sandbox no PID namespace: what its commands leave
        # running (redis-server --daemonize, a timed-out build's daemons) would
        # outlive it, hold fixed ports and keep the batch job alive.
        killed = _kill_rooted(enroot_rootfs(self.name))
        if killed:
            log.info("%s: killed %d leftover process(es)", self.name, killed)
        subprocess.run(
            [ENROOT, "remove", "--force", self.name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if self.tmp:
            shutil.rmtree(self.tmp, ignore_errors=True)
        if self.hosts:
            self.hosts.unlink(missing_ok=True)
