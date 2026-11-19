#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a program under a Landlock sandbox (Linux's unprivileged access control).

Usage::

    lcb_sandbox.py [--read PATH]... [--write PATH]... -- PROGRAM [ARG]...

The step runs the scorer, and so the model-generated code, through this. It
restricts itself, then ``exec``s ``PROGRAM``, which inherits the restrictions, as
does every process it starts. They can't be lifted. It needs no privileges and no
user namespaces: only a kernel with Landlock enabled (``landlock`` in
``/sys/kernel/security/lsm``), ABI 4 or later.

What the sandboxed processes can't do:

- **See another process's environment.** A sandboxed process can't trace a process
  outside its sandbox, and reading ``/proc/<pid>/environ`` counts as tracing. So
  every process above the scorer (the step's shell, the launcher's job-log wrapper,
  the cluster's job step), which runs as the same user and still holds the job's
  credentials, is out of reach, whatever wraps the step.
- **Open files outside the allowlist.** Read and execute only under the system
  directories (``_SYSTEM_READ``), the Python running this (its prefixes and
  ``sys.path``), ``PROGRAM`` itself and each ``--read``; write only under each
  ``--write``. The job user's home, keys and other builds' files can't be opened.
- **Use TCP** (bind or connect).
- **Signal, or connect to an abstract unix socket of, a process outside the
  sandbox** (ABI 6 or later; skipped on older kernels).
- **Change mounts.**

Not covered: UDP and pathname unix sockets, and reading other processes'
``/proc/<pid>/cmdline`` and other non-trace ``/proc`` entries.

It exits 1 with a message, without running ``PROGRAM``, when the kernel can't do all
of the above. Paths that don't exist are skipped.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import os
import sys
from typing import Iterable, List, Optional, Sequence

# Generic syscall numbers, the same on x86_64 and aarch64.
_SYS_CREATE_RULESET = 444
_SYS_ADD_RULE = 445
_SYS_RESTRICT_SELF = 446
_CREATE_RULESET_VERSION = 1
_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38

# Filesystem rights (include/uapi/linux/landlock.h), by the ABI that added them.
_FS_EXECUTE = 1 << 0
_FS_WRITE_FILE = 1 << 1
_FS_READ_FILE = 1 << 2
_FS_READ_DIR = 1 << 3
_FS_ABI1 = (1 << 13) - 1  # EXECUTE .. MAKE_SYM
_FS_REFER = 1 << 13  # ABI 2
_FS_TRUNCATE = 1 << 14  # ABI 3
_FS_IOCTL_DEV = 1 << 15  # ABI 5
# The rights a rule on a file (rather than a directory) may carry.
_FS_FILE_ONLY = (
    _FS_EXECUTE | _FS_WRITE_FILE | _FS_READ_FILE | _FS_TRUNCATE | _FS_IOCTL_DEV
)
_NET_TCP = (1 << 0) | (1 << 1)  # BIND_TCP, CONNECT_TCP: ABI 4
_SCOPE_ALL = (1 << 0) | (1 << 1)  # ABSTRACT_UNIX_SOCKET, SIGNAL: ABI 6

#: The oldest ABI that can also deny TCP.
MIN_ABI = 4

#: Read-only for every sandboxed program: what Python, its libraries and
#: LiveCodeBench's multiprocessing pool read at startup. /proc stays readable; the
#: trace restriction above is what keeps other processes' environments out.
_SYSTEM_READ = (
    "/usr",
    "/lib",
    "/lib64",
    "/bin",
    "/sbin",
    "/etc",
    "/proc",
    "/sys",
    "/dev",
)
#: Writable for every sandboxed program: shared memory for multiprocessing's locks.
_SYSTEM_WRITE = ("/dev/shm",)
#: Files written by path, e.g. subprocess.DEVNULL.
_SYSTEM_WRITE_FILES = ("/dev/null",)


class SandboxError(RuntimeError):
    """The kernel can't apply the sandbox."""


class _RulesetAttr(ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
        ("scoped", ctypes.c_uint64),
    ]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def abi_version(libc: Optional[ctypes.CDLL] = None) -> int:
    """The kernel's Landlock ABI version.

    :param libc: the C library to call through (a stand-in in tests).
    :returns: the version, 1 or more.
    :raises SandboxError: if Landlock is missing or disabled.
    """
    libc = libc or _libc()
    abi = libc.syscall(_SYS_CREATE_RULESET, None, 0, _CREATE_RULESET_VERSION)
    if abi < 1:
        err = ctypes.get_errno()
        why = {
            errno.ENOSYS: "this kernel has no Landlock",
            errno.EOPNOTSUPP: "Landlock is disabled on this kernel (not in /sys/kernel/security/lsm)",
        }.get(err, os.strerror(err))
        raise SandboxError(why)
    return abi


def handled_fs(abi: int) -> int:
    """Every filesystem right the given ABI knows, so all of them are denied by default.

    :param abi: the kernel's Landlock ABI version.
    :returns: the rights mask.
    """
    mask = _FS_ABI1
    if abi >= 2:
        mask |= _FS_REFER
    if abi >= 3:
        mask |= _FS_TRUNCATE
    if abi >= 5:
        mask |= _FS_IOCTL_DEV
    return mask


def python_paths() -> List[str]:
    """Where the running Python reads its own files: its prefixes and ``sys.path``.

    :returns: the paths, possibly including ones that don't exist.
    """
    return [sys.prefix, sys.base_prefix, sys.exec_prefix, *filter(None, sys.path)]


def _open_path(path: str) -> int:
    # O_PATH: a handle on the path itself, no read permission needed. Linux only.
    return os.open(path, getattr(os, "O_PATH", 0) | os.O_CLOEXEC)


def _add_rule(libc: ctypes.CDLL, ruleset: int, path: str, rights: int) -> None:
    try:
        fd = _open_path(path)
    except FileNotFoundError:
        return
    try:
        if not os.path.isdir(path):
            rights &= _FS_FILE_ONLY
        attr = _PathBeneathAttr(rights, fd)
        if libc.syscall(
            _SYS_ADD_RULE, ruleset, _RULE_PATH_BENEATH, ctypes.byref(attr), 0
        ):
            err = ctypes.get_errno()
            raise SandboxError(f"can't add a rule for {path}: {os.strerror(err)}")
    finally:
        os.close(fd)


def restrict(
    read: Iterable[str], write: Iterable[str], libc: Optional[ctypes.CDLL] = None
) -> int:
    """Restrict this process, and every process it later starts, for good.

    :param read: paths to allow read and execute under, besides the system ones.
    :param write: paths to allow everything under, besides ``/dev/shm``.
    :param libc: the C library to call through (a stand-in in tests).
    :returns: the kernel's Landlock ABI version.
    :raises SandboxError: if the ABI is older than :data:`MIN_ABI`, or the kernel
        refuses a step.
    """
    libc = libc or _libc()
    abi = abi_version(libc)
    if abi < MIN_ABI:
        raise SandboxError(
            f"Landlock ABI {abi} can't deny TCP (needs {MIN_ABI}, Linux 6.7 or a backport)"
        )
    fs = handled_fs(abi)
    attr = _RulesetAttr(fs, _NET_TCP, _SCOPE_ALL if abi >= 6 else 0)
    # Older kernels reject a size that includes fields they don't know.
    size = ctypes.sizeof(attr) if abi >= 6 else _RulesetAttr.scoped.offset
    ruleset = libc.syscall(_SYS_CREATE_RULESET, ctypes.byref(attr), size, 0)
    if ruleset < 0:
        raise SandboxError(f"can't create a ruleset: {os.strerror(ctypes.get_errno())}")
    try:
        ro = _FS_EXECUTE | _FS_READ_FILE | _FS_READ_DIR
        for path in (*_SYSTEM_READ, *read):
            _add_rule(libc, ruleset, path, ro)
        # /dev's device ioctls (e.g. isatty on a terminal) stay allowed.
        _add_rule(libc, ruleset, "/dev", ro | (fs & _FS_IOCTL_DEV))
        for path in _SYSTEM_WRITE_FILES:
            _add_rule(libc, ruleset, path, ro | _FS_WRITE_FILE | (fs & _FS_TRUNCATE))
        for path in (*_SYSTEM_WRITE, *write):
            _add_rule(libc, ruleset, path, fs)
        # Required before restricting without CAP_SYS_ADMIN; also stops setuid
        # programs from gaining privileges inside the sandbox.
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0):
            raise SandboxError(
                f"can't set no_new_privs: {os.strerror(ctypes.get_errno())}"
            )
        if libc.syscall(_SYS_RESTRICT_SELF, ruleset, 0):
            raise SandboxError(
                f"can't restrict this process: {os.strerror(ctypes.get_errno())}"
            )
    finally:
        os.close(ruleset)
    return abi


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse the command line.

    :param argv: the arguments, without the program name.
    :returns: ``read`` and ``write`` (lists of paths) and ``command`` (the program and
        its arguments).
    """
    p = argparse.ArgumentParser(
        prog="lcb_sandbox.py", description=__doc__.split("\n")[0]
    )
    p.add_argument("--read", action="append", default=[], metavar="PATH")
    p.add_argument("--write", action="append", default=[], metavar="PATH")
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        p.error("no program to run: put it after --")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    program = args.command[0]
    try:
        abi = restrict([*python_paths(), program, *args.read], args.write)
    except SandboxError as e:
        print(
            f"eval/livecodebench-score: ERROR sandbox is on but this node can't apply it ({e}); "
            "refusing to run model-generated code without it. Set lcb_score_config.sandbox: "
            "false only for completions you trust.",
            file=sys.stderr,
        )
        return 1
    print(
        f"eval/livecodebench-score: sandbox on (Landlock ABI {abi})",
        file=sys.stderr,
        flush=True,
    )
    os.execv(program, args.command)
    return 1  # pragma: no cover - execv doesn't return


if __name__ == "__main__":
    sys.exit(main())
