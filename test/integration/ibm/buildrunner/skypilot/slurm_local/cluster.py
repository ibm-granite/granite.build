# SPDX-License-Identifier: Apache-2.0

"""Shell access to the local Docker SLURM cluster, for tests that inspect what a build
left on its shared filesystem or act on its SLURM jobs."""

import os
import shlex
import subprocess


def on_cluster(script: str, stdin: str | None = None) -> str:
    """Run a shell script on the Docker SLURM login node; return its stdout.

    Same SSH endpoint and key as the skip gate
    (``integration.environment.test_skypilot_slurm_e2e._slurm_cluster_reachable``).
    ``stdin`` is sent on the ssh connection, so data passed that way never appears on
    the remote command line, where ``ps`` would show it.
    """
    proc = subprocess.run(
        [
            "ssh",
            "-i",
            os.path.expanduser("~/.ssh/slurm_docker_key"),
            "-p",
            os.environ.get("SLURM_SSH_PORT", "2222"),
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
            f"root@{os.environ.get('SLURM_SSH_HOST', '127.0.0.1')}",
            script,
        ],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"cluster command failed: {proc.stderr}"
    return proc.stdout


# What a build leaves on the shared filesystem: the step's output, plus the per-build
# workdirs (runs/<id>/, holding the step's .data-designer-home, the Ollama server log and
# the per-run output). Searched for a leaked secret after a test.
_SCANNED = "/shared/builds"


def files_containing(secret: str, *paths: str) -> list[str]:
    """Files under ``paths`` (and the shared build workdirs) that contain ``secret``.

    The secret goes over stdin (``grep -f -``), never on the remote command line, so it
    is neither visible in ``ps`` on the login node nor broken by a quote character in it.
    """
    out = on_cluster(
        f"grep -rlF -f - {' '.join(paths)} {_SCANNED} 2>/dev/null || true",
        stdin=secret + "\n",
    )
    return [line for line in out.splitlines() if line]


def remove_from_shared(path: str) -> None:
    """Remove ``path`` (under ``/shared/``) from the cluster's shared filesystem.

    For build tests whose outputs live at fixed ``/shared`` paths: a later run would
    otherwise find the earlier run's output there.

    :param path: Absolute path below ``/shared/``; ``/shared`` itself is refused.
    :raises ValueError: If ``path`` is not strictly below ``/shared/``.
    """
    norm = os.path.normpath(path)
    if not norm.startswith("/shared/") or norm != path.rstrip("/"):
        raise ValueError(f"not a path strictly below /shared/: {path!r}")
    on_cluster(f"rm -rf -- {shlex.quote(norm)}")
