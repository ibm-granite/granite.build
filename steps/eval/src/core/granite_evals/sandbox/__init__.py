"""Per-task container sandboxes (SWE-bench, Terminal-Bench, ...).

Backends shell out to a container CLI, so the step image only needs the binary:

- ``enroot``: SLURM / SkyPilot, unprivileged, no daemon (default there).
- ``podman`` / ``docker``: laptops and hosts with a container engine.

Pick one with ``GRANITE_EVALS_SANDBOX`` or ``--option sandbox=...``.
"""

from __future__ import annotations

import abc
import os
import subprocess
import uuid
from dataclasses import dataclass


@dataclass
class ExecResult:
    output: str
    returncode: int
    timed_out: bool = False


class Sandbox(abc.ABC):
    """One running container built from ``image`` (a ``docker://`` style ref)."""

    def __init__(self, image: str, *, env: dict[str, str] | None = None):
        self.image = image.removeprefix("docker://")
        self.env = dict(env or {})
        self.name = f"granite-{uuid.uuid4().hex[:12]}"

    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def _exec_argv(self, command: str, cwd: str, env: dict[str, str]) -> list[str]: ...

    @abc.abstractmethod
    def close(self) -> None: ...

    def _run_env(self) -> dict[str, str] | None:
        """Environment of the local process that runs a command (None = ours)."""
        return None

    def execute(
        self,
        command: str,
        *,
        cwd: str = "/",
        timeout: int | None = None,
        env: dict[str, str] | None = None,
        stdin: str | None = None,
    ) -> ExecResult:
        argv = self._exec_argv(command, cwd, {**self.env, **(env or {})})
        try:
            p = subprocess.run(
                argv,
                input=stdin,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                env=self._run_env(),
            )
            return ExecResult(p.stdout, p.returncode)
        except subprocess.TimeoutExpired as e:
            out = e.output.decode("utf-8", "replace") if isinstance(e.output, bytes) else (e.output or "")
            return ExecResult(out, -1, timed_out=True)

    def write_file(self, path: str, content: str) -> None:
        r = self.execute(f"mkdir -p \"$(dirname '{path}')\" && cat > '{path}'", stdin=content)
        if r.returncode != 0:
            raise RuntimeError(f"writing {path} in sandbox failed: {r.output}")

    def __enter__(self) -> Sandbox:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def make_sandbox(image: str, *, backend: str = "", env: dict[str, str] | None = None) -> Sandbox:
    backend = backend or os.environ.get("GRANITE_EVALS_SANDBOX", "enroot")
    if backend == "enroot":
        from granite_evals.sandbox.enroot import EnrootSandbox

        return EnrootSandbox(image, env=env)
    if backend in ("podman", "docker"):
        from granite_evals.sandbox.oci import OCISandbox

        return OCISandbox(image, env=env, engine=backend)
    raise ValueError(f"unknown sandbox backend {backend!r} (enroot, podman, docker)")
