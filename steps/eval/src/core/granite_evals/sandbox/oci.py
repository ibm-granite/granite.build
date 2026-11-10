"""podman / docker backend, for local development and hosts with an engine."""

from __future__ import annotations

import subprocess

from granite_evals.sandbox import Sandbox


class OCISandbox(Sandbox):
    def __init__(self, image: str, *, env: dict[str, str] | None = None, engine: str = "podman"):
        super().__init__(image, env=env)
        self.engine = engine

    def start(self) -> None:
        # SWE-bench images are published for linux/amd64 only.
        subprocess.run(
            [self.engine, "run", "-d", "--platform", "linux/amd64", "--name", self.name,
             "--entrypoint", "", self.image, "sleep", "infinity"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )  # fmt: skip

    def _exec_argv(self, command: str, cwd: str, env: dict[str, str]) -> list[str]:
        argv = [self.engine, "exec", "-i", "-w", cwd]
        for key, value in env.items():
            argv += ["-e", f"{key}={value}"]
        return argv + [self.name, "bash", "-c", command]

    def close(self) -> None:
        subprocess.run(
            [self.engine, "rm", "-f", self.name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
