"""mini-swe-agent ``Environment`` backed by a granite ``Sandbox``.

mini-swe-agent resolves ``environment_class`` by dotted path, so this plugs in
without patching it: the agent loop, prompts and submission protocol stay
upstream, only the place commands run changes (enroot).
"""

from __future__ import annotations

from typing import Any

from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge
from pydantic import BaseModel

from granite_evals.sandbox import make_sandbox

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


class SandboxEnvironmentConfig(BaseModel):
    image: str
    backend: str = ""
    cwd: str = "/"
    env: dict[str, str] = {}
    timeout: int = 60
    interpreter: list[str] = ["bash", "-c"]
    """Accepted for config compatibility with swebench.yaml; always bash -c."""


class SandboxEnvironment:
    def __init__(self, **kwargs):
        self.config = SandboxEnvironmentConfig(**kwargs)
        self.sandbox = make_sandbox(self.config.image, backend=self.config.backend, env=self.config.env)
        self.sandbox.start()

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        r = self.sandbox.execute(
            action.get("command", ""),
            cwd=cwd or self.config.cwd,
            timeout=timeout or self.config.timeout,
        )
        output = {"output": r.output, "returncode": r.returncode, "exception_info": ""}
        if r.timed_out:
            output["exception_info"] = f"Command timed out after {timeout or self.config.timeout}s"
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict) -> None:
        # Same protocol as upstream's docker/singularity environments.
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == SUBMIT_MARKER and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def cleanup(self) -> None:
        self.sandbox.close()
