# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Hardened subprocess implementation of the reward-execution seam.

Untrusted reward code runs in a separate Python process with CPU/memory
rlimits, a restricted builtins namespace, and an import whitelist. The parent
enforces a hard wall-clock timeout by killing the child's process group — the
concrete fix for the 2025 daemon-thread timeout that could not kill runaway
code. On Linux (production/Docker) both rlimits and the kill apply; on macOS
dev RLIMIT_AS may be a no-op, but the timeout kill always applies.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from typing import Any

from autotunex.models.reward import RewardCaseResult, RewardTestResult
from autotunex.services.reward.constants import DEFAULT_MEMORY_BYTES, DEFAULT_TIMEOUT_SECONDS

_CHILD_MODULE = "autotunex.services.reward._child"

# Only these environment variables are passed to the sandbox child. Everything
# else in the parent's environment — DB URLs, GB_TOKEN/HF_TOKEN, OIDC/session
# secrets loaded from .env into os.environ — is withheld, so that even a
# sandbox escape reaching os.environ finds no credentials to exfiltrate. The
# allowlist is only what the interpreter needs to start and to import the
# stdlib and this package: PYTHONPATH covers editable/worktree installs (prod
# resolves the package from site-packages without it); the rest are locale,
# temp, and Windows-startup essentials. None are secrets.
_CHILD_ENV_ALLOWLIST = (
    "PATH",
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONIOENCODING",
    "HOME",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "SYSTEMROOT",
    "PATHEXT",
    "TEMP",
    "TMP",
)


def _build_child_env() -> dict[str, str]:
    """Return the minimal, secret-free environment for the sandbox child.

    Copies only :data:`_CHILD_ENV_ALLOWLIST` keys from the parent's
    environment. Withholding everything else is defense-in-depth: it bounds the
    blast radius of any sandbox escape to "no credentials available" rather than
    handing untrusted code the process's full secret-bearing ``os.environ``.
    """
    return {key: os.environ[key] for key in _CHILD_ENV_ALLOWLIST if key in os.environ}


_CPU_RLIMIT_BUFFER_SECONDS = 30
"""Slack added to the child's own ``RLIMIT_CPU`` above the wall-clock timeout.

The parent's wall-clock ``SIGKILL`` (below) is the primary defense against a
runaway reward function and must win the race for a CPU-bound busy loop: CPU
time and wall time are ~equal for such code, so setting ``RLIMIT_CPU`` to the
same value as the timeout would let the child's own rlimit self-terminate
first, short-circuiting the parent-side kill this executor exists to test and
guarantee. The rlimit is kept as a generous backstop for the case where the
parent-side kill cannot reach the child at all (e.g. the signal is lost)."""


class SubprocessRewardExecutor:
    """Runs reward code in a resource-limited, hard-killed subprocess."""

    def __init__(
        self,
        *,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        memory_bytes: int = DEFAULT_MEMORY_BYTES,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._memory_bytes = memory_bytes

    async def execute(
        self, *, code: str, function_name: str, test_cases: list[dict[str, Any]]
    ) -> RewardTestResult:
        """Execute in a subprocess; return a structured result, never raising on user error."""
        payload = json.dumps(
            {
                "code": code,
                "function_name": function_name,
                "test_cases": test_cases,
                "memory_bytes": self._memory_bytes,
                "cpu_seconds": self._timeout_seconds + _CPU_RLIMIT_BUFFER_SECONDS,
            }
        )
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            _CHILD_MODULE,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,  # own process group so the whole tree can be killed
            env=_build_child_env(),  # secret-free environment (no .env creds reach the child)
        )
        try:
            stdout, _stderr = await asyncio.wait_for(
                proc.communicate(payload.encode()), timeout=self._timeout_seconds + 1
            )
        except TimeoutError:
            self._kill_group(proc)
            await proc.wait()
            return RewardTestResult(
                executed=False,
                error=f"Execution timed out after {self._timeout_seconds}s",
            )

        if proc.returncode != 0 or not stdout:
            return RewardTestResult(
                executed=False,
                error="Sandbox process exited abnormally (possible resource limit or crash)",
            )
        data = json.loads(stdout.decode())
        return RewardTestResult(
            executed=bool(data.get("executed")),
            results=[RewardCaseResult(**item) for item in data.get("results", [])],
            stdout=str(data.get("stdout", "")),
            error=data.get("error"),
            execution_time_ms=data.get("execution_time_ms"),
        )

    @staticmethod
    def _kill_group(proc: asyncio.subprocess.Process) -> None:
        """SIGKILL the child's process group; ignore if already gone.

        ``proc.pid`` is always set once ``create_subprocess_exec`` returns
        (typeshed types it as non-optional), so the only failure mode here is
        the process having already exited between the timeout firing and this
        call — hence catching ``ProcessLookupError``/``PermissionError`` rather
        than checking ``pid`` for ``None``.
        """
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):  # pragma: no cover - race on exit
            proc.kill()
