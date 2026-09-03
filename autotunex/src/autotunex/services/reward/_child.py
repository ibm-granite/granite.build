# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Sandbox child process: run an untrusted reward function under hard limits.

Started as ``python -m autotunex.services.reward._child`` by
:class:`SubprocessRewardExecutor`. Reads ``{code, function_name, test_cases}``
JSON from stdin; writes a result JSON to stdout. Never trusted to self-limit —
the parent also enforces a wall-clock ``SIGKILL``.

**Coverage stops here, deliberately.** Nothing in this module runs in the test
process: it only ever executes as a separate interpreter that
:class:`SubprocessRewardExecutor` spawns, so covering it would mean a subprocess
coverage harness in CI. It is exercised end-to-end through that executor's own
tests instead — what is untested is this file's lines, not the sandbox's
behaviour. See ibm-granite/granite.build#318 for the decision.
"""

from __future__ import annotations

import builtins as _builtins
import contextlib
import io
import json
import sys
import time
from typing import Any

from autotunex.services.reward.constants import (
    ALLOWED_EXEC_MODULES,
    MAX_STDOUT_CHARS,
    MAX_TEST_CASES,
    SAFE_BUILTINS,
)


def _apply_rlimits(memory_bytes: int, cpu_seconds: int) -> None:
    """Best-effort resource caps (Linux enforces; macOS may ignore RLIMIT_AS)."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except (ImportError, ValueError, OSError):
        pass


def _build_globals() -> dict[str, object]:
    """Build the restricted ``__builtins__`` namespace the reward code runs under."""
    safe: dict[str, object] = {
        name: getattr(_builtins, name) for name in SAFE_BUILTINS if hasattr(_builtins, name)
    }
    safe["__build_class__"] = _builtins.__build_class__
    safe["__name__"] = "__main__"

    def _restricted_import(name: str, *args: Any, **kwargs: Any) -> object:  # noqa: ANN401
        if name.split(".")[0] in ALLOWED_EXEC_MODULES:
            return _builtins.__import__(name, *args, **kwargs)
        raise ImportError(f"Import of '{name}' is not allowed")

    safe["__import__"] = _restricted_import
    return {"__builtins__": safe}


def _run(payload: dict[str, Any]) -> dict[str, Any]:
    """Compile, exec, and call the reward function against each test case."""
    code = str(payload["code"])
    function_name = str(payload["function_name"])
    raw_cases = payload.get("test_cases") or []
    cases = raw_cases if isinstance(raw_cases, list) else [raw_cases]

    stdout_capture = io.StringIO()
    started = time.time()
    restricted = _build_globals()
    try:
        compiled = compile(code, "<reward_function>", "exec")
        exec(compiled, restricted)  # the restricted namespace is the sandbox
    except Exception as exc:  # report any compile/exec failure as data, not a crash
        return {
            "executed": True,
            "results": [],
            "stdout": "",
            "error": f"{type(exc).__name__}: {exc}",
            "execution_time_ms": round((time.time() - started) * 1000, 1),
        }

    # The namespace's values are `object` by declaration, but this one came from
    # exec()-ing arbitrary user code, so its real (callable) type is unknowable
    # ahead of time — that is inherent to what this sandbox does.
    fn: Any = restricted.get(function_name)
    if fn is None:
        return {
            "executed": True,
            "results": [],
            "stdout": "",
            "error": f"Function '{function_name}' not found after execution",
            "execution_time_ms": round((time.time() - started) * 1000, 1),
        }

    results: list[dict[str, Any]] = []
    with contextlib.redirect_stdout(stdout_capture):
        for index, case in enumerate(cases[:MAX_TEST_CASES]):
            if not isinstance(case, dict):
                results.append(
                    {
                        "case": index + 1,
                        "inputs": case,
                        "return_value": None,
                        "return_type": None,
                        "error": f"Test case {index + 1} must be an object",
                    }
                )
                continue
            try:
                return_value = fn(**case)
                serializable = isinstance(
                    return_value, (int, float, str, bool, list, dict, type(None))
                )
                results.append(
                    {
                        "case": index + 1,
                        "inputs": case,
                        "return_value": return_value if serializable else str(return_value),
                        "return_type": type(return_value).__name__,
                        "error": None,
                    }
                )
            except Exception as exc:  # a per-case failure is data, not a crash
                results.append(
                    {
                        "case": index + 1,
                        "inputs": case,
                        "return_value": None,
                        "return_type": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    return {
        "executed": True,
        "results": results,
        "stdout": stdout_capture.getvalue()[:MAX_STDOUT_CHARS],
        "error": None,
        "execution_time_ms": round((time.time() - started) * 1000, 1),
    }


def main() -> None:
    """Read the payload from stdin, run it, and write the result JSON to stdout."""
    payload = json.loads(sys.stdin.read())
    _apply_rlimits(int(payload["memory_bytes"]), int(payload["cpu_seconds"]))
    sys.stdout.write(json.dumps(_run(payload)))


if __name__ == "__main__":
    main()
