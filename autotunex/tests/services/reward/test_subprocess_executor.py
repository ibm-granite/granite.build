# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Tests for the hardened subprocess reward executor."""

from __future__ import annotations

import pytest

from autotunex.services.reward.subprocess_executor import (
    SubprocessRewardExecutor,
    _build_child_env,
)

_GOOD = "def compute_score(data_source, solution_str, **kw):\n    return 1.0\n"

# The security-review escape: operator.attrgetter is a string-keyed getattr
# equivalent that used to bypass the AST dunder blocklist and reach
# object.__subclasses__() -> __globals__ -> __import__("os"). Removing operator
# from the exec whitelist must make even importing it fail.
_OPERATOR_ESCAPE = (
    "import operator\n"
    "def compute_score(data_source, solution_str, **kw):\n"
    "    subs = operator.attrgetter('__subclasses__')(object)()\n"
    "    return float(len(subs))\n"
)


async def test_executes_and_returns_a_score() -> None:
    executor = SubprocessRewardExecutor()

    result = await executor.execute(
        code=_GOOD,
        function_name="compute_score",
        test_cases=[{"data_source": "q", "solution_str": "a"}],
    )

    assert result.executed is True
    assert result.error is None
    assert result.results[0].return_value == 1.0
    assert result.results[0].error is None


async def test_per_case_runtime_error_is_captured() -> None:
    code = "def compute_score(data_source, solution_str, **kw):\n    return 1/0\n"
    executor = SubprocessRewardExecutor()

    result = await executor.execute(
        code=code,
        function_name="compute_score",
        test_cases=[{"data_source": "q", "solution_str": "a"}],
    )

    assert result.executed is True
    assert result.results[0].error is not None
    assert "ZeroDivisionError" in result.results[0].error


async def test_infinite_loop_is_hard_killed() -> None:
    code = "def compute_score(data_source, solution_str, **kw):\n    while True:\n        pass\n"
    executor = SubprocessRewardExecutor(timeout_seconds=2)

    result = await executor.execute(
        code=code,
        function_name="compute_score",
        test_cases=[{"data_source": "q", "solution_str": "a"}],
    )

    assert result.executed is False
    assert result.error is not None
    assert "timed out" in result.error.lower()


async def test_operator_reflection_escape_is_blocked() -> None:
    executor = SubprocessRewardExecutor()

    result = await executor.execute(
        code=_OPERATOR_ESCAPE,
        function_name="compute_score",
        test_cases=[{"data_source": "q", "solution_str": "a"}],
    )

    # `import operator` now raises ImportError inside the sandbox, so exec fails
    # before any function is defined — the escape never reaches object.__subclasses__.
    assert result.error is not None
    assert "not allowed" in result.error
    assert result.results == []


def test_build_child_env_withholds_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("GB_TOKEN", "super-secret")
    monkeypatch.setenv("AUTOTUNEX_LLM_API_KEY", "another-secret")

    env = _build_child_env()

    assert env.get("PATH") == "/usr/bin"  # allowlisted essentials pass through
    assert "GB_TOKEN" not in env  # secrets are withheld from the child
    assert "AUTOTUNEX_LLM_API_KEY" not in env
