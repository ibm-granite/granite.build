# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The reward-execution seam.

Isolates *how* untrusted reward code runs from the validation service. The one
shipped implementation is a hardened subprocess; a test double satisfies the
Protocol structurally.
"""

from __future__ import annotations

from typing import Any, Protocol

from autotunex.models.reward import RewardTestResult


class RewardExecutor(Protocol):
    """Runs a reward function against test cases in isolation."""

    async def execute(
        self, *, code: str, function_name: str, test_cases: list[dict[str, Any]]
    ) -> RewardTestResult:
        """Execute and return a structured result; never raises on user-code error."""
        ...
