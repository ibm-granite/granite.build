# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The disabled GbLogReader — used when gb is not configured."""

from __future__ import annotations

from autotunex.core.exceptions import GbLogsUnavailableError


class DisabledGbLogReader:
    """Always reports gb logs unavailable. Satisfies :class:`GbLogReader`."""

    async def fetch(self, build_id: str, *, fetch_all: bool) -> list[str]:
        """Always raise :class:`GbLogsUnavailableError` (gb is not configured)."""
        raise GbLogsUnavailableError()
