# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The GbLogReader seam — fetches a build's live container logs from gb."""

from __future__ import annotations

from typing import Protocol


class GbLogReader(Protocol):
    """Fetches a build's live container logs from the gb log-query server."""

    async def fetch(self, build_id: str, *, fetch_all: bool) -> list[str]:
        """Return the build's log lines, oldest-first.

        ``fetch_all=False`` returns the first page only; ``True`` pages the time
        window with ``logId`` dedup.

        Raises:
            GbLogsUnavailableError: the integration is not configured (no token).
            GbLogsUpstreamError: the gb server was unreachable or errored.
        """
        ...
