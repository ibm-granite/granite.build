# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The gbcli-backed GbLogReader (ports the 2025 gb_service log fetch)."""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

from autotunex.core.exceptions import GbLogsUnavailableError, GbLogsUpstreamError
from autotunex.core.logging import get_logger

logger = get_logger(__name__)

_SECONDS_PER_DAY = 86_400


class GbcliLogReader:
    """Fetches build logs from the gb log-query server. Satisfies :class:`GbLogReader`."""

    def __init__(self, token_env: str, *, window_days: int = 7, page_cap: int = 50) -> None:
        self._token_env = token_env
        self._window_days = window_days
        self._page_cap = page_cap

    async def fetch(self, build_id: str, *, fetch_all: bool) -> list[str]:
        """See :class:`GbLogReader`. Runs the blocking gb call in a thread."""
        token = os.environ.get(self._token_env)
        if not token:
            raise GbLogsUnavailableError()
        return await asyncio.to_thread(self._fetch, token, build_id, fetch_all)

    def _fetch(self, token: str, build_id: str, fetch_all: bool) -> list[str]:
        # gbcli ships no py.typed marker and is absent from the base install CI
        # type-checks against, so mypy cannot resolve these lazy imports. It is
        # treated as untyped via a `[tool.mypy]` override in pyproject.toml
        # (module = "gbcli.*"), exactly like the autotune/pyarrow overrides there —
        # an inline `# type: ignore[import-untyped]` covered only the found-but-
        # untyped case, not the module-absent one CI hits.
        try:
            from gbcli.utils.cli_config import configureGBWorkingEnv
        except ImportError as exc:  # gbcli not installed in this deployment
            raise GbLogsUnavailableError() from exc

        # gbcli's CLI seeds GB_CONFIG / GB_CACHE / DMF_CACHE via this bootstrap
        # before any gbserver call; a *library* consumer (as we are) must run it
        # too, or gbcli's credential resolution dereferences a bare
        # ``os.environ["GB_CONFIG"]`` and raises KeyError. It must precede the
        # imports below, which may read the working env at import time, and is
        # idempotent (it only sets vars that are unset). Ports the
        # ``configureGBWorkingEnv()`` call in the 2025 repo's gb_service.
        configureGBWorkingEnv()

        try:
            from gbcli.utils.gbconstants import BUILD_LOGALL_PAGE_SIZE
            from gbcli.utils.log_query import run_logquery
        except ImportError as exc:  # gbcli not installed in this deployment
            raise GbLogsUnavailableError() from exc

        now = int(time.time())
        window_start = now - self._window_days * _SECONDS_PER_DAY

        def query(start: int) -> list[dict[str, Any]]:
            try:
                response = run_logquery(
                    token,
                    start_epoch_in_s=start,
                    end_epoch_in_s=now,
                    page_size=BUILD_LOGALL_PAGE_SIZE,
                    page_index=0,
                    sort="asc",
                    build_id=build_id,
                )
            except Exception as exc:
                # gbcli raises bare exceptions for a missing working env, absent
                # credentials, or a VPN/network failure. Translate them at this
                # third-party boundary into the reader's declared 503 so the
                # endpoint never leaks a 500. Our own GbLogsUpstreamError is
                # raised below, outside this try, so it is not caught here.
                logger.warning(
                    "gb logquery failed for build_id=%s: %s", build_id, exc, exc_info=True
                )
                raise GbLogsUnavailableError() from exc
            if not response or response.get("status") != 200:
                raise GbLogsUpstreamError()
            entries: list[dict[str, Any]] = response.get("logs") or []
            return entries

        if not fetch_all:
            return _extract_lines(query(window_start))

        lines: list[str] = []
        seen: set[Any] = set()
        start = window_start
        for _ in range(self._page_cap):
            entries = query(start)
            if not entries:
                break
            fresh, last_ts = [], None
            for entry in entries:
                log_id = entry.get("logId")
                if log_id is not None and log_id in seen:
                    continue
                if log_id is not None:
                    seen.add(log_id)
                fresh.append(entry)
                ts = entry.get("timestamp")
                if ts is not None and (last_ts is None or ts > last_ts):
                    last_ts = ts
            lines.extend(_extract_lines(fresh))
            if len(entries) < BUILD_LOGALL_PAGE_SIZE or last_ts is None:
                break
            start = max(int(last_ts / 1000), start + 1)  # ms → s; ensure progress
        else:
            logger.warning("gb logs hit the %d-page cap for build_id=%s", self._page_cap, build_id)
        return lines


def _extract_lines(entries: list[dict[str, Any]]) -> list[str]:
    """Extract the ``log`` string from each record's JSON ``text`` (ports extract_lines)."""
    out: list[str] = []
    for entry in entries:
        try:
            parsed = json.loads(entry["text"])
            message = parsed.get("log")
            out.append(message if message is not None else "<null>")
        except (json.JSONDecodeError, KeyError, TypeError):
            out.append(entry.get("text", ""))
    return out
