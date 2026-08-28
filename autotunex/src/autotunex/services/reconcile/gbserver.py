# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""GbServerStatusReader — reads a build's status from the gbserver REST API.

Injected an ``httpx.AsyncClient`` (whose timeout is set by the caller in
``lifespan``), exactly like ``OpenAiCompatibleLlmClient`` — so tests drive it
with ``httpx.MockTransport`` and no real server. The GB token is read from the
environment at call time (never from ``Settings``), matching the launcher's
credential contract; over HTTP there is no ambient fallback, so a deployment
without it will 401 and be logged once per sweep by the loop.
"""

from __future__ import annotations

import os
from typing import Any
from uuid import UUID

import httpx

from autotunex.services.reconcile.protocols import (
    BuildNotFoundError,
    BuildState,
    BuildStatusAuthError,
    BuildStatusUnavailableError,
    MalformedBuildStatusError,
)


class GbServerStatusReader:
    """Reads ``GET /api/v1/builds/{id}/status`` and normalizes it to a BuildState.

    Satisfies :class:`autotunex.services.reconcile.protocols.BuildStatusReader`.
    """

    def __init__(self, *, http_client: httpx.AsyncClient, base_url: str, token_env: str) -> None:
        self._http = http_client
        self._base_url = base_url.rstrip("/")
        self._token_env = token_env

    async def _get(self, build_id: UUID, resource: str) -> httpx.Response:
        """GET ``/api/v1/builds/{id}/{resource}`` and map HTTP failures to the taxonomy.

        Shared by :meth:`read` (``status``) and :meth:`read_events` (``events``) so
        both endpoints authenticate and classify 404/401/timeout identically.

        Raises:
            BuildNotFoundError: gbserver returned 404.
            BuildStatusAuthError: gbserver returned 401/403.
            BuildStatusUnavailableError: timeout, connection error, or other HTTP failure.
        """
        token = os.environ.get(self._token_env)
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            response = await self._http.get(
                f"{self._base_url}/api/v1/builds/{build_id}/{resource}",
                headers=headers,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code == 404:
                raise BuildNotFoundError(str(build_id)) from exc
            if code in (401, 403):
                raise BuildStatusAuthError() from exc
            raise BuildStatusUnavailableError(f"HTTP {code}") from exc
        except httpx.HTTPError as exc:
            # Timeouts and connection errors subclass HTTPError; retry next sweep.
            raise BuildStatusUnavailableError(str(exc)) from exc
        return response

    async def read(self, build_id: UUID) -> BuildState:
        """Return the build's normalized status.

        Raises:
            BuildNotFoundError: gbserver returned 404.
            BuildStatusAuthError: gbserver returned 401/403.
            BuildStatusUnavailableError: timeout, connection error, or other HTTP failure.
            MalformedBuildStatusError: a 2xx body lacking ``status.build.status``.
        """
        response = await self._get(build_id, "status")
        try:
            body: dict[str, Any] = response.json()
            build = body["status"]["build"]
            status = str(build["status"])
        except (ValueError, KeyError, TypeError) as exc:
            raise MalformedBuildStatusError(str(exc)) from exc
        return BuildState(
            build_id=build_id,
            status=status,
            failure_reason=build.get("failure_reason"),
            created_at=build.get("created_time"),
            updated_at=build.get("updated_time"),
            raw=body,
        )

    async def read_events(self, build_id: UUID) -> dict[str, Any]:
        """Return the build's raw event log (``GET /builds/{id}/events``).

        Feeds ``build_history`` in the Status tab. Kept separate from :meth:`read`
        so the loop pays this extra round trip only at the terminal transition,
        not on every sweep. Same error taxonomy as :meth:`read`.

        Raises:
            BuildNotFoundError: gbserver returned 404.
            BuildStatusAuthError: gbserver returned 401/403.
            BuildStatusUnavailableError: timeout, connection error, or other HTTP failure.
            MalformedBuildStatusError: a 2xx body that is not valid JSON.
        """
        response = await self._get(build_id, "events")
        try:
            body: dict[str, Any] = response.json()
        except ValueError as exc:
            raise MalformedBuildStatusError(str(exc)) from exc
        return body
