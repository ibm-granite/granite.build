# coding=utf-8
# Copyright 2023-present International Business Machines Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lightweight REST client for AutoTuneX pre-registration.

Handles only the three pre-registration API calls needed before a Granite Build
is submitted. All runtime communication (logs, trials, status, results) is
handled by fm-tune's BufferedLogHandler at training time.
"""

import logging
import time
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF = 1.0  # seconds


class AutoTuneXAPIError(Exception):
    """Raised for non-retryable API errors (4xx)."""


class AutoTuneXAPI:
    """Minimal REST client for AutoTuneX pre-registration endpoints.

    Auth: the caller's email is sent on every request, both as the
    ``X-User-Email`` header and the ``email`` cookie, matching the bridge's
    ``get_current_user()`` dependency (header wins, cookie is the fallback).
    """

    def __init__(self, base_url: str, email: str):
        self.base_url = base_url.rstrip("/") + "/fmtune"
        self.session = requests.Session()
        self.session.cookies.set("email", email)
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "X-User-Email": email,
            }
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        retries: int = _DEFAULT_RETRIES,
        **kwargs,
    ) -> requests.Response:
        """Issue an HTTP request with retry on 5xx / connection errors."""
        url = f"{self.base_url}{path}"
        backoff = _DEFAULT_BACKOFF
        last_exc: Optional[Exception] = None

        for attempt in range(1, retries + 1):
            try:
                resp = self.session.request(method, url, **kwargs)
                if resp.status_code < 500:
                    return resp
                last_exc = AutoTuneXAPIError(
                    f"{method} {path} returned {resp.status_code}: {resp.text}"
                )
                logger.warning(
                    "Attempt %d/%d: server returned %d, retrying…",
                    attempt,
                    retries,
                    resp.status_code,
                )
            except requests.ConnectionError as exc:
                last_exc = exc
                logger.warning(
                    "Attempt %d/%d: connection error, retrying…",
                    attempt,
                    retries,
                )

            if attempt < retries:
                time.sleep(backoff)
                backoff *= 2

        raise last_exc  # type: ignore[misc]

    def _get(self, path: str, **kwargs) -> requests.Response:
        return self._request("GET", path, **kwargs)

    def _post(self, path: str, **kwargs) -> requests.Response:
        return self._request("POST", path, **kwargs)

    def _put(self, path: str, **kwargs) -> requests.Response:
        return self._request("PUT", path, **kwargs)

    @staticmethod
    def _raise_for_4xx(resp: requests.Response, context: str) -> None:
        if 400 <= resp.status_code < 500:
            raise AutoTuneXAPIError(f"{context}: {resp.status_code} — {resp.text}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Retrieve job details by ID."""
        resp = self._get(f"/api/job/{job_id}")
        self._raise_for_4xx(resp, f"get job {job_id}")
        return resp.json()

    def bootstrap(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Idempotently register config + dataset + job in one call.

        Sends the whole run descriptor to the bridge's /api/job/bootstrap, which
        resolves (or creates) the config, dataset, and job server-side and
        returns their ids. Replaces the prior find_or_create_* + create_job
        sequence.
        """
        resp = self._post("/api/job/bootstrap", json=payload)
        self._raise_for_4xx(resp, "bootstrap")
        result = resp.json()
        logger.info(
            "Bootstrap complete: job=%s config=%s dataset=%s created=%s",
            result.get("job_id"),
            result.get("config_id"),
            result.get("dataset_id"),
            result.get("created"),
        )
        return result


def get_user_details(base_url: str, build_id: Optional[str]) -> str:
    path = "/fmtune/api/user"
    print(f"Fetching user details for build_id: {build_id}")
    try:
        resp = requests.get(f"{base_url}{path}/{build_id}", timeout=10)
        result = resp.json()
        print(f"User details response: {result}")
        return result.get("user_email") or "builds@example.com"
    except Exception as e:
        logger.error(f"Error fetching user details: {e}")
        return "builds@example.com"
