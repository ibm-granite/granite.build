"""IBM Cloud Logs API client — ported from gb_dashboard/services/cloud_logs.py."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 5.0
INITIAL_TIMEOUT = 15.0
MAX_TIMEOUT = 30.0


@dataclass
class CloudLogsClient:
    """Async client for IBM Cloud Logs API."""

    api_url: str
    api_key: str
    _token: str = field(default="", repr=False)
    _token_expiration: int = field(default=0, repr=False)
    _http: Optional[httpx.AsyncClient] = field(default=None, repr=False)

    def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient()
        return self._http

    async def get_token(self) -> str:
        if self._token and time.time() < self._token_expiration - 60:
            return self._token
        client = self._get_http()
        resp = await client.post(
            "https://iam.cloud.ibm.com/identity/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={self.api_key}",
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        self._token_expiration = data["expiration"]
        return self._token

    async def query_logs(
        self,
        build_id: str,
        page_size: int = 500,
        time_range: int = 5 * 24 * 3600,
        step_id: Optional[str] = None,
        container_name: Optional[str] = None,
        sort_asc: bool = True,
        timeout_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        token = await self.get_token()
        end_time = int(time.time() * 1000)
        start_time = end_time - (time_range * 1000)

        json_filter: Dict[str, Any] = {
            "kubernetes.labels.granite-dot-build/build-id": [build_id]
        }
        if step_id:
            json_filter["kubernetes.labels.granite-dot-build/build-step-id"] = [step_id]
        if container_name == "sidecar":
            json_filter["kubernetes.container_name"] = ["sidecar"]

        query = {
            "queryDef": {
                "startDate": start_time,
                "endDate": end_time,
                "pageSize": page_size,
                "pageIndex": 0,
                "type": "freeText",
                "queryParams": {"jsonObject": json_filter},
                "sortModel": [
                    {
                        "field": "timestamp",
                        "ordering": "asc" if sort_asc else "desc",
                        "missing": "_last",
                    }
                ],
            }
        }

        last_error: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            timeout = timeout_override or min(
                INITIAL_TIMEOUT * (1.5**attempt), MAX_TIMEOUT
            )
            try:
                resp = await self._get_http().post(
                    f"{self.api_url}/api/v1/logquery",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json=query,
                    timeout=timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < MAX_RETRIES - 1:
                    last_error = e
                    await asyncio.sleep(min(2.0 * (2**attempt), 10.0))
                else:
                    raise
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError) as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(
                        min(INITIAL_BACKOFF * (2**attempt), MAX_BACKOFF)
                    )
                else:
                    logger.error(
                        "Cloud Logs query failed after %d attempts: %s", MAX_RETRIES, e
                    )

        raise last_error  # type: ignore[misc]

    def parse_logs(
        self,
        response: Dict[str, Any],
        exclude_container: Optional[str] = None,
    ) -> List[str]:
        """Parse API response into log lines, oldest first."""
        logs = response.get("logs", [])
        lines = []

        for log in reversed(logs):
            if exclude_container:
                container_name = None
                kubernetes = log.get("kubernetes", {})
                if isinstance(kubernetes, dict):
                    container_name = kubernetes.get("container_name")
                if not container_name:
                    try:
                        text_json = json.loads(log.get("text") or "")
                        container_name = (text_json.get("kubernetes") or {}).get(
                            "container_name"
                        )
                    except (json.JSONDecodeError, TypeError):
                        pass
                if container_name == exclude_container:
                    continue

            text = log.get("text")
            if not text:
                continue
            try:
                log_line = json.loads(text).get("log", "")
                if log_line:
                    lines.append(log_line.rstrip("\n"))
            except (json.JSONDecodeError, TypeError):
                lines.append(text.rstrip("\n"))

        return lines


_client: Optional[CloudLogsClient] = None


def get_cloud_logs_client(api_url: str, api_key: str) -> CloudLogsClient:
    global _client
    if _client and _client.api_url == api_url and _client.api_key == api_key:
        return _client
    _client = CloudLogsClient(api_url=api_url, api_key=api_key)
    return _client
