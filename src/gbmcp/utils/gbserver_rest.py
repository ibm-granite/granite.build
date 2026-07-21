"""Shared helpers for tools that call the gbserver REST API.

Centralizes URL/auth wiring so individual tool modules don't reinvent client
construction. Mirrors the pattern in `utils/sandbox.py`.
"""

import os

import httpx

GBSERVER_REST_URL = os.environ.get("GBSERVER_REST_URL", "")

API_BASE_PATH = "/api/v1"


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def get_client(token: str, timeout: float = 30.0) -> httpx.Client:
    """Return an httpx.Client pinned to the gbserver REST base URL.

    The client's base_url includes the `/api/v1` prefix, so callers pass paths
    like `/builds/{id}/files`. Bearer auth is set from the supplied token.
    """
    if not GBSERVER_REST_URL:
        raise RuntimeError("GBSERVER_REST_URL is not configured")
    return httpx.Client(
        base_url=f"{GBSERVER_REST_URL.rstrip('/')}{API_BASE_PATH}",
        headers=_headers(token),
        timeout=timeout,
    )
