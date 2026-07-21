"""Shared helpers for the sandbox tools.

Thin wrappers around the blue-sandbox portal HTTP API plus the SSE parser used
by sandbox_exec and the file-transfer helpers reused by sandbox_path_upload,
sandbox_path_download, and sandbox_path_cp.
"""

import json
import os
from pathlib import Path

import httpx
from fastmcp.utilities.logging import get_logger

logger = get_logger(__name__)

SANDBOX_PORTAL_URL = os.environ.get("SANDBOX_PORTAL_URL", "")


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _client(api_token: str) -> httpx.Client:
    return httpx.Client(
        base_url=SANDBOX_PORTAL_URL,
        headers=_headers(api_token),
        timeout=30,
    )


def _async_client(api_token: str, timeout: float | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=SANDBOX_PORTAL_URL,
        headers=_headers(api_token),
        timeout=timeout,
    )


def _exchange_token(github_token: str) -> str:
    resp = httpx.post(
        f"{SANDBOX_PORTAL_URL}/api/auth/exchange",
        headers={"Authorization": f"Bearer {github_token}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["api_token"]


async def _parse_sse_events(chunk_aiter):
    """Parse an async byte-chunk iterator into decoded JSON event dicts.

    Portal-api emits `data: {json}\n\n` frames. Other fields are ignored.
    """
    buffer = ""
    async for chunk in chunk_aiter:
        if isinstance(chunk, bytes):
            buffer += chunk.decode("utf-8", errors="replace")
        else:
            buffer += chunk
        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            for line in frame.splitlines():
                if line.startswith("data:"):
                    payload = line[len("data:") :].strip()
                    if not payload:
                        continue
                    try:
                        yield json.loads(payload)
                    except json.JSONDecodeError:
                        logger.debug(f"skipping non-JSON SSE frame: {payload!r}")


def _upload_bytes(
    client: httpx.Client,
    sandbox_name: str,
    data: bytes,
    remote_path: str,
    mode: int | None = None,
) -> dict:
    """Upload raw bytes to the sandbox. Returns the portal JSON response."""
    metadata: dict = {"path": remote_path}
    if mode is not None:
        metadata["mode"] = mode
    filename = Path(remote_path).name
    files = {
        "metadata": (None, json.dumps(metadata), "application/json"),
        "file": (filename, data, "application/octet-stream"),
    }
    resp = client.post(
        f"/api/sandboxes/{sandbox_name}/files/upload",
        files=files,
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def _download_bytes(
    client: httpx.Client,
    sandbox_name: str,
    remote_path: str,
) -> bytes:
    """Download a single file from the sandbox. Returns raw bytes."""
    chunks: list[bytes] = []
    with client.stream(
        "GET",
        f"/api/sandboxes/{sandbox_name}/files/download",
        params={"path": remote_path},
        timeout=300,
    ) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_bytes():
            chunks.append(chunk)
    return b"".join(chunks)
