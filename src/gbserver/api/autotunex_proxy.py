#!/usr/bin/env python3

# Copyright LLM.build Authors
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
"""Reverse proxy for AutoTuneX (fm-tune) API calls.

In standalone mode the frontend is served by gbserver at the same origin, so
AutoTuneX calls arrive as same-origin ``/api/autotunex/*`` requests. This module
forwards them server-side to the AutoTuneX FastAPI server's ``/api/v1/*``
routes, so browser cookies flow with no CORS. Mirrors the ``next dev`` rewrite in
frontend/next.config.ts. The AutoTuneX server enforces its own cookie auth;
gbserver treats ``/api/autotunex/*`` as public (see auth._PUBLIC_PATH_PREFIXES).
"""

import os
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

AUTOTUNEX_URL = os.getenv("AUTOTUNEX_API_URL", "http://localhost:8000")
# AutoTuneX API v0.3.5 serves its resource routes under /api/v1 (was /fmtune/api).
_UPSTREAM_PREFIX = "/api/v1"
# Public path this proxy is mounted at; the browser side of the mapping.
_PUBLIC_PREFIX = "/api/autotunex"

# Headers we must not forward verbatim: httpx sets Host from the URL; the
# StreamingResponse sets its own framing on the way back. Content-Length is
# deliberately NOT dropped -- see the body handling in proxy_autotunex.
_DROP_REQUEST_HEADERS = {"host"}
_DROP_RESPONSE_HEADERS = {
    "content-length",
    "transfer-encoding",
    "connection",
    "content-encoding",
}

_PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]

# An unresponsive AutoTuneX (hung, not down) must not pin a gbserver worker
# forever. read/write are per-chunk waits rather than whole-transfer budgets, so
# a generous value still allows slow multipart dataset uploads and large
# result-archive downloads while bounding a truly stalled connection.
_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=10.0)

router = APIRouter()

_client: "httpx.AsyncClient | None" = None


def _get_client() -> httpx.AsyncClient:
    """Return the shared AsyncClient, creating it on first use."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=_TIMEOUT)
    return _client


async def aclose_client() -> None:
    """Close the shared AsyncClient (called from root_api shutdown)."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _rewrite_location(value: str) -> str:
    """Map an upstream Location back into the public ``/api/autotunex/*`` space.

    The upstream can emit an absolute Location built from its own host and its
    ``/api/v1`` mount — e.g. FastAPI's trailing-slash 307 (``/api/v1/jobs/`` ->
    ``http://localhost:8000/api/v1/jobs``), or any redirect that stays inside the
    API. Because the proxy drops the Host header, that host is the upstream's, so
    the browser would send the follow-up request cross-origin and hit CORS.
    Rewrite the ``/api/v1`` path prefix to ``/api/autotunex`` and return a
    host-relative URL so those requests come back through gbserver — regardless
    of the upstream scheme/host or a trailing slash on AUTOTUNEX_API_URL.
    Locations whose path is outside the upstream API space (e.g. an external auth
    redirect) are left untouched.
    """
    parts = urlsplit(value)
    if parts.path == _UPSTREAM_PREFIX or parts.path.startswith(_UPSTREAM_PREFIX + "/"):
        new_path = _PUBLIC_PREFIX + parts.path[len(_UPSTREAM_PREFIX) :]
        return urlunsplit(("", "", new_path, parts.query, parts.fragment))
    return value


@router.api_route("/api/autotunex/{path:path}", methods=_PROXY_METHODS)
async def proxy_autotunex(request: Request, path: str) -> Response:
    url = f"{AUTOTUNEX_URL}{_UPSTREAM_PREFIX}/{path}"
    fwd_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in _DROP_REQUEST_HEADERS
    }
    # The shared AsyncClient keeps a process-wide cookie jar. Without an
    # explicit Cookie header, httpx injects jar cookies captured from a PRIOR
    # proxied response, bleeding one user's AutoTuneX session onto another
    # user's cookie-less request. Always forward an explicit Cookie (the
    # browser's, or empty) so the jar is never consulted for injection.
    if not any(k.lower() == "cookie" for k in fwd_headers):
        fwd_headers["cookie"] = ""

    # Forward the body as a stream rather than reading it with request.body().
    # Buffering would fully materialize a multi-GB training-set upload in this
    # process (and again inside httpx) -- exactly the case the generous write
    # timeout above exists to support.
    #
    # The client's Content-Length is passed through: the bytes are relayed
    # unchanged so it stays accurate, and httpx honours an explicit
    # Content-Length instead of falling back to Transfer-Encoding: chunked,
    # which keeps the upstream wire format identical to the browser's.
    #
    # Only attach a body when the request declares one, so a bodyless GET/HEAD
    # is not sent with a spurious chunked encoding.
    declares_body = (
        request.headers.get("content-length") is not None
        or "transfer-encoding" in request.headers
    )
    content = request.stream() if declares_body else None

    client = _get_client()
    upstream_request = client.build_request(
        request.method,
        url,
        headers=fwd_headers,
        # tuple(), not the list multi_items() returns: httpx accepts either, but
        # list is invariant so mypy rejects list[tuple[str, str]] against the
        # wider pair type it declares. Duplicate keys are preserved either way.
        params=tuple(request.query_params.multi_items()),
        content=content,
    )
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.RequestError:
        logger.warning("AutoTuneX upstream unreachable at %s", AUTOTUNEX_URL)
        return JSONResponse(
            {"detail": f"AutoTuneX server unreachable at {AUTOTUNEX_URL}"},
            status_code=502,
        )

    resp_headers = [
        (k, _rewrite_location(v) if k.lower() == "location" else v)
        for k, v in upstream.headers.multi_items()
        if k.lower() not in _DROP_RESPONSE_HEADERS
    ]
    response = StreamingResponse(
        upstream.aiter_bytes(),
        status_code=upstream.status_code,
        background=BackgroundTask(upstream.aclose),
    )
    # Assign raw_headers directly to preserve duplicates (e.g. multiple
    # Set-Cookie headers from the AutoTuneX login flow), which a dict would drop.
    response.raw_headers = [
        (k.encode("latin-1"), v.encode("latin-1")) for k, v in resp_headers
    ]
    return response
