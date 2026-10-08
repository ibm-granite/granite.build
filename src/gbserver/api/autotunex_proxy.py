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
"""Reverse proxy for AutoTuneX API calls.

Forwards the dashboard's same-origin ``/api/autotunex/*`` requests to the AutoTuneX
server's ``/api/v1/*``, so the browser needs no CORS. Mirrors the ``next dev``
rewrite in frontend/next.config.ts. Mounted only when
``GBSERVER_ENABLE_AUTOTUNEX=true`` (root_api.py).

The prefix authenticates like ``/api/v1/*``. AutoTuneX itself defaults to no auth
(``auth_providers=["disabled"]``), so whoever gbserver admits here reaches an
unauthenticated API.
"""

from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from gbserver.types.constants import AUTOTUNEX_URL
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# AutoTuneX serves its resource routes under /api/v1.
_UPSTREAM_PREFIX = "/api/v1"
# Public path this proxy is mounted at; the browser side of the mapping.
_PUBLIC_PREFIX = "/api/autotunex"

# httpx sets Host from the URL. Accept-Encoding is dropped so httpx asks only for
# encodings it can decode (br/zstd need optional packages): the response's
# Content-Encoding is stripped on the assumption the body arrives decoded.
# Content-Length is kept -- see the body handling in proxy_autotunex. On the way
# back, StreamingResponse sets its own framing.
_DROP_REQUEST_HEADERS = {"host", "accept-encoding"}
_DROP_RESPONSE_HEADERS = {
    "content-length",
    "transfer-encoding",
    "connection",
    "content-encoding",
}

_PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]

# Bound a hung AutoTuneX without cutting off slow dataset uploads or archive
# downloads: read and write are per-chunk waits, not whole-transfer budgets.
_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=10.0)

router = APIRouter()

_client: "httpx.AsyncClient | None" = None

# Warn on the first outage, then log at debug until a request succeeds: without
# AutoTuneX, every build page's linked-job lookup fails here.
_upstream_unreachable_logged = False


def _get_client() -> httpx.AsyncClient:
    """Return the shared AsyncClient, creating it on first use."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=_TIMEOUT)
    return _client


def _rewrite_location(value: str) -> str:
    """Map an upstream Location back into ``/api/autotunex/*``.

    An absolute Location from the upstream (e.g. FastAPI's trailing-slash 307)
    names the upstream host, so the browser would follow it cross-origin. Rewrite
    the ``/api/v1`` prefix to ``/api/autotunex`` as a host-relative URL; leave
    Locations outside the API (e.g. an external auth redirect) alone.
    """
    parts = urlsplit(value)
    if parts.path == _UPSTREAM_PREFIX or parts.path.startswith(_UPSTREAM_PREFIX + "/"):
        new_path = _PUBLIC_PREFIX + parts.path[len(_UPSTREAM_PREFIX) :]
        return urlunsplit(("", "", new_path, parts.query, parts.fragment))
    return value


@router.api_route("/api/autotunex/{path:path}", methods=_PROXY_METHODS)
async def proxy_autotunex(request: Request, path: str) -> Response:
    # httpx removes `..` dot-segments when it parses a URL, so a raw `..` in `path`
    # could escape the /api/v1 mount: resolve first and refuse anything outside it.
    # The base is rstripped so a trailing slash on AUTOTUNEX_API_URL can't produce
    # `//api/v1`. InvalidURL (e.g. a NUL in the path) is not an httpx.RequestError,
    # so it is answered here with the same 400.
    try:
        upstream_url = httpx.URL(
            f"{AUTOTUNEX_URL.rstrip('/')}{_UPSTREAM_PREFIX}/{path}"
        )
    except httpx.InvalidURL:
        logger.warning("rejected malformed AutoTuneX proxy path: %r", path)
        return JSONResponse({"detail": "Invalid proxy path."}, status_code=400)
    if not upstream_url.path.startswith(_UPSTREAM_PREFIX + "/"):
        logger.warning("rejected AutoTuneX proxy path escaping the API mount: %r", path)
        return JSONResponse({"detail": "Invalid proxy path."}, status_code=400)

    # Pairs, not a dict: a header can repeat (Cookie split across lines is legal,
    # and normal over HTTP/2), and a dict would keep only the last value.
    fwd_headers = [
        (k, v)
        for k, v in request.headers.items()
        if k.lower() not in _DROP_REQUEST_HEADERS
    ]
    # The shared client keeps a process-wide cookie jar; always send an explicit
    # Cookie (possibly empty) so httpx never injects another user's session.
    if not any(k.lower() == "cookie" for k, _ in fwd_headers):
        fwd_headers.append(("cookie", ""))

    # Stream the body rather than buffering it, for multi-GB dataset uploads. The
    # client's Content-Length is passed through unchanged, so httpx does not switch
    # to chunked encoding. Attach a body only when the request declares one.
    declares_body = (
        request.headers.get("content-length") is not None
        or "transfer-encoding" in request.headers
    )
    content = request.stream() if declares_body else None

    client = _get_client()
    upstream_request = client.build_request(
        request.method,
        upstream_url,
        # tuple() here and for params: mypy rejects a list against httpx's declared
        # pair type (list is invariant). Duplicate keys are preserved either way.
        headers=tuple(fwd_headers),
        params=tuple(request.query_params.multi_items()),
        content=content,
    )
    global _upstream_unreachable_logged
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.RequestError:
        log = logger.debug if _upstream_unreachable_logged else logger.warning
        log("AutoTuneX upstream unreachable at %s", AUTOTUNEX_URL)
        _upstream_unreachable_logged = True
        return JSONResponse(
            {"detail": f"AutoTuneX server unreachable at {AUTOTUNEX_URL}"},
            status_code=502,
        )
    _upstream_unreachable_logged = False

    # Built from raw bytes, so a non-latin-1 value (a UTF-8 Content-Disposition
    # filename) passes through unchanged, and before the response exists, so a
    # failure here cannot leak the upstream connection. Only Location is decoded,
    # for the rewrite; latin-1 round-trips bytes exactly.
    resp_headers = [
        (
            k,
            (
                _rewrite_location(v.decode("latin-1")).encode("latin-1")
                if k.decode("latin-1").lower() == "location"
                else v
            ),
        )
        for k, v in upstream.headers.raw
        if k.decode("latin-1").lower() not in _DROP_RESPONSE_HEADERS
    ]
    response = StreamingResponse(
        upstream.aiter_bytes(),
        status_code=upstream.status_code,
        background=BackgroundTask(upstream.aclose),
    )
    # Assign raw_headers directly to preserve duplicates (e.g. multiple
    # Set-Cookie headers from the AutoTuneX login flow), which a dict would drop.
    response.raw_headers = resp_headers
    return response
