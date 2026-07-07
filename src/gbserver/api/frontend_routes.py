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

"""Routes that support the gb-ui frontend (standalone mode).

Public endpoints (no auth required — see AuthMiddleware):
  GET  /api/config        — runtime config for frontend bootstrap
  GET  /api/environments  — always returns the single STANDALONE entry

Proxy endpoint:
  *    /api/analytics/{path}  — forwards to the gb_ui_backend sidecar at :8090
"""

from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

frontend_router = APIRouter()

_FORWARDED_HEADERS = {"authorization", "content-type", "accept", "x-request-id"}


@frontend_router.get("/api/config")
async def get_config() -> dict:
    return {
        "environment": os.environ.get("APP_ENVIRONMENT", "STANDALONE"),
        "authProvider": "apikey",
    }


@frontend_router.get("/api/environments")
async def get_environments() -> list:
    return [{"id": "STANDALONE", "label": "Standalone", "url": "http://localhost:8080"}]


@frontend_router.api_route(
    "/api/analytics/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE"],
)
async def analytics_proxy(path: str, request: Request) -> Response:
    """Proxy requests to the gb_ui_backend analytics sidecar."""
    sidecar = os.environ.get("GBSERVER_ANALYTICS_URL", "http://localhost:8090")
    qs = f"?{request.url.query}" if request.url.query else ""
    target = f"{sidecar}/api/analytics/{path}{qs}"
    headers = {k: v for k, v in request.headers.items() if k.lower() in _FORWARDED_HEADERS}

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.request(
                method=request.method,
                url=target,
                headers=headers,
                content=await request.body(),
            )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )
    except httpx.ConnectError:
        return JSONResponse(
            {"detail": "Analytics sidecar unavailable. Start gb_ui_backend to enable analytics."},
            status_code=503,
        )
