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

/api/analytics/* is handled by gb_ui_backend's routers, included directly
into root_api (see gbserver/api/root_api.py) — not proxied from here.
"""

from __future__ import annotations

import os

from fastapi import APIRouter

frontend_router = APIRouter()


@frontend_router.get("/api/config")
async def get_config() -> dict:
    return {
        "environment": os.environ.get("APP_ENVIRONMENT", "STANDALONE"),
        "authProvider": "apikey",
    }


@frontend_router.get("/api/environments")
async def get_environments() -> list:
    return [{"id": "STANDALONE", "label": "Standalone", "url": "http://localhost:8080"}]
