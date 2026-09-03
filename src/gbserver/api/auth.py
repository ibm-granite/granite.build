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

import hmac
import os
import re
from datetime import timedelta
from typing import List, Optional, Self, Tuple

import requests
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from gbcommon.types.constants import (
    DEFAULT_GH_DOMAIN,
    get_gh_api_base,
)
from gbserver.api.auth_providers import (
    AuthProvider,
    build_provider_list,
    resolve_github_email,
)
from gbserver.types.auth import User
from gbserver.utils.logger import get_logger
from gbserver.utils.utils import get_time

logger = get_logger(__name__)

_LOCALHOST_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}

# Exact paths that never require authentication, regardless of auth mode.
# These are the top-level app's own Swagger/OpenAPI doc pages plus the API
# root welcome route (see root_api.read_root). Sub-app docs are matched by
# _PUBLIC_DOCS_RE below, not enumerated here.
#
# Next.js's static export also emits a 404/index.html, but it's unreachable
# in practice: root_api's SPA-fallback 404 handler always serves
# dashboard/index.html for unknown paths (see _spa_fallback), never
# 404/index.html, so there's no route to allow-list here.
_PUBLIC_EXACT_PATHS = frozenset(
    {
        "/",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/docs/oauth2-redirect",
    }
)

# Path prefixes that never require authentication. The Next.js static-export
# frontend (see root_api.py's StaticFiles mount, SPA fallback, and RSC .txt
# payload middleware) is served entirely under /dashboard/* and /_next/*.
# New pages under frontend/app/dashboard/ are covered automatically by the
# /dashboard prefix below; a new *top-level* page outside /dashboard needs a
# new prefix here. /api/v1/auth is the OIDC pre-auth login flow (see
# auth_routes.py) — deliberately public.
# /api/autotunex is the standalone reverse proxy to the AutoTuneX server, which
# enforces its own cookie auth (see api/autotunex_proxy.py) — gbserver must not
# require its own token here or it would block the proxy before forwarding.
_PUBLIC_PATH_PREFIXES = ("/api/v1/auth", "/api/autotunex", "/dashboard", "/_next")

# Every mounted sub-app owns its own Swagger/OpenAPI doc pages directly under
# its mount point (e.g. /api/v1/builds/docs, /api/v1/builds/openapi.json) — see
# openapi_security.enable_api. Match those exactly: a versioned mount root
# (/api/v1/<one-segment>) immediately followed by a doc endpoint and nothing
# else. Anchoring to a *single* mount segment before the doc name is what keeps
# this from matching a data route whose trailing path parameter merely happens
# to be "docs"/"redoc"/"openapi.json" — e.g. GET /api/v1/secrets/user_secrets/
# redoc (a secret literally named "redoc") has an extra segment and must stay
# authenticated. The `$` anchor forbids anything after the doc name.
_PUBLIC_DOCS_RE = re.compile(
    r"^/api/v\d+/[^/]+/(docs|docs/oauth2-redirect|openapi\.json|redoc)$"
)


def _is_public_path(path: str) -> bool:
    """Authenticate everything except this explicit allow-list.

    Deny-by-default: a path must be named here to skip auth. This is the
    inverse of an allow-everything-except-/api/ check — a future endpoint
    registered outside this allow-list requires auth by default instead of
    silently being public.

    Runs on every request, so checks are ordered cheapest-first: a set
    membership test, then string prefix tests (which short-circuit all
    frontend/static/login traffic), and only then the sub-app-docs regex —
    reached only by the small slice of paths still unresolved. The regex is
    compiled once at import (``_PUBLIC_DOCS_RE``); its ``^``-anchor already
    rejects non-matching paths in C, so no Python-level pre-filter is added
    (measured: a startswith/endswith guard around it is slower, not faster).
    """
    if path in _PUBLIC_EXACT_PATHS:
        return True
    if any(
        path == prefix or path.startswith(prefix + "/") or path.startswith(prefix + ".")
        for prefix in _PUBLIC_PATH_PREFIXES
    ):
        return True
    return _PUBLIC_DOCS_RE.match(path) is not None


def _make_synthetic_user(login: str) -> User:
    """Create a synthetic User object for API key / localhost auth."""
    return User(
        login=login,
        id=0,
        url="",
        html_url="",
        name=login,
        email=f"{login}@localhost",
        auth_provider="apikey",
    )


def _is_localhost(request: Request) -> bool:
    """Check if the request originates from a localhost address."""
    if request.client is None:
        return False
    return request.client.host in _LOCALHOST_HOSTS


def get_gh_user(token: str, domain: Optional[str] = None) -> Tuple[Optional[User], str]:
    """Get user info from GitHub.

    Calls the ``/user`` endpoint to retrieve the authenticated user's
    profile, including their email address.  The returned :class:`User`
    object is stored in ``request.state.data["user"]`` by
    :class:`AuthMiddleware` and its ``email`` field is used as the
    user identity for space-access checks.
    """
    if domain is None:
        domain = DEFAULT_GH_DOMAIN

    api_base = get_gh_api_base(domain)
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    try:
        response = requests.get(f"{api_base}/user", headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        user = User.model_validate(data)

        resolve_github_email(user, domain, headers)

        if not user.email:
            logger.warning(
                "GitHub /user returned no email for user %s; "
                "space-access checks may fail",
                user.login,
            )
        return (user, "")
    except Exception as e:
        return (None, f"{e}")


class AuthMiddleware(BaseHTTPMiddleware):
    """Check if the request is authenticated.

    Supports multiple authentication modes controlled by the
    ``GBSERVER_AUTH_MODE`` environment variable:

    * ``"apikey"``  – static API key or localhost access
    * ``"github"``  – GitHub Enterprise token (default)
    * ``"ibmid"``   – IBMid JWT token
    * ``"multi"``   – both GitHub and IBMid simultaneously
    """

    user_cache: dict[str, User]
    user_cache_lifetime: int = 60 * 10  # 10 minutes

    def __init__(self: Self, *args: tuple, **kwargs: dict) -> None:
        self.user_cache = {}
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    @staticmethod
    def _redact_headers(headers) -> dict:
        """Return a copy of *headers* with bearer tokens redacted."""
        out = dict(headers)
        auth = out.get("authorization", "")
        if auth.lower().startswith("bearer ") and len(auth) > 11:
            out["authorization"] = f"Bearer {auth[7:11]}...redacted"
        return out

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        logger.info(
            "auth middleware headers: %s", self._redact_headers(request.headers)
        )

        # Only GET/HEAD requests to the explicit allow-list skip auth — every
        # allow-listed path is GET-only by nature (docs, config bootstrap,
        # OIDC redirects, static/SPA serving), so this also catches a future
        # mutating endpoint accidentally registered under an otherwise-public
        # prefix (e.g. POST /dashboard/something).
        path = request.url.path
        # The AutoTuneX reverse proxy (api/autotunex_proxy.py) forwards to a server
        # that enforces its own auth, so gbserver must not gate ANY method there —
        # GET and mutating verbs alike. Other public prefixes stay GET/HEAD-only so a
        # stray mutating endpoint under them still requires auth.
        _is_autotunex_proxy = path == "/api/autotunex" or path.startswith(
            "/api/autotunex/"
        )
        if _is_public_path(path) and (
            request.method in ("GET", "HEAD") or _is_autotunex_proxy
        ):
            response = await call_next(request)
            return response

        # Read auth mode at request time (not import time) so that env vars
        # set after import (e.g. by the standalone command) are picked up.
        auth_mode = os.getenv("GBSERVER_AUTH_MODE", "github")

        if auth_mode == "apikey":
            return await self._dispatch_apikey(request, call_next)
        return await self._dispatch_oauth(request, call_next, auth_mode)

    async def _dispatch_apikey(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Authenticate using a static API key or allow localhost access."""
        api_key = os.getenv("GBSERVER_API_KEY", "")
        api_user = os.getenv("GBSERVER_API_USER", "standalone")

        auth_header = request.headers.get("authorization", "")

        if api_key:
            # API key is configured -- require a matching Bearer token
            if auth_header == "" or not auth_header.startswith("Bearer "):
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={
                        "detail": "Authorization header is missing/invalid!",
                    },
                )
            token = auth_header.removeprefix("Bearer ")
            if not hmac.compare_digest(token, api_key):
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={
                        "detail": "Invalid API key.",
                    },
                )
            user = _make_synthetic_user(api_user)
        else:
            # No API key configured -- allow localhost only
            if _is_localhost(request):
                user = _make_synthetic_user(api_user)
            else:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={
                        "detail": "No API key configured and request is not from localhost.",
                    },
                )

        request.state.data = {"user": user}
        logger.info("auth middleware user (apikey mode): %s", user)
        response = await call_next(request)
        return response

    async def _dispatch_oauth(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
        auth_mode: str,
    ) -> Response:
        """Authenticate using one of the registered OAuth / token providers."""
        auth_header = request.headers.get("authorization", "")
        if auth_header == "" or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "detail": "Authorization header is missing/invalid!",
                },
            )
        token = auth_header.removeprefix("Bearer ")

        providers: List[AuthProvider] = build_provider_list(auth_mode)

        # --- check cache (keyed by provider_name:token) ---
        cached_user: Optional[User] = None
        cached_key: Optional[str] = None

        for provider in providers:
            key = f"{provider.provider_name}:{token}"
            if key in self.user_cache:
                user = self.user_cache[key]
                curr_time = get_time()
                if curr_time - user.gbserver_created_at > timedelta(
                    seconds=self.user_cache_lifetime
                ):
                    logger.info(
                        "cached user expired, evicting (%s)", provider.provider_name
                    )
                    self.user_cache.pop(key)
                else:
                    logger.info("user found in cache (%s)", provider.provider_name)
                    cached_user = user
                    cached_key = key
                    break

        if cached_user is not None and cached_key is not None:
            request.state.data = {"user": cached_user}
            logger.info("auth middleware user (cached): %s", cached_user)
            response = await call_next(request)
            return response

        # --- not cached: identify + validate ---
        for provider in providers:
            if not provider.identify_token(token):
                continue

            logger.info("token identified as %s, validating...", provider.provider_name)
            user, error = provider.validate_token(token)  # type: ignore[assignment]
            if user is None:
                logger.error(
                    "auth middleware error (%s): %s", provider.provider_name, error
                )
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={
                        "detail": f"The token is invalid: {error}",
                    },
                )

            key = f"{provider.provider_name}:{token}"
            logger.info("saving user info to cache (%s)", provider.provider_name)
            self.user_cache[key] = user
            request.state.data = {"user": user}
            logger.info("auth middleware user: %s", user)
            response = await call_next(request)
            return response

        # No provider claimed the token
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "detail": "The token format is not recognized by any configured auth provider.",
            },
        )
