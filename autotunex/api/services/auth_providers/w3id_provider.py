# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""w3id (OIDC) auth provider — the production session-cookie path.

Reuses the shared JWT decode + cookie name from auth.py; does NOT duplicate
JWT logic. Behavior is verbatim from auth.get_current_user's OIDC branch,
including impersonation precedence.
"""

import logging

from fastapi import HTTPException, status

import models as api

from services.auth_providers.base import AuthProvider

logger = logging.getLogger(__name__)


class W3idAuthProvider(AuthProvider):
    async def get_current_user(self, request):
        from auth import SESSION_COOKIE, decode_session_token  # shared infra

        token = request.cookies.get(SESSION_COOKIE)
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
            )
        claims = decode_session_token(token)
        if not claims:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired or invalid",
            )
        return api.AuthUser(
            email=claims.get("impersonating", claims["email"]),
            role=api.Roles(claims["role"]),
            impersonating=claims.get("impersonating"),
        )
