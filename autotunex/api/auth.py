# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlencode

import dependencies
import httpx
import jwt
import models as api
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from services import user_service

logger = logging.getLogger("auth")

# --- Configuration ---

CLIENT_ID = os.getenv("OIDC_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("OIDC_CLIENT_SECRET", "")
SECURITY_ENDPOINT = os.getenv("OIDC_SECURITY_ENDPOINT", "")

OIDC_ENABLED = bool(CLIENT_ID and CLIENT_SECRET and SECURITY_ENDPOINT)

TOKEN_URL = f"{SECURITY_ENDPOINT}/token" if SECURITY_ENDPOINT else ""
USERINFO_URL = f"{SECURITY_ENDPOINT}/userinfo" if SECURITY_ENDPOINT else ""
AUTHORIZATION_URL = f"{SECURITY_ENDPOINT}/authorize" if SECURITY_ENDPOINT else ""

SESSION_COOKIE = "autotunex_session"
STATE_COOKIE = "autotunex_oidc_state"
SESSION_TTL = timedelta(hours=8)

SESSION_SECRET = os.getenv("SESSION_SECRET", "")
if not SESSION_SECRET:
    SESSION_SECRET = secrets.token_urlsafe(32)
    logger.warning(
        "SESSION_SECRET not set — using random value (sessions won't survive restarts)"
    )


# --- JWT session cookie utilities ---


def create_session_token(
    email: str,
    role: str,
    impersonating: Optional[str] = None,
    impersonator: Optional[str] = None,
) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": email,
        "email": email,
        "role": role,
        "iat": now,
        "exp": now + SESSION_TTL,
    }
    if impersonating:
        payload["impersonating"] = impersonating
    if impersonator:
        payload["impersonator"] = impersonator
    return jwt.encode(payload, SESSION_SECRET, algorithm="HS256")


def decode_session_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, SESSION_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None


# --- OIDC flow functions ---


def _get_scheme(request: Request) -> str:
    return request.headers.get("x-forwarded-proto", request.url.scheme)


def _build_redirect_uri(request: Request) -> str:
    scheme = _get_scheme(request)
    host = request.headers.get("x-forwarded-host", request.headers.get("host", ""))
    return f"{scheme}://{host}/fmtune/api/auth/callback"


async def exchange_code_for_tokens(code: str, redirect_uri: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()
        return resp.json()


async def get_userinfo(access_token: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        resp.raise_for_status()
        return resp.json()


# --- FastAPI dependency ---


async def get_current_user(request: Request) -> api.AuthUser:
    from services.plugins import Seam, resolve

    return await resolve(Seam.AUTH).get_current_user(request)


# --- Auth router ---

auth_router = APIRouter(prefix="/api/auth", tags=["Auth"])


@auth_router.get("/login")
async def login(request: Request):
    if not OIDC_ENABLED:
        return {"authorization": "disabled"}

    state = secrets.token_urlsafe(32)
    redirect_uri = _build_redirect_uri(request)

    params = urlencode(
        {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": "openid profile email",
            "state": state,
        }
    )
    auth_url = f"{AUTHORIZATION_URL}?{params}"

    response = JSONResponse({"authorize": auth_url})
    response.set_cookie(
        STATE_COOKIE,
        state,
        httponly=True,
        samesite="lax",
        max_age=300,
        secure=_get_scheme(request) == "https",
    )
    return response


@auth_router.get("/callback")
async def callback(
    request: Request,
    code: str = "",
    state: str = "",
    grant_id: str = "",  # unused here, but required by the IBM w3id redirect
    user: user_service.User = Depends(dependencies.get_user_service),
):
    if not OIDC_ENABLED:
        raise HTTPException(400, "OIDC is not configured")

    expected_state = request.cookies.get(STATE_COOKIE)
    if not expected_state or expected_state != state:
        raise HTTPException(400, "Invalid state parameter")

    redirect_uri = _build_redirect_uri(request)

    try:
        tokens = await exchange_code_for_tokens(code, redirect_uri)
        user_info = await get_userinfo(tokens["access_token"])
    except httpx.HTTPStatusError as e:
        logger.exception("OIDC token exchange failed")
        raise HTTPException(502, f"OIDC provider error: {e.response.status_code}")
    except Exception as e:
        logger.exception("OIDC callback failed")
        raise HTTPException(502, f"OIDC callback failed: {e}")

    # IBM w3id uses "emailAddress", standard OIDC uses "email"
    email = user_info.get("emailAddress", user_info.get("email", ""))

    # Create or update user in database
    if await user.get_user(email) is None:
        await user.push_user(email)
    else:
        await user.touch_user_login(email)
    db_user = await user.get_user(email)
    role = db_user["role"]

    session_token = create_session_token(email=email, role=role)

    # Redirect to the app
    response = RedirectResponse(url="/autotune")
    response.set_cookie(
        SESSION_COOKIE,
        session_token,
        httponly=True,
        samesite="lax",
        max_age=int(SESSION_TTL.total_seconds()),
        secure=_get_scheme(request) == "https",
    )
    response.delete_cookie(STATE_COOKIE)
    return response


@auth_router.get("/me")
async def me(request: Request):
    if not OIDC_ENABLED:
        return {
            "authenticated": True,
            "oidc_enabled": False,
            "user": {"email": "dev@example.com", "role": "admin"},
        }

    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return {"authenticated": False, "oidc_enabled": True}

    claims = decode_session_token(token)
    if not claims:
        return {"authenticated": False, "oidc_enabled": True}

    return {
        "authenticated": True,
        "oidc_enabled": True,
        "user": {
            "email": claims.get("impersonating", claims["email"]),
            "role": claims["role"],
            "impersonating": claims.get("impersonating"),
            "impersonator": claims.get("impersonator"),
        },
    }


@auth_router.post("/logout")
async def logout():
    response = JSONResponse({"status": "logged_out"})
    response.delete_cookie(SESSION_COOKIE)
    return response


@auth_router.get("/assume/{user_id}")
async def assume(
    user_id: str,
    request: Request,
    user: user_service.User = Depends(dependencies.get_user_service),
):
    # Verify current user is admin
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    claims = decode_session_token(token)
    if not claims or claims.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Only administrators can assume roles.",
        )

    assumed_user = await user.get_user_by_id(user_id)
    if not assumed_user:
        raise HTTPException(status_code=404, detail="User not found")

    # Issue new JWT with impersonation claims. Preserve the admin's own role
    # so admin capabilities remain available while impersonating — only the
    # email context (via `impersonating`) switches to the assumed user.
    session_token = create_session_token(
        email=claims["email"],
        role=claims["role"],
        impersonating=assumed_user["email"],
        impersonator=claims["email"],
    )

    response = JSONResponse(
        {
            "detail": {
                "success": True,
                "message": "Successfully assumed user identity.",
                "assumed_email": assumed_user["email"],
                "assumed_role": assumed_user["role"],
                "effective_role": claims["role"],
            }
        }
    )
    response.set_cookie(
        SESSION_COOKIE,
        session_token,
        httponly=True,
        samesite="lax",
        max_age=int(SESSION_TTL.total_seconds()),
        secure=_get_scheme(request) == "https",
    )
    return response


@auth_router.get("/unassume")
async def unassume(
    request: Request,
    user: user_service.User = Depends(dependencies.get_user_service),
):
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    claims = decode_session_token(token)
    if not claims:
        raise HTTPException(status_code=401, detail="Session expired")

    impersonator = claims.get("impersonator")
    if not impersonator:
        raise HTTPException(status_code=400, detail="Not currently impersonating")

    # Reissue JWT as the original admin
    original_user = await user.get_user(impersonator)
    session_token = create_session_token(
        email=impersonator,
        role=original_user["role"],
    )

    response = JSONResponse(
        {
            "detail": {
                "success": True,
                "message": "Impersonation ended successfully",
            }
        }
    )
    response.set_cookie(
        SESSION_COOKIE,
        session_token,
        httponly=True,
        samesite="lax",
        max_age=int(SESSION_TTL.total_seconds()),
        secure=_get_scheme(request) == "https",
    )
    return response


@auth_router.post("/validate")
async def validate(request: Request):
    """Backward-compatible validation endpoint."""
    if not OIDC_ENABLED:
        return {"valid": True}

    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return {"valid": False}

    claims = decode_session_token(token)
    return {"valid": claims is not None}
