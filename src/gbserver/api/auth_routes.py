#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Token exchange proxy for IBMid OIDC authentication.

The CLI is a public client and must not hold secrets. These endpoints
let gbserver act as the confidential client that holds the
``GBSERVER_IBMID_CLIENT_SECRET`` and performs the token exchange on
behalf of the CLI.

Stateless flow (``/oidc/*``), used by the CLI:
1. ``/oidc/authorize`` redirects to IBMid, passing the PKCE challenge through.
2. IBMid redirects with the code to the CLI's loopback listener or, for
   ``--remote``, to ``/oidc/remote-callback``, which shows it for pasting.
3. ``/oidc/token`` redeems the code with the client secret; IBMid checks the
   CLI's verifier.
Nothing is stored, so any worker or pod can serve any step.

Legacy polling flow, kept for older CLIs:
1. CLI opens browser to ``/authorize`` with a PKCE code_challenge and
   a random ``state``.
2. This endpoint redirects the browser to IBMid's authorize URL.
3. After the user authenticates, IBMid redirects to ``/callback``.
4. The callback exchanges the authorization code for tokens (using the
   server-side client_secret) and stores them in an in-memory session.
5. The CLI polls ``/status`` with the ``state`` and ``code_verifier``.
   Once the PKCE proof is verified, the tokens are returned (one-time).

The legacy flow's in-memory sessions assume a single worker process.
"""

import base64
import hashlib
import html
import json
import os
import re
import time
import urllib.parse
from typing import Optional

import requests
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

auth_api = FastAPI()

# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

SESSION_TTL_SECONDS = 300  # 5 minutes


class _AuthSession:
    __slots__ = (
        "code_challenge",
        "status",
        "created_at",
        "access_token",
        "id_token",
        "refresh_token",
        "expires_in",
        "user_info",
        "error_detail",
    )

    def __init__(self, code_challenge: str):
        self.code_challenge = code_challenge
        self.status = "pending"  # "pending" | "complete" | "error"
        self.created_at: float = time.time()
        self.access_token: str = ""
        self.id_token: str = ""
        self.refresh_token: str = ""
        self.expires_in: int = 0
        self.user_info: Optional[dict] = None
        self.error_detail: str = ""


_sessions: dict[str, _AuthSession] = {}


def _cleanup_expired() -> None:
    now = time.time()
    expired = [
        k for k, v in _sessions.items() if now - v.created_at > SESSION_TTL_SECONDS
    ]
    for k in expired:
        del _sessions[k]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AUTH_COMPLETE_HTML = (
    "<html><body>"
    "<h2>IBMid login successful.</h2>"
    "<p>You may close this tab and return to the terminal.</p>"
    "</body></html>"
)

_AUTH_ERROR_HTML_TEMPLATE = (
    "<html><body>"
    "<h2>Authentication failed</h2>"
    "<p>{detail}</p>"
    "<p>Please close this tab and try again.</p>"
    "</body></html>"
)


def _verify_pkce_s256(code_verifier: str, code_challenge: str) -> bool:
    """Verify that ``SHA256(code_verifier)`` matches ``code_challenge``."""
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    computed = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return computed == code_challenge


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@auth_api.get("/authorize")
def authorize(
    code_challenge: str = Query(...),
    code_challenge_method: str = Query(...),
    state: str = Query(...),
):
    """Start the IBMid OIDC flow via the proxy.

    Stores the PKCE challenge and redirects the browser to IBMid.
    """
    if code_challenge_method != "S256":
        return JSONResponse(
            status_code=400,
            content={"detail": "Only S256 code_challenge_method is supported."},
        )

    if state in _sessions:
        return JSONResponse(
            status_code=409,
            content={"detail": "state value already in use."},
        )

    # Read config at request time (not import time).
    client_id = os.getenv("GBSERVER_IBMID_CLIENT_ID", "")
    authorize_url = os.getenv(
        "GBSERVER_IBMID_AUTHORIZE_URL",
        "https://login.ibm.com/v1.0/endpoint/default/authorize",
    )
    callback_url = os.getenv("GBSERVER_IBMID_CALLBACK_URL", "")

    if not client_id:
        return JSONResponse(
            status_code=500,
            content={"detail": "GBSERVER_IBMID_CLIENT_ID is not configured."},
        )
    if not callback_url:
        return JSONResponse(
            status_code=500,
            content={"detail": "GBSERVER_IBMID_CALLBACK_URL is not configured."},
        )

    _sessions[state] = _AuthSession(code_challenge=code_challenge)

    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": callback_url,
        "scope": "openid profile email",
        "state": state,
    }
    ibmid_url = f"{authorize_url}?{urllib.parse.urlencode(params)}"
    return RedirectResponse(url=ibmid_url, status_code=302)


@auth_api.get("/callback")
def callback(
    state: str = Query(...),
    code: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    error_description: Optional[str] = Query(None),
):
    """IBMid redirects here after the user authenticates.

    Exchanges the authorization code for tokens and stores them in the
    session so the CLI can retrieve them via ``/status``.
    """
    _cleanup_expired()

    session = _sessions.get(state)
    if session is None:
        return HTMLResponse(
            content=_AUTH_ERROR_HTML_TEMPLATE.format(
                detail="Session not found or expired. Please try again."
            ),
            status_code=200,
        )

    if error:
        session.status = "error"
        session.error_detail = error_description or error
        return HTMLResponse(
            content=_AUTH_ERROR_HTML_TEMPLATE.format(
                detail=f"IBMid returned an error: {error_description or error}"
            ),
            status_code=200,
        )

    if not code:
        session.status = "error"
        session.error_detail = "No authorization code received."
        return HTMLResponse(
            content=_AUTH_ERROR_HTML_TEMPLATE.format(
                detail="No authorization code received from IBMid."
            ),
            status_code=200,
        )

    # Exchange the code for tokens (server-side, with client_secret).
    client_id = os.getenv("GBSERVER_IBMID_CLIENT_ID", "")
    client_secret = os.getenv("GBSERVER_IBMID_CLIENT_SECRET", "")
    token_url = os.getenv(
        "GBSERVER_IBMID_TOKEN_URL",
        "https://login.ibm.com/v1.0/endpoint/default/token",
    )
    callback_url = os.getenv("GBSERVER_IBMID_CALLBACK_URL", "")
    userinfo_url = os.getenv(
        "GBSERVER_IBMID_USERINFO_URL",
        "https://login.ibm.com/v1.0/endpoint/default/userinfo",
    )

    # Token exchange
    try:
        token_data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": callback_url,
            "client_id": client_id,
            "client_secret": client_secret,
        }
        token_resp = requests.post(token_url, data=token_data, timeout=30)
        if token_resp.status_code != 200:
            body = {}
            try:
                body = token_resp.json()
            except Exception:
                pass
            desc = body.get("error_description", body.get("error", token_resp.text))
            session.status = "error"
            session.error_detail = f"Token exchange failed: {desc}"
            logger.error(
                "IBMid token exchange failed (%s): %s", token_resp.status_code, desc
            )
            return HTMLResponse(
                content=_AUTH_ERROR_HTML_TEMPLATE.format(
                    detail="Token exchange with IBMid failed. Please try again."
                ),
                status_code=200,
            )

        tokens = token_resp.json()
        session.access_token = tokens.get("access_token", "")
        session.id_token = tokens.get("id_token", "")
        session.refresh_token = tokens.get("refresh_token", "")
        session.expires_in = tokens.get("expires_in", 0)
    except Exception as e:
        session.status = "error"
        session.error_detail = f"Token exchange error: {e}"
        logger.exception("IBMid token exchange error")
        return HTMLResponse(
            content=_AUTH_ERROR_HTML_TEMPLATE.format(
                detail="An error occurred during token exchange. Please try again."
            ),
            status_code=200,
        )

    # Fetch userinfo
    try:
        ui_resp = requests.get(
            userinfo_url,
            headers={"Authorization": f"Bearer {session.access_token}"},
            timeout=10,
        )
        ui_resp.raise_for_status()
        session.user_info = ui_resp.json()
    except Exception as e:
        logger.warning("Failed to fetch IBMid userinfo: %s", e)
        session.user_info = {}

    session.status = "complete"
    return HTMLResponse(content=_AUTH_COMPLETE_HTML, status_code=200)


@auth_api.get("/status")
def status(
    state: str = Query(...),
    code_verifier: Optional[str] = Query(None),
):
    """CLI polls this endpoint to retrieve tokens after the browser flow.

    Returns ``{"status": "pending"}`` until the callback completes.
    Once complete, the CLI must provide ``code_verifier`` to prove it
    initiated the flow (PKCE).  Tokens are returned once and the
    session is deleted.
    """
    _cleanup_expired()

    session = _sessions.get(state)
    if session is None:
        return JSONResponse(
            status_code=404,
            content={"detail": "Auth session not found or expired."},
        )

    if session.status == "pending":
        return {"status": "pending"}

    if session.status == "error":
        detail = session.error_detail
        del _sessions[state]
        return {"status": "error", "error": detail}

    # status == "complete"
    if not code_verifier:
        return JSONResponse(
            status_code=400,
            content={
                "detail": "code_verifier is required to retrieve completed auth result."
            },
        )

    if not _verify_pkce_s256(code_verifier, session.code_challenge):
        return JSONResponse(
            status_code=401,
            content={"detail": "PKCE verification failed."},
        )

    result = {
        "status": "complete",
        "access_token": session.access_token,
        "id_token": session.id_token,
        "refresh_token": session.refresh_token,
        "expires_in": session.expires_in,
        "user_info": session.user_info or {},
    }

    # One-time retrieval: delete the session.
    del _sessions[state]

    return result


# ---------------------------------------------------------------------------
# Stateless flow: /oidc/authorize, /oidc/remote-callback, /oidc/token
# ---------------------------------------------------------------------------

# The CLI's loopback listener (RFC 8252). IBMid's registration pins the port.
_LOOPBACK_REDIRECT_RE = re.compile(r"^http://127\.0\.0\.1:\d{1,5}/callback$")

_REMOTE_CALLBACK_PATH = "/oidc/remote-callback"

# Code collapsed so it stays off screen; Copy works regardless and expands it
# if the clipboard is unavailable. Braces doubled for str.format.
_REMOTE_CODE_HTML_TEMPLATE = (
    "<html><body>"
    "<h2>IBMid sign-in complete</h2>"
    "<p>Copy the verification code and paste it into the gb CLI that is "
    "waiting for it.</p>"
    '<p><button id="copy" type="button">Copy verification code</button> '
    '<span id="status"></span></p>'
    "<details><summary>Show code</summary>"
    '<pre id="code" style="white-space:pre-wrap;word-break:break-all;'
    'padding:8px;border:1px solid #888">{code}</pre></details>'
    "<p><b>Only paste it into a gb CLI you started yourself. Never share it "
    "with anyone.</b></p>"
    "<script>"
    "document.getElementById('copy').onclick = function () {{"
    "  var status = document.getElementById('status');"
    "  function manual() {{"
    "    document.querySelector('details').open = true;"
    "    status.textContent = 'Copy failed: copy the code below by hand.';"
    "  }}"
    "  try {{"
    "    navigator.clipboard"
    "      .writeText(document.getElementById('code').textContent)"
    "      .then(function () {{ status.textContent = 'Copied.'; }}, manual);"
    "  }} catch (e) {{ manual(); }}"
    "}};"
    "</script>"
    "</body></html>"
)


def _remote_callback_url() -> str:
    """The ``--remote`` redirect URI: the env override, else derived from the
    legacy callback URL so deployments need no new setting."""
    url = os.getenv("GBSERVER_IBMID_REMOTE_CALLBACK_URL", "")
    if url:
        return url
    legacy = os.getenv("GBSERVER_IBMID_CALLBACK_URL", "")
    if legacy.endswith("/callback"):
        return legacy[: -len("/callback")] + _REMOTE_CALLBACK_PATH
    return ""


def _resolve_redirect_uri(redirect_uri: Optional[str]) -> Optional[str]:
    """The redirect URI to use, or None if not allowed. Omitted means
    ``--remote``, so the CLI needn't know the registered URL."""
    remote = _remote_callback_url()
    if redirect_uri is None:
        return remote or None
    if _LOOPBACK_REDIRECT_RE.match(redirect_uri) or (remote and redirect_uri == remote):
        return redirect_uri
    return None


def encode_remote_code(state: str, code: str) -> str:
    """The string the user pastes: ``state`` plus ``code``, so the CLI can
    reject a code from another login."""
    raw = json.dumps({"s": state, "c": code}, separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode("utf-8")).rstrip(b"=").decode("ascii")


@auth_api.get("/oidc/authorize")
def oidc_authorize(
    code_challenge: str = Query(...),
    code_challenge_method: str = Query(...),
    state: str = Query(...),
    redirect_uri: Optional[str] = Query(None),
):
    """Redirect the browser to IBMid, passing the PKCE challenge through."""
    if code_challenge_method != "S256":
        return JSONResponse(
            status_code=400,
            content={"detail": "Only S256 code_challenge_method is supported."},
        )
    resolved = _resolve_redirect_uri(redirect_uri)
    if resolved is None:
        return JSONResponse(
            status_code=400, content={"detail": "redirect_uri is not allowed."}
        )
    client_id = os.getenv("GBSERVER_IBMID_CLIENT_ID", "")
    if not client_id:
        return JSONResponse(
            status_code=500,
            content={"detail": "GBSERVER_IBMID_CLIENT_ID is not configured."},
        )
    authorize_url = os.getenv(
        "GBSERVER_IBMID_AUTHORIZE_URL",
        "https://login.ibm.com/v1.0/endpoint/default/authorize",
    )
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": resolved,
        "scope": "openid profile email",
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    return RedirectResponse(
        url=f"{authorize_url}?{urllib.parse.urlencode(params)}", status_code=302
    )


@auth_api.get(_REMOTE_CALLBACK_PATH)
def oidc_remote_callback(
    state: str = Query(...),
    code: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    error_description: Optional[str] = Query(None),
):
    """``--remote`` landing page: show the code for the user to paste."""
    if error or not code:
        detail = (
            f"IBMid returned an error: {error_description or error}"
            if error
            else "No authorization code received from IBMid."
        )
        return HTMLResponse(
            content=_AUTH_ERROR_HTML_TEMPLATE.format(detail=html.escape(detail)),
            status_code=200,
        )
    return HTMLResponse(
        content=_REMOTE_CODE_HTML_TEMPLATE.format(
            code=html.escape(encode_remote_code(state, code))
        ),
        status_code=200,
    )


class OIDCTokenRequest(BaseModel):
    code: str
    code_verifier: str
    # Omitted for --remote.
    redirect_uri: Optional[str] = None


@auth_api.post("/oidc/token")
def oidc_token(body: OIDCTokenRequest):
    """Redeem the code with the client secret; IBMid checks the verifier."""
    redirect_uri = _resolve_redirect_uri(body.redirect_uri)
    if redirect_uri is None:
        return JSONResponse(
            status_code=400, content={"detail": "redirect_uri is not allowed."}
        )
    token_url = os.getenv(
        "GBSERVER_IBMID_TOKEN_URL",
        "https://login.ibm.com/v1.0/endpoint/default/token",
    )
    userinfo_url = os.getenv(
        "GBSERVER_IBMID_USERINFO_URL",
        "https://login.ibm.com/v1.0/endpoint/default/userinfo",
    )
    try:
        token_resp = requests.post(
            token_url,
            data={
                "grant_type": "authorization_code",
                "code": body.code,
                "redirect_uri": redirect_uri,
                "code_verifier": body.code_verifier,
                "client_id": os.getenv("GBSERVER_IBMID_CLIENT_ID", ""),
                "client_secret": os.getenv("GBSERVER_IBMID_CLIENT_SECRET", ""),
            },
            timeout=30,
        )
    except Exception:
        logger.exception("IBMid token exchange error")
        return JSONResponse(
            status_code=502, content={"detail": "Token exchange with IBMid failed."}
        )
    if token_resp.status_code != 200:
        body_json: dict = {}
        try:
            body_json = token_resp.json()
        except Exception:
            pass
        desc = body_json.get("error_description", body_json.get("error", ""))
        logger.error(
            "IBMid token exchange failed (%s): %s", token_resp.status_code, desc
        )
        # 4xx: IBMid rejected the code or verifier.
        status_code = 400 if 400 <= token_resp.status_code < 500 else 502
        return JSONResponse(
            status_code=status_code,
            content={
                "detail": f"Token exchange failed: {desc or token_resp.status_code}"
            },
        )

    tokens = token_resp.json()
    access_token = tokens.get("access_token", "")
    user_info: dict = {}
    try:
        ui_resp = requests.get(
            userinfo_url,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        ui_resp.raise_for_status()
        user_info = ui_resp.json()
    except Exception as e:
        logger.warning("Failed to fetch IBMid userinfo: %s", e)
    return {
        "access_token": access_token,
        "id_token": tokens.get("id_token", ""),
        "refresh_token": tokens.get("refresh_token", ""),
        "expires_in": tokens.get("expires_in", 0),
        "user_info": user_info,
    }
