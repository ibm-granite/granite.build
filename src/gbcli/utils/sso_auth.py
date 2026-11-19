# SPDX-License-Identifier: Apache-2.0

"""SSO (OIDC) login for the CLI via gbserver proxy.

The CLI is a public client: gbserver adds the client secret when redeeming
the code, and the CLI's PKCE verifier proves it started the login.

The code reaches the CLI on a loopback listener (default) or, with
``--remote``, as a verification code the user pastes from gbserver's page.
The CLI then redeems it via gbserver's ``/oidc/token``.
"""

import base64
import hashlib
import html
import http.server
import json
import logging
import secrets
import socket
import socketserver
import sys
import time
import urllib.parse
import webbrowser
from typing import Callable, Optional

import requests
from pydantic import BaseModel

from gbcli.utils.gbconstants import GBSERVER_INSTANCE, SSO_REDIRECT_PORT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GBSERVER_AUTH_BASE = f"{GBSERVER_INSTANCE}/api/v1/auth"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SSOUserInfo(BaseModel):
    sub: str = ""
    name: str = ""
    email: str = ""
    preferred_username: str = ""


class AuthFlowResult(BaseModel):
    access_token: str
    id_token: str = ""
    refresh_token: str = ""
    expires_in: int = 0
    user_info: SSOUserInfo


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------


def _generate_code_verifier() -> str:
    """Generate a random code verifier (43-128 URL-safe characters)."""
    return secrets.token_urlsafe(64)[:96]


def _generate_code_challenge(verifier: str) -> str:
    """Compute the S256 code challenge from the verifier."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


# ---------------------------------------------------------------------------
# SSO (OIDC) client (proxy-based)
# ---------------------------------------------------------------------------

_LOOPBACK_DONE_HTML = (
    "<html><body>"
    "<h2>Sign-in received.</h2>"
    "<p>You may close this tab; the terminal shows whether login completed.</p>"
    "</body></html>"
)

_LOOPBACK_ERROR_HTML_TEMPLATE = (
    "<html><body>"
    "<h2>Authentication failed</h2>"
    "<p>{detail}</p>"
    "<p>Please close this tab and check the terminal.</p>"
    "</body></html>"
)


def _same_state(a: str, b: str) -> bool:
    # Bytes: compare_digest raises TypeError on non-ASCII str.
    return secrets.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def decode_remote_code(pasted: str, state: str) -> str:
    """The code in a pasted verification code; ValueError if malformed
    or from another login."""
    text = "".join(pasted.split())
    try:
        raw = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
        data = json.loads(raw.decode("utf-8"))
        pasted_state, code = data["s"], data["c"]
    except Exception as exc:
        raise ValueError("The verification code is not valid. Copy it again.") from exc
    if not _same_state(str(pasted_state), state):
        raise ValueError(
            "The verification code is from a different login. "
            "Use the code shown after opening the URL printed above."
        )
    return str(code)


class _LoopbackHandler(http.server.BaseHTTPRequestHandler):
    """Serves the one redirect to ``/callback``; anything else is 404."""

    server: "_LoopbackServer"
    timeout = 5  # bounds an idle connection's thread

    def do_GET(self):  # noqa: N802 - http.server naming
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/callback":
            self.send_error(404)
            return
        params = urllib.parse.parse_qs(parsed.query)
        state = params.get("state", [""])[0]
        code = params.get("code", [""])[0]
        error = params.get("error", [""])[0]
        error_description = params.get("error_description", [""])[0]
        if not _same_state(state, self.server.expected_state):
            # A stale tab or another page: reject it, keep waiting for ours.
            self._respond(
                _LOOPBACK_ERROR_HTML_TEMPLATE,
                "This sign-in is from a different login attempt.",
            )
        elif error or not code:
            detail = (
                f"The SSO provider returned an error: {error_description or error}"
                if error
                else "No authorization code received from the SSO provider."
            )
            self._respond(_LOOPBACK_ERROR_HTML_TEMPLATE, detail)
            self.server.error = detail
        else:
            self._respond(_LOOPBACK_DONE_HTML, "")
            self.server.code = code

    def _respond(self, template: str, detail: str) -> None:
        body = template.format(detail=html.escape(detail)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # keep the code out of the terminal
        pass


class _LoopbackServer(http.server.ThreadingHTTPServer):
    # Threaded, so an idle browser preconnect can't delay the real redirect.
    daemon_threads = True
    # POSIX: SO_REUSEADDR allows rebinding through TIME_WAIT (quick retries)
    # but never two live listeners. On Windows it would allow that, so use
    # SO_EXCLUSIVEADDRUSE instead.
    allow_reuse_address = sys.platform != "win32"

    def __init__(self, port: int, expected_state: str):
        super().__init__(("127.0.0.1", port), _LoopbackHandler)
        self.expected_state = expected_state
        self.code: Optional[str] = None
        self.error: Optional[str] = None

    def server_bind(self):
        if sys.platform == "win32":
            self.socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1  # type: ignore[attr-defined]
            )
        # Skip HTTPServer's getfqdn() lookup, which can stall on some networks.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


class SSOLoginClient:
    """Drives SSO (OIDC) login via gbserver proxy."""

    def __init__(self, gbserver_auth_base: str = ""):
        self.gbserver_auth_base = gbserver_auth_base or GBSERVER_AUTH_BASE

    def login_loopback(
        self,
        open_browser: Optional[Callable[[str], object]] = None,
        port: Optional[int] = None,
        timeout: int = 300,
    ) -> AuthFlowResult:
        """Log in via the SSO redirect to a local port; *open_browser* (default
        ``webbrowser.open``) gets the URL once the listener is up."""
        opener: Callable[[str], object] = open_browser or webbrowser.open
        port = port or SSO_REDIRECT_PORT
        verifier, state = _generate_code_verifier(), secrets.token_urlsafe(32)
        redirect_uri = f"http://127.0.0.1:{port}/callback"
        try:
            server = _LoopbackServer(port, state)
        except OSError as e:
            raise Exception(
                f"Cannot listen on 127.0.0.1:{port} for the SSO redirect ({e}). "
                "Free the port, or use 'gb auth login --sso --remote'."
            ) from e
        with server:
            server.timeout = 1
            opener(self._authorize_url(verifier, state, redirect_uri))
            deadline = time.time() + timeout
            while server.code is None and server.error is None:
                if time.time() >= deadline:
                    raise Exception(
                        "Timed out waiting for the SSO browser redirect. If "
                        "sign-in failed in the browser, try again; if the browser "
                        "can't reach this machine, use 'gb auth login --sso --remote'."
                    )
                server.handle_request()
        if server.error is not None:
            raise Exception(server.error)
        return self._redeem(server.code, verifier, redirect_uri)  # type: ignore[arg-type]

    def login_remote(
        self,
        show_url: Optional[Callable[[str], None]] = None,
        read_code: Optional[Callable[[], str]] = None,
        on_bad_code: Optional[Callable[[str], None]] = None,
        attempts: int = 3,
    ) -> AuthFlowResult:
        """Log in via a pasted verification code; *show_url* gets the URL,
        *read_code* returns the paste, and a bad paste is re-prompted."""
        show_url = show_url or print
        on_bad_code = on_bad_code or (lambda msg: print(msg, file=sys.stderr))
        read_code = read_code or (lambda: input("Paste the verification code: "))
        verifier, state = _generate_code_verifier(), secrets.token_urlsafe(32)
        show_url(self._authorize_url(verifier, state, None))
        for attempt in range(max(attempts, 1)):
            try:
                code = decode_remote_code(read_code(), state)
                break
            except ValueError as e:
                if attempt == attempts - 1:
                    raise
                on_bad_code(str(e))
        return self._redeem(code, verifier, None)

    def _authorize_url(
        self, verifier: str, state: str, redirect_uri: Optional[str]
    ) -> str:
        params = {
            "code_challenge": _generate_code_challenge(verifier),
            "code_challenge_method": "S256",
            "state": state,
        }
        if redirect_uri is not None:
            params["redirect_uri"] = redirect_uri
        return (
            f"{self.gbserver_auth_base}/oidc/authorize?"
            f"{urllib.parse.urlencode(params)}"
        )

    def _redeem(
        self, code: str, verifier: str, redirect_uri: Optional[str]
    ) -> AuthFlowResult:
        """Redeem the code via gbserver (no redirect_uri: --remote)."""
        payload = {"code": code, "code_verifier": verifier}
        if redirect_uri is not None:
            payload["redirect_uri"] = redirect_uri
        resp = requests.post(
            f"{self.gbserver_auth_base}/oidc/token", json=payload, timeout=60
        )
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise Exception(f"SSO authentication error: {detail}")
        data = resp.json()
        return AuthFlowResult(
            access_token=data["access_token"],
            id_token=data.get("id_token", ""),
            refresh_token=data.get("refresh_token", ""),
            expires_in=data.get("expires_in", 0),
            user_info=SSOUserInfo.model_validate(data.get("user_info") or {}),
        )
