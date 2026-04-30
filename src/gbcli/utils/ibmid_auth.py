"""IBMid OIDC authentication for the CLI via gbserver proxy.

The CLI is a public client and never holds the IBMid client_secret.
Instead, it delegates the token exchange to gbserver, which acts as
the confidential client.

Flow:
1. CLI generates a PKCE code_verifier/code_challenge and a random state.
2. CLI opens the browser to gbserver's ``/api/v1/auth/authorize``
   endpoint, passing the code_challenge and state.
3. gbserver redirects the browser to IBMid.  After the user
   authenticates, IBMid calls back to gbserver which performs the
   token exchange server-side.
4. CLI polls gbserver's ``/api/v1/auth/status`` with the state and
   code_verifier.  Once the PKCE proof is verified, gbserver returns
   the tokens (one-time).
"""

import base64
import hashlib
import logging
import secrets
import time
import urllib.parse
import webbrowser

import requests
from pydantic import BaseModel

from gbcli.utils.gbconstants import GBSERVER_INSTANCE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GBSERVER_AUTH_BASE = f"{GBSERVER_INSTANCE}/api/v1/auth"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class IBMidUserInfo(BaseModel):
    """I B Mid User Info implementation."""

    sub: str = ""
    name: str = ""
    email: str = ""
    preferred_username: str = ""


class AuthFlowResult(BaseModel):
    """Auth Flow Result implementation."""

    access_token: str
    id_token: str = ""
    refresh_token: str = ""
    expires_in: int = 0
    user_info: IBMidUserInfo


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
# IBMid OIDC client (proxy-based)
# ---------------------------------------------------------------------------


class IBMidOIDCClient:
    """Drives IBMid OIDC authentication via gbserver proxy."""

    def __init__(self, gbserver_auth_base: str = ""):
        self.gbserver_auth_base = gbserver_auth_base or GBSERVER_AUTH_BASE

    def start_auth_code_flow(
        self,
        timeout: int = 120,
        poll_interval: int = 3,
        open_browser=None,
    ) -> AuthFlowResult:
        """Run the proxy-based auth code + PKCE flow.

        Opens the browser pointing at gbserver's ``/authorize`` endpoint,
        then polls ``/status`` until tokens arrive or the flow times out.

        *open_browser* is a callback ``(url) -> None`` invoked with the
        authorize URL.  Defaults to ``webbrowser.open``.
        """
        if open_browser is None:
            open_browser = webbrowser.open

        code_verifier = _generate_code_verifier()
        code_challenge = _generate_code_challenge(code_verifier)
        state = secrets.token_urlsafe(32)

        # Build the gbserver authorize URL and open the browser.
        params = {
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
        }
        authorize_url = f"{self.gbserver_auth_base}/authorize?{urllib.parse.urlencode(params)}"

        open_browser(authorize_url)

        # Poll for completion.
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(poll_interval)
            try:
                resp = requests.get(
                    f"{self.gbserver_auth_base}/status",
                    params={"state": state, "code_verifier": code_verifier},
                    timeout=10,
                )
            except requests.ConnectionError:
                # Server might not be reachable yet; keep trying.
                continue

            if resp.status_code == 404:
                # Session may not exist yet (browser hasn't reached
                # /authorize) or it may have expired.  Either way, keep
                # polling until the timeout — the user may still be
                # navigating to the URL.
                continue
            if resp.status_code == 401:
                raise Exception("PKCE verification failed.")
            if resp.status_code == 400:
                raise Exception(resp.json().get("detail", "Bad request"))
            resp.raise_for_status()

            data = resp.json()

            if data["status"] == "pending":
                continue
            elif data["status"] == "error":
                raise Exception(f"IBMid authentication error: {data.get('error', 'unknown')}")
            elif data["status"] == "complete":
                user_info = IBMidUserInfo.model_validate(data.get("user_info", {}))
                return AuthFlowResult(
                    access_token=data["access_token"],
                    id_token=data.get("id_token", ""),
                    refresh_token=data.get("refresh_token", ""),
                    expires_in=data.get("expires_in", 0),
                    user_info=user_info,
                )

        raise Exception(
            "IBMid authentication timed out. " "Please try again with 'auth login --sso ibm'."
        )
