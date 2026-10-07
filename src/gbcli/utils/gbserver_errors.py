"""HTTP error types raised by the CLI's gbserver client helpers.

Owned by gbcli (rather than reusing fastapi's ``HTTPException``) so the CLI does
not import the web framework at startup.
"""

from typing import Any

_RELOGIN_COMMANDS = {
    "github": "gb auth login",
    "ibmid": "gb auth login --sso ibm",
    "apikey": "gb auth login --gbserver",
}


def relogin_hint() -> str:
    """Re-login instruction for the configured auth provider.

    Bare ``gb auth login`` is the GitHub flow (and makes GitHub the default), so an
    IBMid or API-key user must be pointed at their own flow instead.
    """
    provider = "github"
    try:
        from gbcli.utils.gbcredentials import GBCredentials
        from gbcommon.types.gbenvconfig import is_standalone

        if is_standalone():
            provider = "apikey"
        else:
            provider = (
                GBCredentials().get("default_provider", section="user") or "github"
            )
    except Exception:  # credentials unreadable: fall back to the GitHub flow
        pass
    command = _RELOGIN_COMMANDS.get(provider, _RELOGIN_COMMANDS["github"])
    return f"Run '{command}' to re-authenticate."


class GBServerHTTPError(Exception):
    """A 4xx/5xx response from gbserver; mirrors ``HTTPException``'s fields."""

    def __init__(self, status_code: int, detail: Any = ""):
        super().__init__(f"{status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class GBServerAuthError(GBServerHTTPError):
    """gbserver rejected the credentials (401).

    The CLI no longer pre-validates the token against GitHub on every command, so
    this is where an expired or revoked token surfaces. The re-login hint is part
    of ``detail`` so it survives callers that re-wrap ``status_code``/``detail``.
    """

    def __init__(self, detail: Any = ""):
        # gbserver's detail is usually a string but may be any JSON value.
        text = str(detail).rstrip(". ") if detail else ""
        hint = relogin_hint()
        super().__init__(401, f"{text}. {hint}" if text else hint)


def gbserver_http_error(status_code: int, detail: Any = "") -> GBServerHTTPError:
    """Build the error for a failed gbserver response, picking the 401 subclass."""
    if status_code == 401:
        return GBServerAuthError(detail)
    return GBServerHTTPError(status_code, detail)
