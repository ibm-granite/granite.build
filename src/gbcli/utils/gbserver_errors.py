"""HTTP error types raised by the CLI's gbserver client helpers.

Owned by gbcli (rather than reusing fastapi's ``HTTPException``) so the CLI does
not import the web framework at startup.
"""

from typing import Any

RELOGIN_HINT = "Run 'gb auth login' to re-authenticate."


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
        super().__init__(401, f"{text}. {RELOGIN_HINT}" if text else RELOGIN_HINT)


def gbserver_http_error(status_code: int, detail: Any = "") -> GBServerHTTPError:
    """Build the error for a failed gbserver response, picking the 401 subclass."""
    if status_code == 401:
        return GBServerAuthError(detail)
    return GBServerHTTPError(status_code, detail)
