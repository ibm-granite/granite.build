"""HTTP error types raised by the CLI's gbserver client helpers.

Owned by gbcli (rather than reusing fastapi's ``HTTPException``) so the CLI does
not import the web framework at startup.
"""

RELOGIN_HINT = "Run 'gb auth login' to re-authenticate."


class GBServerHTTPError(Exception):
    """A 4xx/5xx response from gbserver; mirrors ``HTTPException``'s fields."""

    def __init__(self, status_code: int, detail: str = ""):
        super().__init__(f"{status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class GBServerAuthError(GBServerHTTPError):
    """gbserver rejected the credentials (401).

    The CLI no longer pre-validates the token against GitHub on every command, so
    this is where an expired or revoked token surfaces. The re-login hint is part
    of ``detail`` so it survives callers that re-wrap ``status_code``/``detail``.
    """

    def __init__(self, detail: str = ""):
        detail = f"{detail.rstrip('. ')}. {RELOGIN_HINT}" if detail else RELOGIN_HINT
        super().__init__(401, detail)


def gbserver_http_error(status_code: int, detail: str = "") -> GBServerHTTPError:
    """Build the error for a failed gbserver response, picking the 401 subclass."""
    if status_code == 401:
        return GBServerAuthError(detail)
    return GBServerHTTPError(status_code, detail)
