"""Shared caller-identity resolution for gb_ui_backend routes.

gbserver's AuthMiddleware attaches the trusted user to
`request.state.data["user"]` before routes run (these routers are mounted
directly on gbserver's own app); the X-User-Email header is a fallback for
running this app standalone, outside gbserver, where there's no
AuthMiddleware at all. Every route that needs "who is calling this" resolves
it via resolve_identity() below rather than reimplementing the precedence,
so a future change (a new auth mode, a different final fallback) only has
to be made once.
"""

from __future__ import annotations

from fastapi import Request


def resolve_identity(request: Request, fallback: str = "standalone") -> str:
    """AuthMiddleware-trusted user -> X-User-Email header -> `fallback`.

    `fallback` differs by caller: most routes have no per-user identity at
    all in apikey/localhost mode and use the "standalone" sentinel; a
    rate-limiting caller can instead pass the client's IP, since
    limiting-by-IP is still meaningful even without a per-user identity.

    Must never be trusted from a client-supplied field — the trusted-user
    branch is what makes this safe to use for authorization/scoping, not
    just display.
    """
    user = getattr(request.state, "data", {}).get("user")
    if user is not None:
        return user.email
    return request.headers.get("x-user-email") or fallback
