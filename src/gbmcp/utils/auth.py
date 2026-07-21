from fastmcp.server.dependencies import get_access_token


def get_github_token() -> str | None:
    """Return the bearer token from the Authorization header, or None.

    gbmcp runs with **no auth verifier** in standalone (it's mounted inside
    gbserver standalone), so there is usually no access token at all. Callers get
    None, and against the local gbserver that's fine — localhost is served in
    unauthenticated apikey mode. Guarded so a missing/undisabled auth context
    never raises.
    """
    try:
        access_token = get_access_token()
    except Exception:
        return None
    return access_token.token if access_token is not None else None
