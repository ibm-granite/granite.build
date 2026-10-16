# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Backend-for-frontend: browser sessions via W3ID authorization code + PKCE.

Stateless by design. ``state`` and the PKCE verifier live in a short-lived
signed cookie set by ``/login`` — never server memory, so this behaves
identically behind any number of uvicorn workers. ``client_secret`` is used
only in the server-side token exchange in ``/callback`` and never reaches
the browser.

Mounted unconditionally in ``main.py``, whether or not ``"session"`` is an
enabled provider: a 404-vs-401 difference here would let a caller probe which
providers a deployment has configured, so "not configured" has to look
*exactly* like "rejected" (``SECURITY.md`` records the same reasoning for the
unconditional OpenAPI security schemes). Both routes fail closed, but not
identically: most rejections in each raise the same opaque
``InvalidCredentialsError`` a genuinely wrong credential gets, but
``/callback`` delegates ID-token verification to ``OidcBearerVerifier``, which
raises the sibling ``ExpiredCredentialsError`` instead for an expired ID
token — a different detail and a different ``WWW-Authenticate``
``error_description``. Not exploitable (an attacker cannot make the
operator's own token endpoint hand back an expired ID token), just worth
knowing the failure shape is not perfectly uniform.

Every rejection below is logged — at WARNING, except the one case that is
routine and self-correcting (an expired flow cookie), which is INFO — so an
operator can tell causes apart even though every caller sees the same opaque
body. Nothing caller-influenced (credential, cookie value, token fragment,
``state``, ``code``, email) ever reaches a log line.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from typing import Annotated
from urllib.parse import urlencode
from uuid import UUID

import httpx
import jwt
from fastapi import APIRouter, Cookie, Depends
from fastapi.responses import JSONResponse, RedirectResponse, Response
from sqlalchemy.exc import SQLAlchemyError

from autotunex.api.deps import (
    PrincipalDep,
    SettingsDep,
    get_http_client,
    get_id_token_verifier,
    get_principal,
    get_user_repository,
)
from autotunex.core.auth.impersonation import mint_assume_token
from autotunex.core.auth.oidc import OidcBearerVerifier
from autotunex.core.auth.session import mint_session_token
from autotunex.core.config import Settings
from autotunex.core.exceptions import (
    AdminRequiredError,
    CannotImpersonateSelfError,
    ImpersonationUnavailableError,
    InvalidCredentialsError,
    UserNotFoundError,
)
from autotunex.core.logging import get_logger
from autotunex.db.repositories.protocols import UserRepository
from autotunex.models.auth import Principal

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

_FLOW_COOKIE = "oauth_flow"
_SESSION_COOKIE = "session"
_ASSUME_COOKIE = "autotunex_assume"
_FLOW_TTL = timedelta(minutes=5)
_FLOW_ALGORITHM = "HS256"


def _code_challenge(verifier: str) -> str:
    """Return the PKCE ``S256`` code challenge for ``verifier``."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


@dataclass(frozen=True)
class _LoginSettings:
    """The BFF settings ``/login`` needs, narrowed from ``Settings``'s Optional fields."""

    secret: str
    authorization_endpoint: str
    client_id: str
    base_url: str


def _require_login_settings(settings: Settings) -> _LoginSettings:
    """Fail closed unless the session provider is enabled and fully configured.

    This *is* the provider gate for ``/login``, not merely a type-narrowing
    check for mypy — that distinction matters. Checking only
    ``session_secret is None`` (as an earlier draft of this module did) misses
    the case where a deployment runs ``auth_providers=["oidc"]`` with a stray
    ``AUTOTUNEX_SESSION_SECRET`` left set in the environment: ``session_secret``
    would pass a bare not-None check while ``oidc_authorization_endpoint``,
    ``oidc_client_id`` and ``public_base_url`` are all still ``None``, and the
    handler would go on to build a redirect to the literal URL
    ``None?response_type=code&client_id=None&...`` — a real, if inert, ``302``
    the client sees. Checking ``"session" in settings.auth_providers`` first,
    and raising on it before any of the four fields is read, is what makes that
    scenario 401 instead. ``Settings``'s rule-4 validator guarantees the four
    fields below are non-empty and non-whitespace whenever ``"session"`` is
    enabled, so this is also, secondarily, the mypy-narrowing check that
    guarantee implies — both are true at once, which is why one function does
    both jobs rather than two.
    """
    if (
        "session" not in settings.auth_providers
        or settings.session_secret is None
        or settings.oidc_authorization_endpoint is None
        or settings.oidc_client_id is None
        or settings.public_base_url is None
    ):
        logger.warning(
            'Rejecting /auth/login: the "session" provider is not enabled, or its '
            "settings are incomplete. Check AUTOTUNEX_AUTH_PROVIDERS and the "
            "session/OIDC settings it requires."
        )
        raise InvalidCredentialsError()
    return _LoginSettings(
        secret=settings.session_secret.get_secret_value(),
        authorization_endpoint=settings.oidc_authorization_endpoint,
        client_id=settings.oidc_client_id,
        base_url=settings.public_base_url,
    )


@dataclass(frozen=True)
class _CallbackSettings:
    """The BFF settings ``/callback`` needs, narrowed from ``Settings``'s Optional fields."""

    secret: str
    token_endpoint: str
    client_id: str
    client_secret: str
    base_url: str


def _require_callback_settings(settings: Settings) -> _CallbackSettings:
    """Narrow every BFF setting ``/callback`` touches, or fail closed.

    Unlike :func:`_require_login_settings`, this does not repeat the
    ``"session" in settings.auth_providers`` check: ``/callback``'s provider
    gate is ``id_token_verifier is None`` (the dependency below, built from
    that exact condition in ``core.auth.registry.build_id_token_verifier``),
    already checked by the caller before this helper runs. So — unlike the
    login helper above, where the same claim would be false — this really is
    only a type-narrowing check: ``Settings``'s rule-4 validator guarantees
    every field below is set, non-empty and non-whitespace whenever
    ``"session"`` is enabled, and the gate above already established that it
    is. Kept as an explicit check (rather than an ``assert`` or a cast)
    because a request-time failure here should read as the same opaque 401
    every other rejection in this module does, not crash to a 500. Logged at
    WARNING like every other rejection in this module even though it should
    be unreachable in practice, so a race — or a future change to the gate
    above — leaves a trace instead of silently 401ing.
    """
    if (
        settings.session_secret is None
        or settings.oidc_token_endpoint is None
        or settings.oidc_client_id is None
        or settings.oidc_client_secret is None
        or settings.public_base_url is None
    ):
        logger.warning(
            "Rejecting /auth/callback: session/OIDC settings are incomplete despite "
            "the provider being enabled. Check AUTOTUNEX_SESSION_SECRET and the "
            "OIDC client/token settings."
        )
        raise InvalidCredentialsError()
    return _CallbackSettings(
        secret=settings.session_secret.get_secret_value(),
        token_endpoint=settings.oidc_token_endpoint,
        client_id=settings.oidc_client_id,
        client_secret=settings.oidc_client_secret.get_secret_value(),
        base_url=settings.public_base_url,
    )


def _clear_cookie(response: Response, key: str, settings: Settings) -> None:
    """Expire ``key`` with the same attributes it was set with.

    A cookie is matched for deletion by name, domain, and path alone (RFC 6265
    §5.3's storage-model replacement rule; Chromium and Firefox both key their
    cookie store the same three ways, and Chromium's own source comment notes
    that folding ``Secure``/``HttpOnly``/``SameSite`` into that key "would
    make sense, but the RFC doesn't specify this"). So ``delete_cookie(key)``
    alone would already clear this cookie — passing the rest of the
    attributes here is harmless hygiene, kept for symmetry with the
    ``set_cookie`` call that created it.
    """
    response.delete_cookie(
        key, httponly=True, secure=True, samesite=settings.session_cookie_same_site
    )


@router.get("/login", summary="Start the W3ID login flow")
async def login(settings: SettingsDep) -> RedirectResponse:
    """Redirect to W3ID's authorization endpoint with state and a PKCE (S256) challenge.

    ``redirect_uri`` is built from ``settings.public_base_url`` only, never
    from a request header: ``X-Forwarded-Host`` / ``X-Forwarded-Proto`` are
    attacker-controlled, and building the redirect target from them would let
    a poisoned value send the authorization code to a host of the attacker's
    choosing.
    """
    bff = _require_login_settings(settings)
    state = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    now = datetime.now(UTC)
    flow_token = jwt.encode(
        {"state": state, "verifier": verifier, "exp": int((now + _FLOW_TTL).timestamp())},
        bff.secret,
        algorithm=_FLOW_ALGORITHM,
    )
    redirect_uri = f"{bff.base_url}/auth/callback"
    params = {
        "response_type": "code",
        "client_id": bff.client_id,
        "redirect_uri": redirect_uri,
        "scope": "openid email",
        "state": state,
        "code_challenge": _code_challenge(verifier),
        "code_challenge_method": "S256",
    }
    response = RedirectResponse(
        f"{bff.authorization_endpoint}?{urlencode(params)}", status_code=HTTPStatus.FOUND
    )
    response.set_cookie(
        _FLOW_COOKIE,
        flow_token,
        max_age=int(_FLOW_TTL.total_seconds()),
        httponly=True,
        secure=True,
        samesite=settings.session_cookie_same_site,
    )
    return response


@router.get("/callback", summary="Complete the W3ID login flow")
async def callback(
    settings: SettingsDep,
    id_token_verifier: Annotated[OidcBearerVerifier | None, Depends(get_id_token_verifier)],
    http_client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
    user_repository: Annotated[UserRepository, Depends(get_user_repository)],
    code: str | None = None,
    state: str | None = None,
    oauth_flow: Annotated[str | None, Cookie(alias=_FLOW_COOKIE)] = None,
) -> RedirectResponse:
    """Validate ``state``, exchange the code, verify the ID token, and mint a session.

    ``id_token_verifier is None`` is the provider gate (``"session"`` not
    enabled); a missing ``oauth_flow`` cookie is the "no login in progress"
    case — a direct hit, a replay after the 5-minute flow TTL, or a stale
    cookie from before this cookie's claim shape changed. ``code``/``state``
    are optional at the signature level, not because either is genuinely
    optional to the flow, but so that a caller who clicked "Cancel" at
    W3ID — whose redirect back carries ``error=access_denied&state=...``
    with no ``code`` at all — fails through this same opaque 401 rather than
    FastAPI's request-validation layer returning a 422 before this function
    even runs, which would be a second, distinguishable "not configured"
    signal alongside the 404-vs-401 one this module's docstring already
    guards. The IdP's own ``error`` value is deliberately never read: it is
    caller-influenced text, and the absence of ``code`` is sufficient to
    detect and reject the case without logging any of it. All four failure
    reasons above resolve to the same opaque 401.
    """
    if id_token_verifier is None:
        logger.warning(
            'Rejecting /auth/callback: the "session" provider is not enabled. '
            "Check AUTOTUNEX_AUTH_PROVIDERS."
        )
        raise InvalidCredentialsError()
    if oauth_flow is None:
        logger.warning(
            "Rejecting /auth/callback: no oauth_flow cookie was presented. Likely a "
            "direct hit, a flow older than the 5-minute TTL, or a proxy stripping "
            "cookies before they reach this service."
        )
        raise InvalidCredentialsError()
    if code is None or state is None:
        logger.warning(
            "Rejecting /auth/callback: the redirect back from the IdP carried no code or state."
        )
        raise InvalidCredentialsError()
    bff = _require_callback_settings(settings)

    try:
        flow_claims = jwt.decode(
            oauth_flow,
            key=bff.secret,
            algorithms=[_FLOW_ALGORITHM],
            # Forces rejection of a signed-but-malformed cookie rather than
            # letting `.get("verifier")` return `None` and fail later with a
            # less specific error, or — before the isinstance checks below
            # existed — a bare `flow_claims["verifier"]` raising `KeyError`
            # and escaping as a 500. A stale cookie from before this claim
            # shape changed is exactly the case this guards.
            options={"require": ["exp", "state", "verifier"]},
        )
    # The split exists purely to log expiry at INFO rather than WARNING; it
    # does not change what is raised — both clauses below raise the same
    # `InvalidCredentialsError`, so the caller's response is identical
    # either way, only the server-side log level differs. Clause order is
    # load-bearing regardless: the installed PyJWT's hierarchy is
    # `ExpiredSignatureError -> InvalidTokenError -> PyJWTError` (checked
    # directly via `jwt.ExpiredSignatureError.__mro__`), and Python runs the
    # first `except` clause whose type matches — so if the general
    # `PyJWTError` clause below were listed first, it would catch
    # `ExpiredSignatureError` too and this INFO-level branch would never
    # run. INFO, not WARNING, because a flow cookie outliving its 5-minute
    # TTL is routine (the user took a while at the IdP, or came back to a
    # stale tab) and self-correcting (clicking "log in" again mints a fresh
    # one) — unlike every other rejection in this function.
    except jwt.ExpiredSignatureError:
        logger.info("Rejecting /auth/callback: the flow cookie has expired.")
        raise InvalidCredentialsError() from None
    except jwt.PyJWTError as exc:
        # Only the exception's class name is logged — see the `httpx.HTTPError`
        # handler below for why library exception text is never logged verbatim.
        logger.warning(
            "Rejecting /auth/callback: the flow cookie failed to decode: %s.",
            type(exc).__name__,
        )
        raise InvalidCredentialsError() from None

    flow_state = flow_claims.get("state")
    flow_verifier = flow_claims.get("verifier")
    if (
        not isinstance(flow_state, str)
        # Constant-time, not `!=`: matches the house precedent in
        # `core/auth/api_key.py` (`hmac.compare_digest`) for comparing a
        # server-held secret against caller-supplied input. Forging `state`
        # without the signing secret is not feasible either way — this cookie
        # is signed — so the practical value is defence-in-depth, not closing
        # a live hole.
        #
        # Compared as UTF-8 *bytes*, not `str`: `secrets.compare_digest`
        # raises `TypeError` when either `str` argument contains a
        # non-ASCII character. `api_key.py`'s call is safe from this because
        # both its arguments are hex digests, a fixed ASCII alphabet;
        # `state` here is a caller-supplied query parameter that Starlette
        # percent-decodes as UTF-8, so it can legitimately contain one. An
        # unauthenticated `GET /auth/callback?...&state=%C3%BC` with nothing
        # more than a live flow cookie would otherwise raise past this
        # handler's `except` blocks — all of which are later in this
        # function — and escape as a 500 with a full traceback logged, once
        # per request. Byte comparison has no such restriction regardless of
        # content, so mismatched non-ASCII input is simply unequal, not fatal.
        or not secrets.compare_digest(flow_state.encode("utf-8"), state.encode("utf-8"))
    ):
        logger.warning(
            "Rejecting /auth/callback: the flow cookie's state claim did not match "
            "the caller's, or was not a string at all (a stale cookie from before "
            "this claim shape changed). A mismatch is likely a proxy stripping "
            "cookies, a stale or replayed callback URL, or a flow cookie left over "
            "from a different login attempt."
        )
        raise InvalidCredentialsError()
    if not isinstance(flow_verifier, str) or not flow_verifier:
        logger.warning("Rejecting /auth/callback: the flow cookie carried no PKCE verifier.")
        raise InvalidCredentialsError()

    redirect_uri = f"{bff.base_url}/auth/callback"
    try:
        token_response = await http_client.post(
            bff.token_endpoint,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": bff.client_id,
                "client_secret": bff.client_secret,
                "code_verifier": flow_verifier,
            },
        )
    except httpx.HTTPError as exc:
        # Only the exception's class name is logged, never `str(exc)`. For an
        # unreachable or misbehaving token endpoint the class name — one of
        # ConnectError, ConnectTimeout, ReadTimeout, RemoteProtocolError — *is*
        # the diagnosis an operator needs; the message text on any of those can
        # embed content from whatever actually answered instead of the real
        # endpoint (an intercepting proxy, a misrouted ingress), and that text
        # is attacker-influenced in exactly the way `core/auth/oidc.py`'s
        # `_log_safe` was built to keep out of this project's logs.
        logger.warning("W3ID token exchange request failed: %s", type(exc).__name__)
        raise InvalidCredentialsError() from None

    if token_response.status_code != HTTPStatus.OK:
        logger.warning("W3ID token exchange failed with status %s", token_response.status_code)
        raise InvalidCredentialsError()

    try:
        token_body = token_response.json()
    except ValueError as exc:
        # `.json()` raises `json.JSONDecodeError` (a `ValueError`) on a
        # non-JSON body — an HTML error page from a misconfigured proxy, for
        # instance. Same reasoning as the `httpx.HTTPError` handler above:
        # only the type name is logged.
        logger.warning("W3ID token exchange returned a non-JSON body: %s", type(exc).__name__)
        raise InvalidCredentialsError() from None

    id_token = token_body.get("id_token") if isinstance(token_body, dict) else None
    if not isinstance(id_token, str):
        logger.warning("Rejecting /auth/callback: the token response carried no id_token.")
        raise InvalidCredentialsError()

    principal = await id_token_verifier.verify(id_token)
    # `OidcBearerVerifier` accepts any truthy string, so a whitespace-only
    # `email` claim (`"   "`) would otherwise pass `principal.email is None`
    # and get encoded unvalidated. `SessionCookieVerifier` strips and rejects
    # it on the next request, so the outcome without this `.strip()` is an
    # unusable cookie and a login loop, not a bypass — but minting a cookie
    # for an identity this codebase's own session verifier would reject
    # contradicts the discipline `session.py` documents at length.
    email = (principal.email or "").strip()
    if not email:
        logger.warning("Rejecting /auth/callback: the ID token's email claim was missing or empty.")
        raise InvalidCredentialsError()
    # A completed login is the one unambiguous "this person is here" event the
    # system gets, so it is recorded unconditionally — unlike the throttled
    # per-request refresh in `deps._record_activity`, which would otherwise skip
    # a fresh login that landed inside its window. Best-effort for the same
    # reason that one is: the caller has authenticated successfully, and failing
    # the login over a bookkeeping column would deny them the app entirely. A
    # caller with no `users` row (provisioning off) records nothing — the
    # repository treats that as a no-op, not an error.
    try:
        await user_repository.touch_login(email)
    except SQLAlchemyError as exc:
        logger.warning(
            "Could not record the last-login time for a completed login: %s. The "
            "login itself succeeded.",
            type(exc).__name__,
        )
    session_token = mint_session_token(
        email=email, secret=bff.secret, ttl_hours=settings.session_ttl_hours
    )

    response = RedirectResponse(bff.base_url, status_code=HTTPStatus.FOUND)
    _clear_cookie(response, _FLOW_COOKIE, settings)
    response.set_cookie(
        _SESSION_COOKIE,
        session_token,
        max_age=settings.session_ttl_hours * 3600,
        httponly=True,
        secure=True,
        samesite=settings.session_cookie_same_site,
    )
    return response


@router.get("/me", summary="The current principal")
async def me(principal: PrincipalDep) -> Principal:
    """Return whoever the caller's session (or other credential) resolves to."""
    return principal


@router.post("/assume/{user_id}", summary="Assume another user's identity (admin only)")
async def assume(
    user_id: UUID,
    real: Annotated[Principal, Depends(get_principal)],
    user_repository: Annotated[UserRepository, Depends(get_user_repository)],
    settings: SettingsDep,
) -> JSONResponse:
    """Begin impersonating ``user_id``.

    Gated on the *real* caller (``get_principal``, not the effective principal) so
    the control action can never be exercised through an existing impersonation as
    someone else. Preserves the admin's privileges — only the effective data-owner
    identity switches, resolved per request by ``get_effective_principal``.
    """
    if not real.is_admin:
        raise AdminRequiredError()
    if settings.session_secret is None:
        raise ImpersonationUnavailableError()
    if user_id == real.user_id:
        raise CannotImpersonateSelfError()
    target = await user_repository.get(user_id)
    if target is None:
        raise UserNotFoundError(user_id)
    token = mint_assume_token(
        user_id,
        secret=settings.session_secret.get_secret_value(),
        ttl_hours=settings.session_ttl_hours,
    )
    logger.info(
        "Impersonation started: impersonator=%s assumed_user_id=%s assumed_email=%s",
        real.email,
        target.id,
        target.email,
    )
    response = JSONResponse(
        {
            "message": "Impersonation started.",
            "assumed_user_id": str(target.id),
            "assumed_email": target.email,
        }
    )
    response.set_cookie(
        _ASSUME_COOKIE,
        token,
        max_age=settings.session_ttl_hours * 3600,
        httponly=True,
        secure=True,
        samesite=settings.session_cookie_same_site,
    )
    return response


@router.post("/unassume", summary="Stop impersonating and return to your own identity")
async def unassume(
    settings: SettingsDep,
    real: Annotated[Principal, Depends(get_principal)],
) -> JSONResponse:
    """Clear the impersonation overlay cookie. Idempotent — a no-op if not impersonating."""
    logger.info("Impersonation ended: impersonator=%s", real.email)
    response = JSONResponse({"message": "Impersonation ended."})
    _clear_cookie(response, _ASSUME_COOKIE, settings)
    return response


@router.post("/logout", summary="Clear the session cookie")
async def logout(settings: SettingsDep, _principal: PrincipalDep) -> JSONResponse:
    """Clear the session cookie and hand back the IdP's own logout endpoint.

    Requires authentication, so an unauthenticated caller cannot use it as an
    oracle for whether an IdP is configured at all. Clearing goes through
    ``_clear_cookie`` so path and domain match what the cookie was set with —
    those two are what a deletion is actually matched against (see
    ``_clear_cookie``'s docstring); passing ``Secure``/``SameSite`` too is
    hygiene, not a requirement.
    """
    response = JSONResponse({"end_session_endpoint": settings.oidc_end_session_endpoint})
    _clear_cookie(response, _SESSION_COOKIE, settings)
    _clear_cookie(response, _ASSUME_COOKIE, settings)
    return response
