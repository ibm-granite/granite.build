"""Every route requires authentication except the documented allowlist.

Walks the live route tree rather than trusting that each router remembered to
add the dependency — router-level attachment is only actually safe once
something asserts it stays attached. Also guards mounted sub-apps: a Starlette
``Mount`` does not inherit router-level dependencies, so a future ``/mcp`` mount
would otherwise be invisible to a test that only inspects ``APIRoute``s.

Two facts about FastAPI 0.140.9 shape this file, and both were checked against
the installed version rather than assumed:

* ``app.routes`` is **not** flat. ``include_router`` leaves a lazy
  ``fastapi.routing._IncludedRouter`` placeholder behind, and the included
  ``APIRoute``s are not in ``app.routes`` at all. A loop over ``app.routes``
  looking for ``APIRoute`` finds none of the job endpoints and passes while
  asserting nothing. ``iter_route_contexts`` is FastAPI's own public helper for
  flattening that, and the context it yields exposes the *effective* dependency
  tree — router-level ``dependencies=[...]`` included.
* FastAPI creates **four** documentation routes, not three:
  ``/openapi.json``, ``/docs``, ``/docs/oauth2-redirect`` and ``/redoc``. They
  are plain ``starlette.routing.Route`` objects with no ``dependant`` at all, so
  they are exempt structurally and simply have to be named. The spec's §5 list
  omits ``/docs/oauth2-redirect``; it belongs to the same Swagger-UI machinery
  and exposes no data.
"""

from __future__ import annotations

from fastapi.routing import APIRoute, iter_route_contexts
from starlette.routing import Mount

from autotunex.api.deps import get_principal
from autotunex.main import create_app
from tests.conftest import make_settings

_ALLOWLIST = frozenset(
    {
        "/",
        "/health",
        "/health/live",
        "/health/ready",
        "/api/v1/app-config",
        "/docs",
        "/docs/oauth2-redirect",
        "/redoc",
        "/openapi.json",
        "/auth/login",
        "/auth/callback",
    }
)
"""Paths that legitimately carry no principal dependency.

``/`` because it is a bare convenience redirect to ``/docs`` that exposes no
data — gating it would only mean a browser cannot even be bounced to a public
docs page. ``/health``, ``/health/live`` and ``/health/ready`` because an
orchestrator's liveness and readiness probes cannot present a credential — and
``/health/ready`` deliberately fails closed on its own terms, returning a 503
when the database is unreachable rather than depending on this walk;
``/api/v1/app-config`` because the frontend needs it before and independent of
any user request; the four doc routes because they publish schema, not data, and
gating ``/docs`` would break the Authorize workflow — you cannot log in through
a page you cannot load; ``/auth/login`` and ``/auth/callback`` because you cannot
be logged in while logging in — they fail closed on their own (a missing or
disabled session provider is an opaque 401, never a 404), just not through this
dependency.
"""


def _dependant_uses(dependant: object, target: object) -> bool:
    if getattr(dependant, "call", None) is target:
        return True
    return any(_dependant_uses(sub, target) for sub in getattr(dependant, "dependencies", []))


def test_every_route_requires_authentication_except_the_documented_allowlist() -> None:
    app = create_app(make_settings())

    checked: list[str] = []
    failures: list[str] = []
    for context in iter_route_contexts(app.routes):
        path = context.path
        original = context.original_route
        if isinstance(original, Mount):
            # KNOWN GAP, deliberate and load-bearing: this branch enforces only
            # that a mount's *path* is allowlisted. It cannot see inside the
            # sub-app, and a Starlette ``Mount`` does not inherit the
            # router-level ``Depends(get_principal)``. So adding a future
            # ``/mcp`` mount to ``_ALLOWLIST`` turns this test green while
            # leaving the mounted sub-app completely unauthenticated. Whoever
            # adds the first mount owns authenticating it inside the sub-app and
            # owns extending this test to assert that — allowlisting the path is
            # not the fix.
            if path not in _ALLOWLIST:
                failures.append(f"unguarded mount: {path}")
            continue
        if path in _ALLOWLIST:
            continue
        if not isinstance(original, APIRoute):
            failures.append(f"unexpected unguarded route type: {path}")
            continue
        dependant = getattr(context, "dependant", None)
        if dependant is None:
            failures.append(f"route with no dependant: {path}")
            continue
        if not _dependant_uses(dependant, get_principal):
            failures.append(f"unprotected route: {path}")
            continue
        checked.append(path or "")

    # Collected rather than asserted per iteration: a regression that spans two
    # routers must name both, not halt on whichever the walk reached first.
    assert not failures, f"routes failing the authentication walk: {failures}"
    assert checked, "the walk found no protected routes — it is asserting nothing"


def test_the_walk_actually_reaches_the_job_endpoints() -> None:
    """Guards the guard.

    If a FastAPI upgrade changes how included routers are flattened, the test
    above degrades to vacuously green. This pins the two paths that must be in
    its coverage.
    """
    app = create_app(make_settings())

    paths = {context.path for context in iter_route_contexts(app.routes)}

    assert "/api/v1/jobs" in paths
    assert "/api/v1/jobs/{job_id}" in paths


def test_health_has_no_dependant_on_get_principal() -> None:
    app = create_app(make_settings())

    health = [c for c in iter_route_contexts(app.routes) if c.path == "/health"]

    assert len(health) == 1
    assert not _dependant_uses(getattr(health[0], "dependant", None), get_principal)
