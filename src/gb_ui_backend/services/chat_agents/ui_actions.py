"""Framework-agnostic page-navigation registry and route builder.

This module has nothing to do with gbmcp — it never touches gbserver or build
state, and it can only ever produce a route from NAVIGABLE_ROUTES (the model
cannot invent or free-type a URL). Whether the user actually goes there is a
separate confirmation step in the frontend (frontend/components/ChatWidget.tsx)
— this module never navigates anything itself.

NAVIGABLE_ROUTES mirrors the routes actually present under frontend/app/ —
everything lives under /dashboard/*; there is no /plans or /infrastructure
route in this frontend. Adding a new route to the app means adding a matching
entry here, or the agent simply can't offer it — that fails safe, not
dangerous, if forgotten.
"""

from __future__ import annotations

from typing import TypedDict
from urllib.parse import parse_qsl


class RouteEntry(TypedDict):
    template: str
    params: list[str]
    description: str


NAVIGABLE_ROUTES: dict[str, RouteEntry] = {
    "dashboard": {
        "template": "/dashboard",
        "params": [],
        "description": "Summary tiles — recent builds, status counts",
    },
    "builds": {
        "template": "/dashboard/builds",
        "params": [],
        "description": "Build list with filters, pagination",
    },
    "build_detail": {
        # Not /dashboard/builds/{build_id} — that dynamic segment is only ever
        # statically generated as the literal "_" placeholder (see
        # generateStaticParams() in app/dashboard/builds/[buildId]/page.tsx).
        # BuildDetailPageClient.tsx reads the real ID exclusively from the
        # ?id= query param (useSearchParams(), never the route param), so a
        # client-side router.push() must target this query-param form
        # directly — the same convention ClientShell.tsx's own
        # useDeepLinkRedirect() falls back to for hard-loaded bookmark URLs.
        "template": "/dashboard/builds/_/?id={build_id}",
        "params": ["build_id"],
        "description": "Build detail. This is also where a build can be cancelled.",
    },
    "artifacts": {
        "template": "/dashboard/artifacts",
        "params": [],
        "description": "Artifact list",
    },
    "artifact_detail": {
        # See build_detail above — same "_" + ?id= query-param convention;
        # ArtifactDetailPageClient.tsx also reads the ID only via useSearchParams().
        "template": "/dashboard/artifacts/_/?id={artifact_id}",
        "params": ["artifact_id"],
        "description": "Artifact detail",
    },
    "analytics": {
        "template": "/dashboard/analytics",
        "params": [],
        "description": "Build status chart and failure trends",
    },
    "data_processing": {
        "template": "/dashboard/data-processing",
        "params": [],
        "description": "Data processing pipelines and datasets",
    },
}


class UnknownPageError(ValueError):
    """Raised when a page key isn't in NAVIGABLE_ROUTES."""


class MissingRouteParamsError(ValueError):
    """Raised when a route template's required params weren't all supplied."""


def build_navigation_route(page: str, reason: str, **params: str) -> dict[str, str]:
    """Resolve a page key + params into a concrete route + label.

    Raises UnknownPageError / MissingRouteParamsError on anything not in
    NAVIGABLE_ROUTES — the caller (an agent tool handler) can never produce an
    arbitrary URL through this function.
    """
    entry = NAVIGABLE_ROUTES.get(page)
    if entry is None:
        raise UnknownPageError(
            f"Unknown page {page!r}. Valid pages: {sorted(NAVIGABLE_ROUTES)}"
        )
    missing = [p for p in entry["params"] if p not in params]
    if missing:
        raise MissingRouteParamsError(
            f"Missing required params for {page!r}: {missing}"
        )
    route = entry["template"].format(**params)
    # ChatWidget.tsx only renders the ui_action card when `label` is
    # truthy — an empty/whitespace `reason` from the model used to produce
    # an empty label, silently dropping the card while the tool still
    # reported success back to the model (which then believed it had shown
    # the user an offer that never actually appeared).
    label = reason.strip() or f"Navigate to {page}"
    return {"route": route, "label": label}


# NAVIGABLE_ROUTES is static, so this reverse lookup (normalized template
# path -> entry) is computed once at import time rather than recomputed on
# every describe_current_page() call — a hot-ish path, called once per chat
# turn that carries page context.
_PATH_TO_ROUTE: dict[str, RouteEntry] = {
    entry["template"].split("?", 1)[0].rstrip("/") or "/": entry
    for entry in NAVIGABLE_ROUTES.values()
}


def describe_current_page(pathname: str, query_string: str = "") -> str:
    """Best-effort human-readable description of a frontend route the user is
    currently viewing, for passive context (see tool_loop_backend.py's
    _build_augmented_message()) — never used to authorize or perform
    anything, only to help the model resolve references like "this build".

    Deliberately reuses NAVIGABLE_ROUTES (via _PATH_TO_ROUTE above) rather
    than a second route table, so this can never drift out of sync with what
    suggest_navigation itself knows about. None of our route templates have
    a placeholder in the path segment (only in the query string, via
    build_detail/artifact_detail's "?id=" convention) — so matching is a
    plain path lookup, no regex/placeholder parsing needed.
    """
    normalized = pathname.rstrip("/") or "/"
    entry = _PATH_TO_ROUTE.get(normalized)
    if entry is None:
        return "An unrecognized page in the dashboard"

    params = dict(parse_qsl(query_string.lstrip("?")))
    if entry["params"] and "id" in params:
        return f"{entry['description']} (id={params['id']})"
    return entry["description"]
