#!/usr/bin/env python3
# Copyright LLM.build Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""root_api wires the AutoTuneX proxy before the SPA static mount.

Importing root_api boots the app the same way test/unit/standalone/
test_regression_smoke.py does (TestClient(root_api)), so this is a supported
test-time import.
"""

import httpx
from fastapi.testclient import TestClient


def test_proxy_route_registered_before_static_mount(monkeypatch):
    """A request to `/api/autotunex/*` must reach the proxy handler, not be
    swallowed by the catch-all `"/"` static Mount + SPA-fallback 404 handler.

    root_api.include_router(autotunex_router) wraps the router in an internal
    _IncludedRouter that computes its effective path lazily, so (unlike a
    .mount()'d sub-app) it never appears as a flat `route.path` string on
    root_api.routes — see test/unit/standalone/test_regression_smoke.py::
    test_analytics_routes_included, which documents the identical behavior
    for the analytics routers and verifies by request behavior instead.
    This test follows the same approach, and mirrors test_autotunex_proxy.py's
    MockTransport pattern so the assertion is deterministic regardless of
    whether anything is actually listening on localhost:8000 in the
    environment running the test.

    If the proxy were registered *after* the static mount, the mount's
    catch-all would claim this request first, the file lookup would 404, and
    root_api's SPA-fallback 404 handler would return a JSON 404 (paths under
    /api/ are excluded from the HTML SPA shell) instead of the mocked
    response below.
    """
    import gbserver.api.autotunex_proxy as proxy_mod
    from gbserver.api.root_api import root_api

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"from": "autotunex-proxy"})

    monkeypatch.setattr(proxy_mod, "AUTOTUNEX_URL", "http://autotunex.test")
    monkeypatch.setattr(
        proxy_mod,
        "_client",
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    client = TestClient(root_api)
    response = client.get("/api/autotunex/some/path")

    assert response.status_code == 200
    assert response.json() == {"from": "autotunex-proxy"}
