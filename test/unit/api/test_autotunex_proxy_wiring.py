#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""root_api mounts the AutoTuneX proxy only when GBSERVER_ENABLE_AUTOTUNEX is set,
and then before the SPA static mount.

Importing root_api boots the app the same way test/unit/standalone/
test_regression_smoke.py does (TestClient(root_api)), so this is a supported
test-time import.
"""

import importlib
from contextlib import contextmanager

import httpx
from fastapi.testclient import TestClient


@contextmanager
def _root_api_with_autotunex(enabled: bool):
    """root_api as imported with the flag at `enabled`.

    The mount is decided at import, so flip the constant and reload root_api if
    the cached copy disagrees; on exit, restore both so nothing leaks into later
    tests in the same worker. Same approach as test_regression_smoke.py's
    analytics fixture.
    """
    import gbserver.api.root_api as root_api_mod
    import gbserver.types.constants as constants

    prev = constants.GBSERVER_ENABLE_AUTOTUNEX
    constants.GBSERVER_ENABLE_AUTOTUNEX = enabled
    reloaded = False
    try:
        if root_api_mod.GBSERVER_ENABLE_AUTOTUNEX != enabled:
            root_api_mod = importlib.reload(root_api_mod)
            reloaded = True
        yield root_api_mod.root_api
    finally:
        constants.GBSERVER_ENABLE_AUTOTUNEX = prev
        if reloaded:
            importlib.reload(root_api_mod)


def _mock_upstream(monkeypatch, calls):
    import gbserver.api.autotunex_proxy as proxy_mod

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"from": "autotunex-proxy"})

    monkeypatch.setattr(proxy_mod, "AUTOTUNEX_URL", "http://autotunex.test")
    monkeypatch.setattr(
        proxy_mod,
        "_client",
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def test_proxy_route_registered_before_static_mount(monkeypatch):
    """With the flag on, `/api/autotunex/*` reaches the proxy handler rather than
    the catch-all `"/"` static Mount + SPA-fallback 404 handler.

    include_router() never shows up as a flat `route.path` on root_api.routes
    (see test_regression_smoke.py::test_analytics_routes_included), so this
    checks by request behaviour, against a MockTransport so nothing needs to
    listen on localhost:8000.
    """
    calls = []
    _mock_upstream(monkeypatch, calls)

    with _root_api_with_autotunex(True) as app:
        response = TestClient(app).get("/api/autotunex/some/path")

    assert response.status_code == 200
    assert response.json() == {"from": "autotunex-proxy"}
    assert len(calls) == 1


def test_proxy_not_mounted_by_default(monkeypatch):
    """With the flag off, nothing is relayed: the request falls through to the
    SPA-fallback 404, and the upstream is never called."""
    calls = []
    _mock_upstream(monkeypatch, calls)

    with _root_api_with_autotunex(False) as app:
        response = TestClient(app).get("/api/autotunex/some/path")

    assert response.status_code == 404
    assert calls == []
