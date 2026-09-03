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

import gzip

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import gbserver.api.autotunex_proxy as proxy_mod
from gbserver.api.autotunex_proxy import router


def _client_with_handler(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_forwards_method_path_query_and_cookie(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["cookie"] = request.headers.get("cookie")
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(proxy_mod, "AUTOTUNEX_URL", "http://autotunex.test")
    monkeypatch.setattr(proxy_mod, "_client", _client_with_handler(handler))

    client = TestClient(_make_app())
    resp = client.get(
        "/api/autotunex/jobs/by-build-id/abc",
        params={"scope": "own"},
        headers={"cookie": "session=xyz"},
    )

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert seen["method"] == "GET"
    assert seen["url"] == (
        "http://autotunex.test/api/v1/jobs/by-build-id/abc?scope=own"
    )
    assert seen["cookie"] == "session=xyz"


def test_streams_response_status_and_content_type(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401, json={"detail": "unauthorized"}, headers={"x-trace": "t1"}
        )

    monkeypatch.setattr(proxy_mod, "AUTOTUNEX_URL", "http://autotunex.test")
    monkeypatch.setattr(proxy_mod, "_client", _client_with_handler(handler))

    client = TestClient(_make_app())
    resp = client.post("/api/autotunex/jobs", json={"a": 1})

    assert resp.status_code == 401
    assert resp.headers["x-trace"] == "t1"
    assert resp.json() == {"detail": "unauthorized"}


def test_returns_502_when_upstream_unreachable(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    monkeypatch.setattr(proxy_mod, "AUTOTUNEX_URL", "http://autotunex.test")
    monkeypatch.setattr(proxy_mod, "_client", _client_with_handler(handler))

    client = TestClient(_make_app())
    resp = client.get("/api/autotunex/jobs")

    assert resp.status_code == 502
    assert "unreachable" in resp.json()["detail"]


def test_forwards_duplicate_query_params(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(proxy_mod, "AUTOTUNEX_URL", "http://autotunex.test")
    monkeypatch.setattr(proxy_mod, "_client", _client_with_handler(handler))

    client = TestClient(_make_app())
    resp = client.get(
        "/api/autotunex/jobs",
        params=[("tag", "a"), ("tag", "b"), ("x", "1")],
    )

    assert resp.status_code == 200
    assert "tag=a" in seen["url"]
    assert "tag=b" in seen["url"]


def test_content_encoding_dropped_and_body_decoded(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        body = gzip.compress(b'{"v":1}')
        return httpx.Response(
            200,
            content=body,
            headers={"content-encoding": "gzip", "content-type": "application/json"},
        )

    monkeypatch.setattr(proxy_mod, "AUTOTUNEX_URL", "http://autotunex.test")
    monkeypatch.setattr(proxy_mod, "_client", _client_with_handler(handler))

    client = TestClient(_make_app())
    resp = client.get("/api/autotunex/jobs")

    assert resp.status_code == 200
    assert resp.json() == {"v": 1}
    assert "content-encoding" not in resp.headers


def test_preserves_multiple_set_cookie_headers(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers=[("set-cookie", "a=1"), ("set-cookie", "b=2")]
        )

    monkeypatch.setattr(proxy_mod, "AUTOTUNEX_URL", "http://autotunex.test")
    monkeypatch.setattr(proxy_mod, "_client", _client_with_handler(handler))

    client = TestClient(_make_app())
    resp = client.get("/api/autotunex/jobs")

    assert resp.status_code == 200
    assert resp.headers.get_list("set-cookie") == ["a=1", "b=2"]


def test_rewrites_absolute_upstream_location_header(monkeypatch):
    """The upstream can emit an absolute Location pointing at its own host and
    /api/v1 prefix (e.g. FastAPI's trailing-slash 307, or any in-API redirect).
    Left as-is, the browser would send the follow-up request straight to the
    upstream, cross-origin (CORS). The proxy must map it back into the public
    /api/autotunex/* space so the browser keeps talking to gbserver."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            307,
            headers={"location": "http://autotunex.test/api/v1/datasets/ABC123"},
        )

    monkeypatch.setattr(proxy_mod, "AUTOTUNEX_URL", "http://autotunex.test")
    monkeypatch.setattr(proxy_mod, "_client", _client_with_handler(handler))

    client = TestClient(_make_app())
    resp = client.post("/api/autotunex/datasets", follow_redirects=False)

    assert resp.status_code == 307
    assert resp.headers["location"] == "/api/autotunex/datasets/ABC123"


def test_rewrites_location_when_upstream_url_has_trailing_slash(monkeypatch):
    """A trailing slash in AUTOTUNEX_API_URL must not defeat the rewrite. The
    match is on the /api/v1 path prefix, not an exact origin-string compare,
    so the follow-up requests still come back through gbserver."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            307,
            headers={"location": "http://autotunex.test/api/v1/datasets/XYZ"},
        )

    monkeypatch.setattr(proxy_mod, "AUTOTUNEX_URL", "http://autotunex.test/")
    monkeypatch.setattr(proxy_mod, "_client", _client_with_handler(handler))

    client = TestClient(_make_app())
    resp = client.post("/api/autotunex/datasets", follow_redirects=False)

    assert resp.headers["location"] == "/api/autotunex/datasets/XYZ"


def test_leaves_non_upstream_location_untouched(monkeypatch):
    """A redirect that does not point into the upstream /api/v1 space (e.g.
    an external auth provider) must be passed through verbatim, not mangled."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302, headers={"location": "https://idp.example.com/oauth/authorize?x=1"}
        )

    monkeypatch.setattr(proxy_mod, "AUTOTUNEX_URL", "http://autotunex.test")
    monkeypatch.setattr(proxy_mod, "_client", _client_with_handler(handler))

    client = TestClient(_make_app())
    resp = client.get("/api/autotunex/auth/login", follow_redirects=False)

    assert resp.headers["location"] == "https://idp.example.com/oauth/authorize?x=1"


def test_does_not_leak_cookies_between_requests(monkeypatch):
    """A shared AsyncClient must not replay one request's Set-Cookie onto a
    later cookie-less request (cross-user session bleed)."""
    seen_cookies = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_cookies.append(request.headers.get("cookie"))
        # Upstream sets a session cookie on the first (authenticated) request.
        # Real auth cookies are scoped Path=/ (as AutoTuneX's are), so httpx's
        # jar considers them valid for every later request path, not just
        # ones under the same directory as the request that set them.
        return httpx.Response(
            200,
            headers=[("set-cookie", "session=USER_A; Path=/")],
            json={"ok": True},
        )

    monkeypatch.setattr(proxy_mod, "AUTOTUNEX_URL", "http://autotunex.test")
    monkeypatch.setattr(proxy_mod, "_client", _client_with_handler(handler))

    # Two separate TestClients simulate two different browsers/users. Each
    # TestClient is itself an httpx client with its OWN cookie jar, so if we
    # reused a single TestClient for both requests, its jar would legitimately
    # resend USER_A's cookie on request 2 and mask the actual bug: the
    # gbserver-side *shared upstream* `_client`'s jar bleeding a cookie across
    # unrelated requests/users. Using two clients isolates that.
    app = _make_app()
    # Request 1: user A carries their cookie; upstream returns Set-Cookie.
    TestClient(app).get("/api/autotunex/auth/me", headers={"cookie": "session=USER_A"})
    # Request 2: a different, cookie-less request must NOT carry USER_A upstream.
    TestClient(app).get("/api/autotunex/job/by_build_id/x")

    assert seen_cookies[0] == "session=USER_A"
    assert seen_cookies[1] in (
        None,
        "",
    ), f"cookie leaked to a cookie-less request: {seen_cookies[1]!r}"
