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

"""Regression tests for the Swagger Bearer-auth advertisement.

Guards against a FastAPI upgrade or refactor silently dropping the security
scheme (which would make the /docs Authorize button disappear) and confirms the
helpers only touch the generated schema, never the runtime request path.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gbserver.api.openapi_security import add_bearer_auth, enable_api

_SCHEME = "BearerAuth"


def _app_with_route() -> FastAPI:
    app = FastAPI()

    @app.get("/api/thing")
    async def thing():
        return {"ok": True}

    return app


class TestAddBearerAuth:
    def test_scheme_declared_in_components(self):
        app = _app_with_route()
        add_bearer_auth(app)
        schema = app.openapi()
        scheme = schema["components"]["securitySchemes"][_SCHEME]
        assert scheme["type"] == "http"
        assert scheme["scheme"] == "bearer"

    def test_requirement_declared_at_document_level(self):
        app = _app_with_route()
        add_bearer_auth(app)
        schema = app.openapi()
        # Top-level security is inherited by every operation → Swagger shows the
        # Authorize button and a lock on each endpoint.
        assert schema["security"] == [{_SCHEME: []}]

    def test_openapi_json_endpoint_serves_scheme(self):
        app = _app_with_route()
        add_bearer_auth(app)
        resp = TestClient(app).get("/openapi.json")
        assert resp.status_code == 200
        body = resp.json()
        assert _SCHEME in body["components"]["securitySchemes"]

    def test_does_not_add_route_or_change_responses(self):
        """The helper must be docs-only: no new routes, endpoints behave the same."""
        app = _app_with_route()
        before = {r.path for r in app.routes}
        add_bearer_auth(app)
        after = {r.path for r in app.routes}
        assert before == after
        assert TestClient(app).get("/api/thing").json() == {"ok": True}

    def test_schema_is_cached_and_stable(self):
        app = _app_with_route()
        add_bearer_auth(app)
        first = app.openapi()
        second = app.openapi()
        assert first is second  # cached, not rebuilt
        # No duplicate security entries accumulate on repeated access.
        assert second["security"] == [{_SCHEME: []}]


class TestEnableApi:
    def test_mounts_and_advertises_by_default(self):
        parent = FastAPI()
        sub = _app_with_route()
        enable_api(parent, "/api/v1/sub", sub)
        # Mounted under the parent...
        assert any(getattr(r, "path", None) == "/api/v1/sub" for r in parent.routes)
        # ...and the scheme is advertised on the sub-app's own schema.
        assert _SCHEME in sub.openapi()["components"]["securitySchemes"]

    def test_advertise_auth_false_mounts_without_scheme(self):
        """Exempt sub-apps (e.g. the login flow) mount but must not claim auth."""
        parent = FastAPI()
        sub = _app_with_route()
        enable_api(parent, "/api/v1/auth", sub, advertise_auth=False)
        assert any(getattr(r, "path", None) == "/api/v1/auth" for r in parent.routes)
        assert "security" not in sub.openapi()

    def test_parent_only_advertises_without_mounting(self):
        """enable_api(parent) advertises on the top-level app and mounts nothing."""
        parent = _app_with_route()
        before = {getattr(r, "path", None) for r in parent.routes}
        enable_api(parent)
        after = {getattr(r, "path", None) for r in parent.routes}
        assert before == after  # no new mount
        assert _SCHEME in parent.openapi()["components"]["securitySchemes"]
