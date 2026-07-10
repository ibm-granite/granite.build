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

"""Advertise the Bearer-token auth scheme to Swagger UI.

Authentication is enforced entirely by :class:`~gbserver.api.auth.AuthMiddleware`,
which reads the ``Authorization: Bearer <token>`` header in pure Starlette
middleware. Because that enforcement lives outside FastAPI's dependency system,
nothing in the generated OpenAPI schema declares that a bearer token exists — so
Swagger UI never renders its "Authorize" button and "Try it out" requests go out
without the header.

:func:`add_bearer_auth` patches an app's ``openapi()`` to inject a
``BearerAuth`` security scheme and mark every operation as requiring it. This is
purely a documentation/UI convenience: it only affects the generated
``/openapi.json`` (consumed by the docs page) and has no effect on request
routing, validation, or the actual auth decision, which remains the middleware's
sole responsibility.
"""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

_SCHEME_NAME = "BearerAuth"


def add_bearer_auth(app: FastAPI, *, title: str = "granite.build API") -> None:
    """Make Swagger UI show an "Authorize" button for *app* that sends the
    ``Authorization: Bearer <token>`` header on "Try it out" requests.

    Safe to call on every mounted sub-app; each app owns its own ``/docs`` and
    ``/openapi.json``, so the scheme must be declared per app to appear on that
    app's docs page. Does not change runtime behavior — see the module docstring.
    """

    def custom_openapi() -> dict:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=title,
            version="1.0.0",
            routes=app.routes,
        )
        schema.setdefault("components", {})["securitySchemes"] = {
            _SCHEME_NAME: {
                "type": "http",
                "scheme": "bearer",
                "description": (
                    "Paste your API token (without the 'Bearer ' prefix). "
                    "Swagger will send it as 'Authorization: Bearer <token>'."
                ),
            }
        }
        # Apply the scheme to every operation so the single "Authorize" action
        # sends the header for all endpoints on the page.
        for path_item in schema.get("paths", {}).values():
            for operation in path_item.values():
                if isinstance(operation, dict):
                    operation.setdefault("security", []).append({_SCHEME_NAME: []})
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi  # type: ignore[method-assign]
