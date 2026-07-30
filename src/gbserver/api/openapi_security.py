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

from typing import Optional

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from gbserver.types.constants import GBSERVER_GIT_COMMIT

_SCHEME_NAME = "BearerAuth"


def add_bearer_auth(app: FastAPI) -> None:
    """Make Swagger UI show an "Authorize" button for *app* that sends the
    ``Authorization: Bearer <token>`` header on "Try it out" requests.

    Safe to call on every mounted sub-app; each app owns its own ``/docs`` and
    ``/openapi.json``, so the scheme must be declared per app to appear on that
    app's docs page. Does not change runtime behavior — see the module docstring.

    The requirement is declared once at the document level (OpenAPI's top-level
    ``security`` field), which every operation inherits — Swagger renders the
    Authorize button and a per-operation lock exactly as it would for a
    per-operation declaration, without iterating the paths.
    """

    def custom_openapi() -> dict:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=GBSERVER_GIT_COMMIT or app.version,
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
        # Document-level requirement: inherited by every operation unless one
        # overrides it. Equivalent Swagger rendering to marking each operation,
        # without walking the paths.
        schema["security"] = [{_SCHEME_NAME: []}]
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi  # type: ignore[method-assign]


def enable_api(
    parent: FastAPI,
    path: str = "",
    sub_api: Optional[FastAPI] = None,
    *,
    advertise_auth: bool = True,
) -> None:
    """Wire up an API app's Swagger auth, and optionally mount it.

    Two forms:

    * ``enable_api(parent, path, sub_api)`` — mount *sub_api* under *parent* at
      *path*, then advertise the Bearer-token scheme on it.
    * ``enable_api(parent)`` — advertise the scheme on *parent* itself, for the
      top-level app that is never mounted. Nothing is mounted.

    Keeping both the mount and the Swagger-auth wiring behind this one function
    means an API app can't be added without its auth advertisement being
    considered, and ``root_api.py`` never has to reach for ``add_bearer_auth``
    directly. Pass ``advertise_auth=False`` for apps whose routes
    ``AuthMiddleware`` exempts from authentication (e.g. the login flow), so the
    docs don't claim a token is required where the server ignores it.
    """
    target = parent if sub_api is None else sub_api
    if sub_api is not None:
        parent.mount(path, sub_api)
    if advertise_auth:
        add_bearer_auth(target)
