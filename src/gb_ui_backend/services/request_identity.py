# Copyright Granite.Build Authors
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

"""Shared caller-identity resolution for gb_ui_backend routes.

gbserver's AuthMiddleware attaches the trusted user to
`request.state.data["user"]` before routes run (these routers are mounted
directly on gbserver's own app); the X-User-Email header is a fallback for
running this app standalone, outside gbserver, where there's no
AuthMiddleware at all. Every route that needs "who is calling this" resolves
it via resolve_identity() below rather than reimplementing the precedence,
so a future change (a new auth mode, a different final fallback) only has
to be made once.
"""

from __future__ import annotations

from fastapi import Request


def resolve_identity(request: Request, fallback: str = "standalone") -> str:
    """AuthMiddleware-trusted user -> X-User-Email header -> `fallback`.

    `fallback` differs by caller: most routes have no per-user identity at
    all in apikey/localhost mode and use the "standalone" sentinel; a
    rate-limiting caller can instead pass the client's IP, since
    limiting-by-IP is still meaningful even without a per-user identity.

    Must never be trusted from a client-supplied field — the trusted-user
    branch is what makes this safe to use for authorization/scoping, not
    just display.
    """
    user = getattr(request.state, "data", {}).get("user")
    if user is not None:
        return user.email
    return request.headers.get("x-user-email") or fallback
