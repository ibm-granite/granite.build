# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""AuthProvider seam contract: abstracts identity resolution only.

The OIDC flow, JWT session create/decode, cookies, and admin impersonation stay
in auth.py as shared infrastructure. A provider only answers "who is this
request?" by returning a models.AuthUser.
"""

from abc import ABC, abstractmethod


class AuthProvider(ABC):
    @abstractmethod
    async def get_current_user(self, request):  # -> models.AuthUser
        ...

    def login_routes(self):
        """Optional provider-specific FastAPI routes. Default: None — the shared
        auth_router owns login/callback for the built-in providers."""
        return None
