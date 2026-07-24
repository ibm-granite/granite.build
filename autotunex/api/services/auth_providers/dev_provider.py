# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Dev auth provider — the no-OIDC bypass (verbatim from auth.get_current_user)."""

import logging
import os

import models as api
from services.auth_providers.base import AuthProvider

logger = logging.getLogger(__name__)


class DevAuthProvider(AuthProvider):
    async def get_current_user(self, request):
        dev_email = os.getenv("DEV_USER_EMAIL", "dev@example.com")
        dev_role = os.getenv("DEV_USER_ROLE", "admin")

        # Unlike the OIDC path, dev mode has no /callback to provision the user,
        # so the row never exists on a fresh DB and downstream get_user(email)["id"]
        # lookups 500. Ensure it exists here, mirroring auth.callback's push_user.
        import dependencies
        from services import user_service

        user = user_service.User(dependencies.get_database())
        if await user.get_user(dev_email) is None:
            await user.push_user(dev_email)

        return api.AuthUser(email=dev_email, role=api.Roles(dev_role))
