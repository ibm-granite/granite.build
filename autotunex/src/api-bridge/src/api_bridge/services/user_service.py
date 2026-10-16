# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""User service for api-bridge (local copy from api/services/user_service.py)."""

import logging

from api_bridge import database, models

logger = logging.getLogger(__name__)


class User:
    def __init__(self, db: database.Database):
        self.db = db

    def push_user(self, email: str):
        """Create a new user."""
        if email is not None:
            user_id = self.db.insert_user(email)
            return {"status": models.Status.CREATED, "id": user_id}

    def update_user(self, user: models.User):
        """Update an existing user."""
        if user is not None:
            user_id = self.db.update_user(user=user)
            return {"status": models.Status.UPDATED, "id": user_id}

    def get_user(self, email: str):
        """Get user by email."""
        return self.db.get_user(email)
