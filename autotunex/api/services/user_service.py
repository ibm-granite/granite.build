# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import logging

import models as api
from services import db_service

logger = logging.getLogger(__name__)


class User:
    def __init__(self, db: db_service.Database):
        self.db = db

    async def push_user(self, email: str) -> api.Response:
        """Push an user."""
        if email is not None:
            user_id = await self.db.insert_user(email)
            return {"status": api.Status.CREATED, "id": user_id}

    async def touch_user_login(self, email: str) -> None:
        await self.db.touch_user_login(email)

    async def update_user(self, user: api.User) -> api.Response:
        """Push an user."""
        if user is not None:
            user_id = await self.db.update_user(user=user)
            return {"status": api.Status.UPDATED, "id": user_id}

    async def get_user(self, email: str) -> api.User:
        """Get User data."""
        return await self.db.get_user(email)

    async def get_user_by_id(self, id: str) -> api.User:
        """Get User data by id"""
        return await self.db.get_user_by_id(id)

    async def get_users(self) -> list[api.User]:
        """Get User data."""
        return await self.db.get_users()

    async def get_user_detail(self, id: str):
        """Get User data."""
        return await self.db.get_user_detail(id)

    async def get_user_metadata(self, id: str):
        """Get User data."""
        return await self.db.get_user_metadata(id)
