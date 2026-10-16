# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Config service for api-bridge (local copy from api/services/config_service.py)."""

import logging

from fastapi import HTTPException

from api_bridge import database, models
from api_bridge.normalization import normalized

logger = logging.getLogger(__name__)
MAX_CONFIG_SUFFIX = 100


class Config:
    def __init__(self, db: database.Database):
        self.db = db

    async def push_config(self, config: models.Configuration) -> models.Response:
        """Create or update a configuration."""
        try:
            if config.id is not None:
                self.db.update_configuration(config)
                return {"status": models.Status.UPDATED, "id": config.id}
            else:
                existing_config = self.db.get_config_by_name_and_user(
                    config_name=config.name, user_id=config.user_id
                )
                if existing_config is not None and existing_config.get("name", None) == config.name:
                    logger.error(f"Configuration '{config.name}' already exists for this user")
                    raise Exception(f"Configuration '{config.name}' already exists for this user")

                config_id = self.db.insert_configuration(config)
                return {"status": models.Status.CREATED, "id": config_id}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    def get_configs(self, user_id: str, ids: list = None) -> list:
        """Get all configs for a user."""
        return self.db.get_configs(user_id=user_id, ids=ids)

    def get_config_by_name(self, user_id: str, config_name: str) -> dict | None:
        """Get a config by name for a user."""
        return self.db.get_config_by_name_and_user(config_name=config_name, user_id=user_id)

    def find_or_create(
        self,
        name: str,
        tuner_type: str,
        config_data: dict,
        user_id: str,
        rl_tuner_type: str | None = None,
    ) -> str:
        """Return a config id, reusing an existing match or creating a new one.

        Looks up by (name, user_id). If found and config_data matches (after
        normalization), reuses it. If it differs, searches name-1, name-2, ...
        for a normalized match or a free slot.
        """

        def _insert(cfg_name: str) -> str:
            cfg = models.Configuration(
                user_id=user_id,
                name=cfg_name,
                tuner_type=tuner_type,
                config_data=config_data,
            )
            new_id = self.db.insert_configuration(cfg)
            logger.info("Config created: %s (id=%s)", cfg_name, new_id)
            return str(new_id)

        existing = self.db.get_config_by_name_and_user(config_name=name, user_id=user_id)
        if existing is None:
            return _insert(name)
        if normalized(existing.get("config_data")) == normalized(config_data):
            logger.info("Config reused: %s (id=%s)", name, existing["id"])
            return str(existing["id"])

        for suffix in range(1, MAX_CONFIG_SUFFIX + 1):
            candidate = f"{name}-{suffix}"
            existing = self.db.get_config_by_name_and_user(config_name=candidate, user_id=user_id)
            if existing is None:
                return _insert(candidate)
            if normalized(existing.get("config_data")) == normalized(config_data):
                logger.info("Config reused: %s (id=%s)", candidate, existing["id"])
                return str(existing["id"])

        raise Exception(
            f"Too many config name collisions for '{name}' (checked up to {name}-{MAX_CONFIG_SUFFIX})"
        )
