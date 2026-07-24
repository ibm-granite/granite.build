# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

# import os
import models as api
import constants
from services import db_service, gb_service
import logging
from fastapi import HTTPException
# from utils import is_gb_enabled, extract_uuid_uri
# from .yaml_service import YAMLManager

logger = logging.getLogger(__name__)
gb: gb_service.GBService = gb_service.GBService()


class Config:
    def __init__(self, db: db_service.Database):
        self.db = db

    async def push_config(self, config: api.Configuration) -> api.Response:
        """Push an agent."""
        try:
            if config.id is not None:
                # On update, guard against renaming onto another config this user already owns.
                existing = await self.db.get_config_by_name_and_user(
                    config_name=config.name, user_id=config.user_id
                )
                if existing is not None and existing.get("id") != config.id:
                    raise HTTPException(
                        status_code=409,
                        detail=f"A configuration named '{config.name}' already exists for this user.",
                    )
                await self.db.update_configuration(config)
                return {"status": api.Status.UPDATED, "id": config.id}
            else:
                existing_config = await self.db.get_config_by_name_and_user(
                    config_name=config.name, user_id=config.user_id
                )
                if existing_config is not None:
                    raise HTTPException(
                        status_code=409,
                        detail=f"A configuration named '{config.name}' already exists for this user.",
                    )
                # Prevent shadowing a SYSTEM_USER configuration: get_configs includes
                # system-owned configs in every user's view, so the name would clash
                # in the UI even though the DB constraint is only per-user.
                system_existing = await self.db.get_config_by_name_and_user(
                    config_name=config.name, user_id=constants.SYSTEM_USER
                )
                if system_existing is not None:
                    raise HTTPException(
                        status_code=409,
                        detail=f"'{config.name}' is a reserved system configuration name.",
                    )
                # if is_gb_enabled():
                #     sanitized_config_name = config.name.replace(" ", "_")
                #     logger.debug("Push Config: GB ENABLE")
                #     config_path = (
                #         f"{os.getcwd()}/temp_yaml/configs/{sanitized_config_name}"
                #     )
                #     yaml_file = YAMLManager(
                #         f"{config_path}/{sanitized_config_name}.yaml"
                #     )
                #     yaml_file.write_yaml(config.config_data)
                #     command = [
                #         "artifact",
                #         "push",
                #         "--from-local",
                #         config_path,
                #         "--artifact-name",
                #         config.name.replace(" ", "_"),
                #         "--type",
                #         "fileset",
                #         "--certify-no-restrictions",
                #     ]
                #     logger.debug(f"command for config creation: {command}")
                #     result = gb.command_executor(command)
                #     logger.info(result)

                #     uuid, uri = extract_uuid_uri(result)
                #     if uuid is None and uri is None:
                #         error = {
                #             "status": "GB_ARTIFACT_ERROR",
                #             "message": "Error occured while creating artifact",
                #         }
                #         logger.error(error)
                #         raise Exception("Error occured while creating artifact")
                #     config.artifact_id = uuid.strip()
                #     config.artifact_url = uri.strip()
                #     logger.debug(f"config: {config}")

                config_id = await self.db.insert_configuration(config)
                return {"status": api.Status.CREATED, "id": config_id}
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("push_config failed")
            raise HTTPException(status_code=400, detail=str(e))

    async def get_configs(
        self, user_id: str, ids: list[str] = None
    ) -> list[api.Config]:
        """Get all configs."""
        return await self.db.get_configs(user_id=user_id, ids=ids)

    async def delete_config(self, id: str, user_id: str) -> bool:
        """Delete config."""
        return await self.db.delete_config(id, user_id)

    async def get_config(self, id: str, user_id: str) -> api.Configuration:
        """Get config data."""
        return await self.db.get_config(id, user_id)

    def get_config_for_ui(self):
        """Get supported autotune configuration data."""
        try:
            # Lazy import: the autotune training core is an optional IBM
            # dependency, absent in a credential-free install. Import it only
            # when this UI-config path is actually exercised.
            from autotune.utils import get_autotune_config

            config = get_autotune_config()
            return config
            # with open("./config/autotune_debug.yaml") as f:
            #     config = yaml.safe_load(f)
            #     return config
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=404, detail="Config file not found")
