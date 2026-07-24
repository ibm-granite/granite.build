# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import dependencies
import models as api
from auth import get_current_user
from fastapi import APIRouter, Depends
from services import (  # for type annotations config_service.Config / user_service.User
    config_service,
    user_service,
)

router = APIRouter()


@router.get(
    "/api/config",
    tags=["Configurations"],
    summary="Get configuration template",
    response_description="Default configuration template for the UI",
)
def get_config_template(
    configuration: config_service.Config = Depends(dependencies.get_config_service),
):
    """
    Retrieve the default AutoTune configuration template.

    Returns a template configuration that can be used as a starting point
    for creating new tuning configurations. Includes:
    - Available hyperparameters and their ranges
    - Tuner settings (optimizer, search space, etc.)
    - Training parameters
    - Default values and constraints

    **Returns:**
    - Configuration template object

    **Use Case:**
    Use this to understand the expected configuration structure before
    creating a new configuration via POST /api/config
    """
    return configuration.get_config_for_ui()


@router.post(
    "/api/config",
    tags=["Configurations"],
    summary="Create or update a configuration",
    response_description="Response with configuration ID and status",
)
async def create_config(
    config: api.Configuration,
    auth_user: api.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
    configuration: config_service.Config = Depends(dependencies.get_config_service),
) -> api.Response:
    """
    Create a new configuration or update an existing one.

    Configurations define the hyperparameter search space and tuning strategy
    for fine-tuning jobs. They include:
    - **name**: Unique configuration name
    - **tuner_type**: Type of HPO algorithm (Bayesian, Grid Search, etc.)
    - **config_data**: Hyperparameter definitions and ranges

    **Request Body Example:**
    ```json
    {
      "name": "my-lora-config",
      "tuner_type": "bayesian",
      "config_data": {
        "learning_rate": {"min": 1e-5, "max": 1e-3},
        "lora_rank": {"values": [8, 16, 32]},
        "batch_size": {"values": [4, 8, 16]}
      }
    }
    ```

    **Returns:**
    - Configuration ID and creation status

    **Note:** If config with same name exists, it will be updated
    """
    user_id = (await user.get_user(auth_user.email))["id"]
    config.user_id = user_id
    return await configuration.push_config(config)


@router.get(
    "/api/configs",
    tags=["Configurations"],
    summary="List all configurations",
    response_description="Array of configuration objects for the current user",
)
async def get_configs(
    auth_user: api.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
    configuration: config_service.Config = Depends(dependencies.get_config_service),
) -> list[api.SimpleConfiguration]:
    """
    Retrieve all configurations created by the authenticated user.

    Returns a list of saved configurations that can be used to start
    new fine-tuning jobs. Each configuration includes:
    - Configuration ID and name
    - Tuner type and settings
    - Hyperparameter search space
    - Creation timestamp

    **Returns:**
    - Array of configuration objects

    **Use Case:**
    - View existing configurations before starting a job
    - Reuse proven configurations for new experiments
    """
    user_id = (await user.get_user(auth_user.email))["id"]
    return await configuration.get_configs(user_id=user_id)


@router.put(
    "/api/config/{config_id}",
    tags=["Configurations"],
    summary="Update an existing configuration",
    response_description="Response with configuration ID and status",
)
async def update_config(
    config_id: str,
    config: api.Configuration,
    auth_user: api.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
    configuration: config_service.Config = Depends(dependencies.get_config_service),
) -> api.Response:
    """
    Update an existing configuration.

    **Parameters:**
    - **config_id**: Unique configuration identifier (UUID)
    - **config**: Updated configuration object

    **Returns:**
    - Configuration ID and update status

    **Errors:**
    - 403: User doesn't have permission to update this configuration
    - 404: Configuration not found
    """
    user_id = (await user.get_user(auth_user.email))["id"]
    config.id = config_id
    config.user_id = user_id
    return await configuration.push_config(config)


@router.delete(
    "/api/config/{config_id}",
    tags=["Configurations"],
    summary="Delete a configuration",
    response_description="Boolean indicating deletion success",
)
async def delete_config(
    config_id: str,
    auth_user: api.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
    configuration: config_service.Config = Depends(dependencies.get_config_service),
) -> bool:
    """
    Permanently delete a configuration.

    **Warning:** This operation cannot be undone. Any jobs using this
    configuration will maintain their copy, but you cannot create new
    jobs with this configuration.

    **Parameters:**
    - **config_id**: Unique configuration identifier (UUID)

    **Returns:**
    - `true` if deletion was successful
    - `false` if configuration was not found

    **Errors:**
    - 403: User doesn't have permission to delete this configuration
    """
    user_id = (await user.get_user(auth_user.email))["id"]
    return await configuration.delete_config(config_id, user_id)


@router.get(
    "/api/config/{config_id}",
    tags=["Configurations"],
    summary="Get specific configuration",
    response_description="Complete configuration object",
)
async def get_config(
    config_id: str,
    auth_user: api.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
    configuration: config_service.Config = Depends(dependencies.get_config_service),
) -> api.Configuration:
    """
    Retrieve detailed information about a specific configuration.

    Returns the complete configuration including:
    - Hyperparameter search space definitions
    - Tuner type and settings
    - Constraints and validation rules
    - Metadata (name, creation date, etc.)

    **Parameters:**
    - **config_id**: Unique configuration identifier (UUID)

    **Returns:**
    - Configuration object with all details

    **Use Cases:**
    - Review configuration before starting a job
    - Clone/modify existing configurations
    - Audit hyperparameter choices

    **Errors:**
    - 404: Configuration not found or access denied
    """
    user_id = (await user.get_user(auth_user.email))["id"]
    return await configuration.get_config(config_id, user_id)
