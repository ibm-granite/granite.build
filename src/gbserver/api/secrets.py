#!/usr/bin/env python3

# Copyright Granite.secret Authors
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


import base64
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel

from gbserver.api.utils import get_lh_token_if_needed
from gbserver.spaces.user_spaces_list import user_spaces_list
from gbserver.types.constants import (
    ENV_VAR_IBM_SEC_MAN_API_KEY,
    ENV_VAR_IBM_SEC_MAN_ENDPOINT,
    PUBLIC_SPACE_NAME,
)
from gbserver.types.secret import MySecret
from gbserver.utils.get_header_auth_token import get_header_auth_token
from gbserver.utils.logger import get_logger
from gbserver.utils.secretmanager import MySecretsManagerAPI

logger = get_logger(__name__)


def _get_ibm_secret_manager_admin():
    from gbserver.spacesecretmanager.ibmcloudspacesecretmanager import (
        IbmcloudSpaceSecretManagerAdmin,
    )

    return IbmcloudSpaceSecretManagerAdmin()


secret_manager: Optional[MySecretsManagerAPI] = None

sec_man_api_key = os.getenv(ENV_VAR_IBM_SEC_MAN_API_KEY, "")
sec_man_endpoint = os.getenv(ENV_VAR_IBM_SEC_MAN_ENDPOINT, "")

if sec_man_api_key != "":
    secret_manager = MySecretsManagerAPI(
        api_endpoint=sec_man_endpoint,
        api_key=sec_man_api_key,
    )

secrets_api = FastAPI()


class SecretCreateRequest(BaseModel):
    secret_name: str
    secret_value: str
    encoding: str


class SecretUpdateRequest(BaseModel):
    secret_value: str
    encoding: str


def _get_space_for_admin(username: str, space_name: str, lh_token: Optional[str] = None):
    spaces = user_spaces_list(username, lh_token=lh_token)
    # logger.info(spaces)
    space = list(filter(lambda x: x["name"] == space_name, spaces))
    if space is None or len(space) != 1:
        raise Exception("Space lookup failed")
    if not space[0]["is_admin"]:
        raise Exception("Only space admin can perform this operation")
    return space[0]


@secrets_api.get("/space_secrets/{space_name}")
def list_space_secrets(request: Request, space_name: str):
    """Get the list of secrets for a space."""
    try:
        username = request.state.data["user"].email
        lh_token = get_lh_token_if_needed(request)

        space = _get_space_for_admin(username, space_name, lh_token=lh_token)
        manager = _get_ibm_secret_manager_admin()
        logger.info("Fetching secrets for", space)
        secret_group_name = manager.get_secret_group_for_space(space)
        if secret_group_name is None:
            raise Exception(f"Secret group is unavailable")
        return {
            "space_name": space_name,
            "secrets": manager.list_secret_names(secret_group_name),
        }
    except Exception as e:
        logger.error("Failed to get the list of space secrets: %s", e)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=repr(e))


@secrets_api.get("/space_secrets/{space_name}/{secret_name}")
def get_space_secret(request: Request, space_name: str, secret_name: str):
    """Get a secret value."""
    try:
        username = request.state.data["user"].email
        lh_token = get_lh_token_if_needed(request)

        space = _get_space_for_admin(username, space_name, lh_token=lh_token)
        manager = _get_ibm_secret_manager_admin()
        logger.info("Fetching a secret for", space)
        secret_group_name = manager.get_secret_group_for_space(space)
        if secret_group_name is None:
            raise Exception(f"Secret group is unavailable")
        secret_value = manager.get_secret_value(secret_group_name, secret_name, True)
        if secret_value is None:
            raise Exception("secret not found")
        return {
            "space_name": space_name,
            "secret_name": secret_name,
            "secret_value": secret_value,
            "encoding": "base64",
        }
    except Exception as e:
        logger.error("Failed to get a space secret: %s", e)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=repr(e))


@secrets_api.post("/space_secrets/{space_name}")
def create_space_secret(request: Request, space_name: str, secret_request: SecretCreateRequest):
    """Create a new secret."""
    try:
        username = request.state.data["user"].email
        lh_token = get_lh_token_if_needed(request)

        if secret_request.secret_name is None:
            raise Exception("Invalid secret name")
        if secret_request.secret_value is None:
            raise Exception("Invalid secret value")
        if (
            secret_request.encoding is not None
            and secret_request.encoding != "base64"
            and secret_request.encoding != "plain"
        ):
            raise Exception("Unsupported encoding")

        space = _get_space_for_admin(username, space_name, lh_token=lh_token)
        manager = _get_ibm_secret_manager_admin()
        logger.info("Creating a secret for", space)
        secret_group_name = manager.get_secret_group_for_space(space)
        if secret_group_name is None:
            raise Exception(f"Secret group is unavailable")
        secret_value = (
            base64.b64decode(secret_request.secret_value.encode("ascii")).decode("utf-8")
            if secret_request.encoding == "base64"
            else secret_request.secret_value
        )
        manager.create_secret(
            secret_group_name=secret_group_name,
            secret_name=secret_request.secret_name,
            secret_value=secret_value,
        )
        return {"result": "success"}
    except Exception as e:
        logger.error("Failed to create a space secret: %s", e)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=repr(e))


@secrets_api.put("/space_secrets/{space_name}/{secret_name}")
def update_space_secret(
    request: Request,
    space_name: str,
    secret_name: str,
    secret_request: SecretUpdateRequest,
):
    """Update an existing secret."""
    try:
        username = request.state.data["user"].email
        lh_token = get_lh_token_if_needed(request)

        if secret_name is None:
            raise Exception("Invalid secret name")
        if secret_request.secret_value is None:
            raise Exception("Invalid secret value")
        if (
            secret_request.encoding is not None
            and secret_request.encoding != "base64"
            and secret_request.encoding != "plain"
        ):
            raise Exception("Unsupported encoding")

        space = _get_space_for_admin(username, space_name, lh_token=lh_token)
        manager = _get_ibm_secret_manager_admin()
        logger.info("Updating a secret for", space)
        secret_group_name = manager.get_secret_group_for_space(space)
        if secret_group_name is None:
            raise Exception(f"Secret group is unavailable")
        secret_value = (
            base64.b64decode(secret_request.secret_value.encode("ascii")).decode("utf-8")
            if secret_request.encoding == "base64"
            else secret_request.secret_value
        )
        manager.update_secret_value(secret_group_name, secret_name, secret_value)
        return {"result": "success"}
    except Exception as e:
        logger.error("Failed to update a space secret: %s", e)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=repr(e))


@secrets_api.delete("/space_secrets/{space_name}/{secret_name}")
def delete_space_secret(request: Request, space_name: str, secret_name: str):
    """Delete a secret."""
    try:
        username = request.state.data["user"].email
        lh_token = get_lh_token_if_needed(request)

        if secret_name is None:
            raise Exception("Invalid secret name")

        space = _get_space_for_admin(username, space_name, lh_token=lh_token)
        manager = _get_ibm_secret_manager_admin()
        logger.info("Deleting a secret for", space)
        secret_group_name = manager.get_secret_group_for_space(space)
        if secret_group_name is None:
            raise Exception(f"Secret group is unavailable")
        manager.delete_secret(secret_group_name, secret_name)
        return {"result": "success"}
    except Exception as e:
        logger.error("Failed to delete a space secret: %s", e)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=repr(e))


@secrets_api.get("/user_secrets")
def list_user_secrets(request: Request):
    """Get the list of secrets for a user."""
    try:
        user_id = request.state.data["user"].login
        if user_id is None:
            # if the above dereference fails somewhere it will trigger an exception anyway
            raise Exception("Failed to obtain username")
        manager = _get_ibm_secret_manager_admin()
        logger.info("Fetching user secrets")
        secret_group_name = manager.get_secret_group_for_users()
        if secret_group_name is None:
            raise Exception(f"Secret group is unavailable")
        secrets_all_users = manager.list_secret_names(secret_group_name)
        return {
            "user": user_id,
            "secrets": manager.filter_user_secrets(user_id, secrets_all_users),
        }
    except Exception as e:
        logger.error("Failed to get the list of user secrets: %s", e)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=repr(e))


@secrets_api.get("/user_secrets/{secret_name}")
def get_user_secret(request: Request, secret_name: str):
    """Get a secret value."""
    try:
        user_id = request.state.data["user"].login
        if user_id is None:
            # if the above dereference fails somewhere it will trigger an exception anyway
            raise Exception("Failed to obtain username")
        manager = _get_ibm_secret_manager_admin()
        logger.info("Fetching a user secret")
        secret_group_name = manager.get_secret_group_for_users()
        if secret_group_name is None:
            raise Exception(f"Secret group is unavailable")
        secret_name_for_user = manager.get_secret_name_for_user(user_id, secret_name)
        secret_value = manager.get_secret_value(secret_group_name, secret_name_for_user, True)
        if secret_value is None:
            raise Exception("secret not found")
        return {
            "user_id": user_id,
            "secret_name": secret_name,
            "secret_value": secret_value,
            "encoding": "base64",
        }
    except Exception as e:
        logger.error("Failed to get a user secret: %s", e)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=repr(e))


@secrets_api.post("/user_secrets")
def create_user_secret(request: Request, secret_request: SecretCreateRequest):
    """Create a new secret."""
    try:
        user_id = request.state.data["user"].login
        if user_id is None:
            # if the above dereference fails somewhere it will trigger an exception anyway
            raise Exception("Failed to obtain username")

        if secret_request.secret_name is None:
            raise Exception("Invalid secret name")
        if secret_request.secret_value is None:
            raise Exception("Invalid secret value")
        if (
            secret_request.encoding is not None
            and secret_request.encoding != "base64"
            and secret_request.encoding != "plain"
        ):
            raise Exception("Unsupported encoding")

        manager = _get_ibm_secret_manager_admin()
        logger.info("Creating a user secret")
        secret_group_name = manager.get_secret_group_for_users()
        if secret_group_name is None:
            raise Exception(f"Secret group is unavailable")
        secret_name_for_user = manager.get_secret_name_for_user(user_id, secret_request.secret_name)
        secret_value = (
            base64.b64decode(secret_request.secret_value.encode("ascii")).decode("utf-8")
            if secret_request.encoding == "base64"
            else secret_request.secret_value
        )
        manager.create_secret(
            secret_group_name=secret_group_name,
            secret_name=secret_name_for_user,
            secret_value=secret_value,
        )
        return {"result": "success"}
    except Exception as e:
        logger.error("Failed to create a user secret: %s", e)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=repr(e))


@secrets_api.put("/user_secrets/{secret_name}")
def update_user_secret(
    request: Request,
    secret_name: str,
    secret_request: SecretUpdateRequest,
):
    """Update an existing secret."""
    try:
        user_id = request.state.data["user"].login
        if user_id is None:
            # if the above dereference fails somewhere it will trigger an exception anyway
            raise Exception("Failed to obtain username")

        if secret_name is None:
            raise Exception("Invalid secret name")
        if secret_request.secret_value is None:
            raise Exception("Invalid secret value")
        if (
            secret_request.encoding is not None
            and secret_request.encoding != "base64"
            and secret_request.encoding != "plain"
        ):
            raise Exception("Unsupported encoding")

        manager = _get_ibm_secret_manager_admin()
        logger.info("Updating a user secret")
        secret_group_name = manager.get_secret_group_for_users()
        if secret_group_name is None:
            raise Exception(f"Secret group is unavailable")
        secret_name_for_user = manager.get_secret_name_for_user(user_id, secret_name)
        secret_value = (
            base64.b64decode(secret_request.secret_value.encode("ascii")).decode("utf-8")
            if secret_request.encoding == "base64"
            else secret_request.secret_value
        )
        manager.update_secret_value(secret_group_name, secret_name_for_user, secret_value)
        return {"result": "success"}
    except Exception as e:
        logger.error("Failed to update a user secret: %s", e)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=repr(e))


@secrets_api.delete("/user_secrets/{secret_name}")
def delete_user_secret(request: Request, secret_name: str):
    """Delete a secret."""
    try:
        user_id = request.state.data["user"].login
        if user_id is None:
            # or if the above dereference fails somewhere it will trigger an exception anyway
            raise Exception("Failed to obtain username")

        if secret_name is None:
            raise Exception("Invalid secret name")

        manager = _get_ibm_secret_manager_admin()
        logger.info("Deleting a user secret")
        secret_group_name = manager.get_secret_group_for_users()
        if secret_group_name is None:
            raise Exception(f"Secret group is unavailable")
        secret_name_for_user = manager.get_secret_name_for_user(user_id, secret_name)
        manager.delete_secret(secret_group_name, secret_name_for_user)
        return {"result": "success"}
    except Exception as e:
        logger.error("Failed to delete a user secret: %s", e)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=repr(e))


@secrets_api.get("/lakehouse/artifact_token")
def get_artifact_key(space: str = PUBLIC_SPACE_NAME):
    """Get a Lakehouse token for artifact upload/download"""
    if space != PUBLIC_SPACE_NAME:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Artifact token is only available for the '{PUBLIC_SPACE_NAME}' space",
        )
    try:
        from gbserver.utils.lakehouse_token_generator import (
            generate_lakehouse_key_for_artifact,
        )

        lakehouse_token = generate_lakehouse_key_for_artifact(space)
        return {"lakehouse_token": lakehouse_token}
    except Exception as e:
        logger.error("failed to get a Lakehouse token error: %s", e)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=repr(e))


@secrets_api.get("/lakehouse/user_token")
def get_user_key(request: Request):
    """Get a Lakehouse token using user token"""
    try:
        user_token = get_header_auth_token(request.headers.get("authorization", ""))

        user = request.state.data.get("user")
        auth_provider = getattr(user, "auth_provider", "github") if user else "github"

        if auth_provider == "ibmid":
            from gbserver.utils.lakehouse_token_generator import (
                generate_lakehouse_key_from_ibmid_token,
            )

            lakehouse_token = generate_lakehouse_key_from_ibmid_token(user_token)
        else:
            from gbserver.utils.lakehouse_token_generator import (
                generate_lakehouse_key_from_user_token,
            )

            lakehouse_token = generate_lakehouse_key_from_user_token(user_token)

        return {"lakehouse_token": lakehouse_token}
    except Exception as e:
        logger.error("failed to get a Lakehouse token error: %s", e)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=repr(e))
