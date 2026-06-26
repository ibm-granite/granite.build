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

from gbserver.spaces.user_spaces_list import user_spaces_list
from gbserver.types.constants import (
    ENV_VAR_IBM_SEC_MAN_API_KEY,
    ENV_VAR_IBM_SEC_MAN_ENDPOINT,
    PUBLIC_SPACE_NAME,
)
from gbserver.types.secret import MySecret
from gbserver.usersecretmanager.factory import get_user_secret_manager
from gbserver.utils.get_header_auth_token import get_header_auth_token
from gbserver.utils.logger import get_logger
from gbserver.utils.secretmanager import MySecretsManagerAPI

logger = get_logger(__name__)


def _get_ibm_secret_manager_admin():
    from gbserver.spacesecretmanager.ibmcloudspacesecretmanager import (
        IbmcloudSpaceSecretManagerAdmin,
    )

    return IbmcloudSpaceSecretManagerAdmin()


class _SpaceSecretAdmin:
    """Uniform CRUD facade over a space's secret backend for the admin REST API.

    Methods: list_names(), get_value(name) -> plain str | None,
    create(name, value), update(name, value), delete(name).
    Read-only backends raise NotImplementedError on writes (mapped to HTTP 405).
    """

    def __init__(self, manager, secret_group_name: str):
        self._m = manager
        self._group = secret_group_name

    def list_names(self):
        return self._m.list_secret_names(self._group)

    def get_value(self, secret_name: str):
        # SpaceSecretManager.get_secret returns {"value": ...} or {}.
        result = self._m.get_secret(secret_name, secret_group_name=self._group)
        if isinstance(result, dict):
            return result.get("value")
        return result

    def create(self, secret_name: str, secret_value: str):
        self._m.create_secret(
            secret_name=secret_name,
            secret_value=secret_value,
            secret_group_name=self._group,
        )

    def update(self, secret_name: str, secret_value: str):
        self._m.update_secret(
            secret_name=secret_name,
            secret_value=secret_value,
            secret_group_name=self._group,
        )

    def delete(self, secret_name: str):
        self._m.delete_secret(secret_name, secret_group_name=self._group)


class _IbmSpaceSecretAdmin:
    """Cloud space-secret admin facade preserving the prior IbmcloudSpaceSecretManagerAdmin behavior."""

    def __init__(self, admin, secret_group_name: str):
        self._a = admin
        self._group = secret_group_name

    def list_names(self):
        return self._a.list_secret_names(self._group)

    def get_value(self, secret_name: str):
        # Prior handler used encode=True and returned base64 directly; here we
        # return the plain value and let the handler base64-encode uniformly.
        return self._a.get_secret_value(self._group, secret_name, False)

    def create(self, secret_name: str, secret_value: str):
        self._a.create_secret(
            secret_group_name=self._group,
            secret_name=secret_name,
            secret_value=secret_value,
        )

    def update(self, secret_name: str, secret_value: str):
        self._a.update_secret_value(self._group, secret_name, secret_value)

    def delete(self, secret_name: str):
        self._a.delete_secret(self._group, secret_name)


# Cache of SpaceSecretManager instances keyed by (space name, space uri) so the
# standalone space-secret admin path does not re-pull the space repo (just to read
# space.yaml) on every /space_secrets request.
_space_secret_manager_cache: dict = {}


def _build_space_secret_manager(space_uri: str):
    """Build a SpaceSecretManager from a space's space.yaml.

    Pulls the space into a temporary directory (cleaned up on return) only to read
    space.yaml; the resulting manager reads from its own configured location
    (env vars / a configured secrets dir), not from the pulled copy.
    """
    import glob
    import tempfile
    from pathlib import Path

    from gbcommon.uri.uri import URI
    from gbserver.spacesecretmanager.spacesecretmanager import SpaceSecretManager
    from gbserver.types.spaceconfig import SpaceConfig

    space_yaml_name = "space.yaml"
    uriobj = URI.get_uri(uri=space_uri, default_scheme="file")
    with tempfile.TemporaryDirectory() as tmpdir:
        uriobj.pull(dest=Path(tmpdir))
        space_yamls = glob.glob(
            str(Path(tmpdir) / "**" / space_yaml_name), recursive=True
        )
        if not space_yamls:
            raise ValueError(f"No '{space_yaml_name}' found for space at {space_uri}")
        space_config: SpaceConfig = SpaceConfig.from_yaml(Path(space_yamls[0]))

    SpaceSecretManager.load_spacesecretmanagers()
    return SpaceSecretManager.get_spacesecretmanager(
        secret_manager_type=space_config.secret_manager.type,
        uri=space_uri,
        **space_config.secret_manager.config,
    )


def _get_space_secret_admin(space: dict):
    """Return a uniform space-secret admin facade for the given space.

    Outside standalone (cloud) the previous IBM admin behavior is preserved. In
    standalone mode the pluggable SpaceSecretManager configured in the space's
    space.yaml is used, so space-secret administration needs no IBM dependency.
    """
    from gbcommon.types.gbenvconfig import is_standalone

    if not is_standalone():
        admin = _get_ibm_secret_manager_admin()
        group = admin.get_secret_group_for_space(space)
        if group is None:
            raise Exception("Secret group is unavailable")
        return _IbmSpaceSecretAdmin(admin, group)

    space_uri = space["git_repo_uri"]
    cache_key = (space.get("name", ""), space_uri)
    manager = _space_secret_manager_cache.get(cache_key)
    if manager is None:
        manager = _build_space_secret_manager(space_uri)
        _space_secret_manager_cache[cache_key] = manager
    return _SpaceSecretAdmin(manager, space.get("name", ""))


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


def _get_space_for_admin(username: str, space_name: str):
    spaces = user_spaces_list(username)
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
        space = _get_space_for_admin(username, space_name)
        admin = _get_space_secret_admin(space)
        logger.info("Fetching secrets for space %s", space_name)
        return {
            "space_name": space_name,
            "secrets": admin.list_names(),
        }
    except Exception as e:
        logger.error("Failed to get the list of space secrets: %s", e)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=repr(e))


@secrets_api.get("/space_secrets/{space_name}/{secret_name}")
def get_space_secret(request: Request, space_name: str, secret_name: str):
    """Get a secret value."""
    try:
        username = request.state.data["user"].email
        space = _get_space_for_admin(username, space_name)
        admin = _get_space_secret_admin(space)
        logger.info("Fetching a secret for space %s", space_name)
        secret_value = admin.get_value(secret_name)
        if secret_value is None:
            raise Exception("secret not found")
        return {
            "space_name": space_name,
            "secret_name": secret_name,
            "secret_value": base64.b64encode(secret_value.encode("utf-8")).decode(
                "utf-8"
            ),
            "encoding": "base64",
        }
    except Exception as e:
        logger.error("Failed to get a space secret: %s", e)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=repr(e))


@secrets_api.post("/space_secrets/{space_name}")
def create_space_secret(
    request: Request, space_name: str, secret_request: SecretCreateRequest
):
    """Create a new secret."""
    try:
        username = request.state.data["user"].email

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

        space = _get_space_for_admin(username, space_name)
        admin = _get_space_secret_admin(space)
        logger.info("Creating a secret for space %s", space_name)
        secret_value = (
            base64.b64decode(secret_request.secret_value.encode("ascii")).decode(
                "utf-8"
            )
            if secret_request.encoding == "base64"
            else secret_request.secret_value
        )
        admin.create(secret_request.secret_name, secret_value)
        return {"result": "success"}
    except NotImplementedError as e:
        logger.error("Space secret backend is read-only: %s", e)
        raise HTTPException(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail=repr(e)
        )
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

        space = _get_space_for_admin(username, space_name)
        admin = _get_space_secret_admin(space)
        logger.info("Updating a secret for space %s", space_name)
        secret_value = (
            base64.b64decode(secret_request.secret_value.encode("ascii")).decode(
                "utf-8"
            )
            if secret_request.encoding == "base64"
            else secret_request.secret_value
        )
        admin.update(secret_name, secret_value)
        return {"result": "success"}
    except NotImplementedError as e:
        logger.error("Space secret backend is read-only: %s", e)
        raise HTTPException(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail=repr(e)
        )
    except Exception as e:
        logger.error("Failed to update a space secret: %s", e)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=repr(e))


@secrets_api.delete("/space_secrets/{space_name}/{secret_name}")
def delete_space_secret(request: Request, space_name: str, secret_name: str):
    """Delete a secret."""
    try:
        username = request.state.data["user"].email

        if secret_name is None:
            raise Exception("Invalid secret name")

        space = _get_space_for_admin(username, space_name)
        admin = _get_space_secret_admin(space)
        logger.info("Deleting a secret for space %s", space_name)
        admin.delete(secret_name)
        return {"result": "success"}
    except NotImplementedError as e:
        logger.error("Space secret backend is read-only: %s", e)
        raise HTTPException(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail=repr(e)
        )
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
        manager = get_user_secret_manager()
        logger.info("Fetching user secrets")
        return {
            "user": user_id,
            "secrets": manager.list_user_secrets(user_id),
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
        manager = get_user_secret_manager()
        logger.info("Fetching a user secret")
        secret_value = manager.get_user_secret(user_id, secret_name)
        if secret_value is None:
            raise Exception("secret not found")
        return {
            "user_id": user_id,
            "secret_name": secret_name,
            "secret_value": base64.b64encode(secret_value.encode("utf-8")).decode(
                "utf-8"
            ),
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

        manager = get_user_secret_manager()
        logger.info("Creating a user secret")
        secret_value = (
            base64.b64decode(secret_request.secret_value.encode("ascii")).decode(
                "utf-8"
            )
            if secret_request.encoding == "base64"
            else secret_request.secret_value
        )
        manager.create_user_secret(
            user_id=user_id,
            secret_name=secret_request.secret_name,
            secret_value=secret_value,
        )
        return {"result": "success"}
    except NotImplementedError as e:
        # The configured user-secret backend is read-only (e.g. the env backend).
        logger.error("User secret backend is read-only: %s", e)
        raise HTTPException(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail=repr(e)
        )
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

        manager = get_user_secret_manager()
        logger.info("Updating a user secret")
        secret_value = (
            base64.b64decode(secret_request.secret_value.encode("ascii")).decode(
                "utf-8"
            )
            if secret_request.encoding == "base64"
            else secret_request.secret_value
        )
        manager.update_user_secret(user_id, secret_name, secret_value)
        return {"result": "success"}
    except NotImplementedError as e:
        # The configured user-secret backend is read-only (e.g. the env backend).
        logger.error("User secret backend is read-only: %s", e)
        raise HTTPException(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail=repr(e)
        )
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

        manager = get_user_secret_manager()
        logger.info("Deleting a user secret")
        manager.delete_user_secret(user_id, secret_name)
        return {"result": "success"}
    except NotImplementedError as e:
        # The configured user-secret backend is read-only (e.g. the env backend).
        logger.error("User secret backend is read-only: %s", e)
        raise HTTPException(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail=repr(e)
        )
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
