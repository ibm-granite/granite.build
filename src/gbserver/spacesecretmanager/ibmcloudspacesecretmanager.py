#!/usr/bin/env python3

# Copyright LLM.build Authors
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

"""Fetch from the IBM Cloud Secrets manager."""

from __future__ import annotations

import base64
import os
import re
from typing import Any, Dict, List, Optional, Self
from urllib.parse import urlparse

from gbserver.utils.optional_imports import HAS_IBM_SDK

if HAS_IBM_SDK:
    import ibm_cloud_sdk_core
    from ibm_cloud_sdk_core.authenticators.iam_authenticator import IAMAuthenticator
    from ibm_secrets_manager_sdk.secrets_manager_v2 import (
        SecretsManagerV2,
        SecretsPager,
    )

from gbserver.spacesecretmanager.spacesecretmanager import SpaceSecretManager
from gbserver.types.constants import (
    GBSERVER_SECRET_GROUP_FOR_USERS,
    GBSERVER_SECRET_NAME_SEPARATOR,
)
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

PUBLIC_SECRET_GROUP = "gbspace-public"


class IbmcloudSpaceSecretManager(SpaceSecretManager):
    """Fetch from the IBM Cloud Secrets manager."""

    def __init__(
        self: Self,
        uri: str,
        service_url: Optional[str] = None,
        service_apikey: Optional[str] = None,
        **kwargs,
    ) -> None:
        if not HAS_IBM_SDK:
            raise ImportError(
                "IBM Cloud SDK required. Install with: pip install gbserver[ibmcloud]"
            )
        super().__init__(uri=uri)
        self.service_url = service_url
        apikey = (
            os.getenv("IBM_CLOUD_API_KEY")
            if service_apikey is None or service_apikey == ""
            else service_apikey
        )
        assert apikey, "IBM_CLOUD_API_KEY env is required"
        self.service_url = (
            os.getenv("IBM_CLOUD_SECRETS_MANAGER_SERVICE_URL")
            if service_url is None or service_url == ""
            else service_url
        )
        assert self.service_url != "", "IBM_CLOUD_SECRETS_MANAGER_SERVICE_URL env required"
        self.secrets_manager_service = SecretsManagerV2(
            authenticator=IAMAuthenticator(apikey=apikey)
        )
        self.secrets_manager_service.set_service_url(self.service_url)
        self.secrets_manager_service.enable_retries(
            max_retries=10, retry_interval=90
        )  # We have seen the IBM Secret Manager issue 500s
        status_forcelist = self.secrets_manager_service.retry_config.status_forcelist
        logger.info("adding 403 to the list of HTTP status codes to retry on")
        if isinstance(status_forcelist, list):
            if 403 not in status_forcelist:
                status_forcelist.append(403)
        elif isinstance(status_forcelist, set):
            status_forcelist.add(403)
        self.secret_groups = self.get_secret_groups()

    def get_secrets_with_groups(
        self: Self,
        username: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return secrets config following the local file format.
        Returns:
        {
        SECRET_NAME: {
            "payload": value,
            "secret_group": group_name,
            "labels": [...]
        }
        }
        """
        secrets: Dict[str, Dict[str, Any]] = {}

        if not self.secret_groups:
            return secrets

        def _add(groups):
            pager = SecretsPager(
                client=self.secrets_manager_service,
                limit=10,
                sort="created_at",
                groups=groups,
            )
            while pager.has_next():
                for secret in pager.get_next():
                    name = secret["name"]
                    secret_type = secret["secret_type"]
                    group_id = secret["secret_group_id"]
                    group_name = self.secret_groups[group_id]

                    value = self.get_secret(
                        secret_name=name,
                        secret_type=secret_type,
                        secret_group_name=group_name,
                    )

                    if not value:
                        pass

                    labels: List = secret.get("labels")
                    if not labels:
                        # if labels not found -> assign encode:base64 since
                        # we are expecting payload to be base64 encoded
                        labels = ["encode:base64"]
                    else:
                        # if labels exist, append "encode:base64" if not already present
                        # we need this because payload is base64 encoded
                        if "encode:base64" not in labels:
                            labels.append("encode:base64")

                    if value:
                        clean_name = name.removeprefix(group_name + "-")
                        secrets[clean_name] = {
                            "payload": value,
                            "secret_group": group_name,
                            **({"labels": labels} if labels else {}),
                        }

        public = [g for g, n in self.secret_groups.items() if n == PUBLIC_SECRET_GROUP]
        others = [g for g, n in self.secret_groups.items() if n != PUBLIC_SECRET_GROUP]

        if public:
            _add(public)
        if others:
            _add(others)

        if username:
            # optional: merge per-user secrets here later
            pass

        return secrets

    def get_secret_groups(
        self: Self,
    ) -> Dict[str, str]:  # secret_group_id : secret_group_name
        """Fetch the secret groups that match the space config URI."""
        response = self.secrets_manager_service.list_secret_groups()
        secret_group_collection = response.get_result()
        groups: Dict[str, str] = {}
        public_group_id = None
        if secret_group_collection is not None:
            for secret_group in secret_group_collection["secret_groups"]:
                description_lines = str(secret_group["description"]).split("\n")
                for description_line in description_lines:
                    line = description_line.strip()
                    if line is None or line == "":
                        continue
                    match = re.search(line, self.uri)
                    if match:
                        if secret_group["name"] == PUBLIC_SECRET_GROUP:
                            public_group_id = secret_group["id"]
                        else:
                            groups[secret_group["id"]] = secret_group["name"]
        if public_group_id is not None:
            groups[public_group_id] = PUBLIC_SECRET_GROUP
        if len(groups) == 0:
            logger.warning("No matching secret group not found for space: %s", self.uri)
        return groups

    def get_secret(
        self: Self,
        secret_name: str,
        secret_type: str = "arbitrary",
        secret_group_name: str = "",
    ) -> Any:
        response = self.secrets_manager_service.get_secret_by_name_type(
            secret_type=secret_type,
            name=secret_name,
            secret_group_name=secret_group_name,
        )
        secret = response.get_result()
        payload = secret.get("payload")
        if payload is None:
            logger.error("Unable to access payload for secret %s . Ignoring.", secret_name)
            return None
        labels = secret.get("labels")
        if labels is not None and "encode:base64" in labels:
            value = base64.b64decode(payload.encode("utf-8")).decode("utf-8")
        else:
            value = payload
        return value

    def get_secrets(self: Self, username: Optional[str] = None) -> Optional[Dict[str, str]]:
        secrets = self.get_space_secrets() or {}
        if username is not None:
            # Apply per-user secrets. User secrets have priority over space secrets
            per_user_secrets = self.get_user_secrets(username) or {}
            secrets = secrets | per_user_secrets
        return secrets

    def create_secret(
        self: Self,
        secret_name: str,
        secret_value: str,
        secret_type: str = "arbitrary",
        secret_group_name: str = "",
    ) -> None:
        """Create a secret under a security group"""
        raise NotImplementedError(
            "create_secret is not implemented by IbmcloudSpaceSecretManager,"
            + " use IbmcloudSpaceSecretManagerAdmin"
        )

    def get_space_secrets(self: Self) -> Optional[Dict[str, str]]:
        """
        Fetch the secrets that are part of the secret groups the space has access to."""
        if self.secret_groups is None or len(self.secret_groups) == 0:
            return None
        try:
            # secret_group_name, self.secret_group_id
            space_secrets: Dict[str, str] = {}

            def _add_secrets_from_groups(space_secrets: Dict, groups: List) -> Dict:
                pager = SecretsPager(
                    client=self.secrets_manager_service,
                    limit=10,
                    sort="created_at",
                    search=None,
                    groups=groups,
                )
                while pager.has_next():
                    next_page = pager.get_next()
                    assert next_page is not None
                    for secret in next_page:
                        name: str = secret.get("name")
                        secret_type = secret.get("secret_type")
                        secret_group_id = secret.get("secret_group_id")
                        secret_group_name = self.secret_groups[secret_group_id]
                        secret = self.get_secret(
                            secret_name=name,
                            secret_type=secret_type,
                            secret_group_name=secret_group_name,
                        )
                        if secret is not None and secret != "":
                            secret_name = (
                                name
                                if secret_group_name is None
                                else name.removeprefix(secret_group_name + "-")
                            )
                            if secret_name in space_secrets:
                                logger.warning(
                                    "The secret named '%s' already exists. Replacing with the secret from the group '%s'",
                                    secret_name,
                                    secret_group_name,
                                )
                            space_secrets[secret_name] = secret
                return space_secrets

            public_group = [
                g for g, gvalue in self.secret_groups.items() if gvalue == PUBLIC_SECRET_GROUP
            ]
            other_groups = [
                g for g, gvalue in self.secret_groups.items() if gvalue != PUBLIC_SECRET_GROUP
            ]
            # Evaluate PUBLIC_SECRET_GROUP first so that it's least prioritized (the secrets loaded later will override the early ones)
            if len(public_group) > 0:
                space_secrets = _add_secrets_from_groups(space_secrets, public_group)
            if len(other_groups) > 0:
                space_secrets = _add_secrets_from_groups(space_secrets, other_groups)
            return space_secrets
        except ibm_cloud_sdk_core.api_exception.ApiException as e:
            raise e

    def get_user_secrets(self: Self, username: str) -> Optional[Dict[str, str]]:
        """Obtain user secrets"""
        manager = IbmcloudSpaceSecretManagerAdmin(service_url=self.service_url)
        user_secrets = manager.get_user_secret_values(username)
        return user_secrets


class IbmcloudSpaceSecretManagerAdmin:
    """This class provides direct SecretManager operations beyond what the common SpaceSecretManager interface provides.
    Encaptulate secret group IDs and secret IDs which are internal to the IBM Secret Manager. Secret groups and secrets
    are referenced by their name.
    The code instantiating this class should be extremely careful not to compromize the security design, as it allows
    direct read/write/update/delete access to all security groups and secrets under the account.
    https://github.com/IBM/secrets-manager-python-sdk
    https://cloud.ibm.com/apidocs/secrets-manager/secrets-manager-v2
    """

    def __init__(
        self: Self,
        service_url: Optional[str] = None,
        service_apikey: Optional[str] = None,
        **kwargs,
    ) -> None:
        if not HAS_IBM_SDK:
            raise ImportError(
                "IBM Cloud SDK required. Install with: pip install gbserver[ibmcloud]"
            )
        self.service_url = service_url
        apikey = (
            os.getenv("IBM_CLOUD_API_KEY")
            if service_apikey is None or service_apikey == ""
            else service_apikey
        )
        assert apikey, "IBM_CLOUD_API_KEY env is required"
        self.service_url = (
            os.getenv("IBM_CLOUD_SECRETS_MANAGER_SERVICE_URL")
            if service_url is None or service_url == ""
            else service_url
        )
        assert self.service_url != "", "IBM_CLOUD_SECRETS_MANAGER_SERVICE_URL env required"
        self.secrets_manager_service = SecretsManagerV2(
            authenticator=IAMAuthenticator(apikey=apikey)
        )
        self.secrets_manager_service.set_service_url(self.service_url)
        self.secret_groups_cache = None

    def get_all_secret_groups(self: Self, refresh: bool = False):
        """Get all of the secret groups."""
        if refresh or self.secret_groups_cache is None:
            response = self.secrets_manager_service.list_secret_groups()
            self.secret_groups_cache = response.get_result()["secret_groups"]
        return self.secret_groups_cache

    def get_secret_group_by_name(self: Self, secret_group_name: str, refresh: bool = False):
        """Find a secret group entry by secret name. It throws an exception if the name isn't unique"""
        secret_groups = self.get_all_secret_groups()
        if secret_groups is not None:
            my_secret_groups = list(filter(lambda g: g["name"] == secret_group_name, secret_groups))
            if len(my_secret_groups) > 1:
                raise ValueError(f"len(my_secret_groups) > 1: {len(my_secret_groups)}")
            if len(my_secret_groups) == 1:
                return my_secret_groups[0]
            return None
        return None

    def create_secret_group(self: Self, secret_group_name: str, description: Optional[str] = None):
        """Create a secret group."""
        if self.get_secret_group_by_name(secret_group_name=secret_group_name):
            raise ValueError(f"the secret group already exists: {secret_group_name}")
        self.secrets_manager_service.create_secret_group(
            name=secret_group_name, description=description
        )
        self.get_all_secret_groups(refresh=True)

    def update_secret_group_description(
        self: Self, secret_group_name: str, description: Optional[str] = None
    ):
        """Update a secret group description. Because we refer to the group by name, only description can be updated"""
        secret_group = self.get_secret_group_by_name(secret_group_name=secret_group_name)
        if not secret_group:
            raise ValueError(f"there is no secret group with the name: {secret_group_name}")
        secret_group_id = secret_group["id"]
        if description != secret_group["description"]:
            update_patch = {"description": description}
            self.secrets_manager_service.update_secret_group(
                id=secret_group_id, secret_group_patch=update_patch
            )
            self.get_all_secret_groups(refresh=True)

    def delete_secret_group(self: Self, secret_group_name: str):
        """Delete a secret group. It cannot be deleted if there'a any secret under the group."""
        secret_group = self.get_secret_group_by_name(secret_group_name=secret_group_name)
        if not secret_group:
            raise ValueError(f"there is no secret group with the name: {secret_group_name}")
        self.secrets_manager_service.delete_secret_group(id=secret_group["id"])
        self.get_all_secret_groups(refresh=True)

    def create_secret(
        self: Self,
        secret_name: str,
        secret_value: str,
        secret_type: str = "arbitrary",
        secret_group_name: str = "",
    ) -> None:
        """Create a secret under a security group"""
        secret_group = self.get_secret_group_by_name(secret_group_name)
        secret_prototype = {
            "name": secret_name,
            "secret_group_id": secret_group["id"],
            "secret_type": secret_type,
            "payload": secret_value,
        }
        self.secrets_manager_service.create_secret(secret_prototype)

    def list_secrets(self: Self, secret_group_name: str):
        """List secrets under a security group"""
        secret_group = self.get_secret_group_by_name(secret_group_name)
        return self.secrets_manager_service.list_secrets(groups=[secret_group["id"]]).get_result()[
            "secrets"
        ]

    def list_secret_names(self: Self, secret_group_name: str):
        """List the name of secrets under a security group"""
        return list(
            map(
                lambda x: x["name"],
                self.list_secrets(secret_group_name=secret_group_name),
            )
        )

    def get_secret(self: Self, secret_group_name: str, secret_name: str):
        """Get a secret under a security group"""
        return self.secrets_manager_service.get_secret_by_name_type(
            name=secret_name,
            secret_group_name=secret_group_name,
            secret_type="arbitrary",
        ).get_result()

    def get_secret_value(
        self: Self, secret_group_name: str, secret_name: str, encode: bool = False
    ):
        """Get a secret value under a security group"""
        try:
            secret_value = self.get_secret(
                secret_group_name=secret_group_name, secret_name=secret_name
            )["payload"]
            if secret_value:
                if encode:
                    return base64.b64encode(secret_value.encode("utf-8")).decode("utf-8")
                return secret_value
            return None
        except Exception:
            return None

    def update_secret_value(
        self: Self, secret_group_name: str, secret_name: str, secret_value: str
    ):
        """Get a value of the a secret value under a security group"""
        secret = self.get_secret(secret_group_name=secret_group_name, secret_name=secret_name)
        if not secret:
            raise ValueError(f"failed to get the secret {secret_name} in group {secret_group_name}")
        secret_prototype = {
            "payload": secret_value,
        }
        self.secrets_manager_service.create_secret_version(
            secret_id=secret["id"], secret_version_prototype=secret_prototype
        )

    def delete_secret(self: Self, secret_group_name: str, secret_name: str):
        """Delete a secret value under a security group"""
        secret = self.get_secret(secret_group_name=secret_group_name, secret_name=secret_name)
        if not secret:
            raise ValueError(f"failed to get the secret {secret_name} in group {secret_group_name}")
        self.secrets_manager_service.delete_secret(id=secret["id"])

    def get_secret_group_for_space(self: Self, space):
        """Obtain the secret group name for a given space. Unlike IbmcloudSpaceSecretManager, the approach here doesn't make use of regular expressions stored in secret group description. Instead, it just assumes that the secret name is the same as the git repo name."""
        git_repo_uri = space["git_repo_uri"]
        repo_name = urlparse(git_repo_uri).path.split("/")[-1]
        return repo_name

    def get_secret_group_for_users(self: Self):
        """Obtain the secret group name for per-user secrets.
        The current design choice is to store per-user secrets in a single common secret group per environment (prod/staging/dev)
        """
        return GBSERVER_SECRET_GROUP_FOR_USERS

    def get_secret_name_for_user(self: Self, user_id: str, secret_name: str):
        """Get a secret name to use for user-specific secrets."""
        if GBSERVER_SECRET_NAME_SEPARATOR in user_id:
            # A safeguard to prevent name collision
            raise ValueError(f"GBSERVER_SECRET_NAME_SEPARATOR in user_id: {user_id}")
        group_prefix = self.get_secret_group_for_users()
        return f"{group_prefix}-{user_id}{GBSERVER_SECRET_NAME_SEPARATOR}{secret_name}"

    def filter_user_secrets(self: Self, user_id: str, secret_names: List[str]):
        """Filter the secrets belonging to user_id and the return the list stripping the prefix"""
        group_prefix = self.get_secret_group_for_users()
        prefix = f"{group_prefix}-{user_id}{GBSERVER_SECRET_NAME_SEPARATOR}"
        return list(s.replace(prefix, "") for s in secret_names if s.startswith(prefix))

    def get_user_secret_values(self: Self, user_id: str):
        """Runtime function to return the list of secrets (key-value pairs). This method should return the secret names after
        applying the naming conventions"""
        secret_group_name = self.get_secret_group_for_users()
        if secret_group_name is None:
            raise ValueError("Secret group is unavailable")
        secrets_all_users = self.list_secret_names(secret_group_name)
        user_secret_names = self.filter_user_secrets(user_id, secrets_all_users)
        secrets = {}
        for secret_name in user_secret_names:
            secret_name_for_user = self.get_secret_name_for_user(user_id, secret_name)
            secret_value = self.get_secret_value(secret_group_name, secret_name_for_user, False)
            secrets[secret_name] = secret_value
        return secrets
