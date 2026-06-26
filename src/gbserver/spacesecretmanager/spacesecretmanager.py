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

"""The base class for all the secret managers."""

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Self, Type

from gbserver.utils.secretmanager_discovery import discover_secret_managers

logger = logging.getLogger(__name__)


class SpaceSecretManager(ABC):
    """The base class for all the secret managers."""

    spacesecretmanagers: ClassVar[Dict[str, Type[Self]]] = {}

    # Users should have access to all secrets in the space_name and in `public` space.
    # If space_name is empty, they should have access to only public space
    # A user can be part of many spaces, but they can access only one space's resources at a time
    def __init__(self: Self, uri: str, **kwargs) -> None:
        self.uri = uri

    @abstractmethod
    def get_secret(
        self: Self,
        secret_name: str,
        secret_type: str = "arbitrary",
        secret_group_name: str = "",
    ) -> Any:
        """
        Gets as input the secret_name and first checks if the secret exists in the space and if it does, return it.
        If it does not exist, check if exists in the public space, and return it.
        """

    @abstractmethod
    def get_secrets(
        self: Self, username: Optional[str] = None
    ) -> Optional[Dict[str, str]]:
        """
        List the secrets that belong to the space
        """

    @abstractmethod
    def create_secret(
        self: Self,
        secret_name: str,
        secret_value: str,
        secret_type: str = "arbitrary",
        secret_group_name: str = "",
    ) -> None:
        """
        Creates a secret in the secret manager
        """

    # The following management operations back the /space_secrets admin REST API.
    # They have read-only-safe defaults (writes raise NotImplementedError) so a
    # backend only needs to override what it supports; read-only backends (e.g.
    # env) inherit the correct "not writable" behavior.

    def list_secret_names(self: Self, secret_group_name: str = "") -> List[str]:
        """Return the names of the secrets managed for this space."""
        return list((self.get_secrets() or {}).keys())

    def update_secret(
        self: Self,
        secret_name: str,
        secret_value: str,
        secret_type: str = "arbitrary",
        secret_group_name: str = "",
    ) -> None:
        """Update an existing secret. Defaults to create_secret (upsert)."""
        self.create_secret(
            secret_name=secret_name,
            secret_value=secret_value,
            secret_type=secret_type,
            secret_group_name=secret_group_name,
        )

    def delete_secret(
        self: Self, secret_name: str, secret_group_name: str = ""
    ) -> None:
        """Delete a secret from the secret manager."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support deleting secrets"
        )

    @staticmethod
    def get_spacesecretmanager(
        secret_manager_type: str, uri: str, **kwargs
    ) -> "SpaceSecretManager":
        """Get a secret manager of the given type."""
        return SpaceSecretManager.spacesecretmanagers[secret_manager_type](
            uri=uri, **kwargs
        )

    @staticmethod
    def load_spacesecretmanagers() -> None:
        """Load all the secret managers."""
        discover_secret_managers(
            package_file=__file__,
            package_name="gbserver.spacesecretmanager",
            base_class=SpaceSecretManager,
            registry=SpaceSecretManager.spacesecretmanagers,
        )
