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

"""The base class and factory for per-user secret managers.

A ``UserSecretManager`` stores secrets scoped to a ``user_id``. Implementations
are auto-discovered from the files in this package: a module named
``<key>usersecretmanager.py`` exposing a ``<Key>UserSecretManager`` class is
registered under ``<key>`` (e.g. ``local`` -> ``LocalUserSecretManager``).

The interface is deliberately user-scoped (every method takes a ``user_id``) so
that the per-user namespacing is encapsulated inside each backend rather than
leaked to callers such as the REST API.
"""

import logging
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, List, Optional, Self, Type

from gbserver.utils.secretmanager_discovery import discover_secret_managers

logger = logging.getLogger(__name__)


class UserSecretManager(ABC):
    """The base class for all per-user secret managers."""

    usersecretmanagers: ClassVar[Dict[str, Type[Self]]] = {}

    def __init__(self: Self, uri: str = "", **kwargs) -> None:
        self.uri = uri

    @property
    def read_only(self: Self) -> bool:
        """Whether this backend rejects create/update/delete operations."""
        return False

    @abstractmethod
    def list_user_secrets(self: Self, user_id: str) -> List[str]:
        """Return the names of the secrets that belong to ``user_id``."""

    @abstractmethod
    def get_user_secret(self: Self, user_id: str, secret_name: str) -> Optional[str]:
        """Return the plain (decoded) value of a single user secret, or None."""

    @abstractmethod
    def get_user_secrets(self: Self, user_id: str) -> Dict[str, str]:
        """Return all of ``user_id``'s secrets as a ``{name: value}`` dict.

        Used by the build-time secret resolution path.
        """

    @abstractmethod
    def create_user_secret(
        self: Self, user_id: str, secret_name: str, secret_value: str
    ) -> None:
        """Create a new secret for ``user_id``."""

    @abstractmethod
    def update_user_secret(
        self: Self, user_id: str, secret_name: str, secret_value: str
    ) -> None:
        """Update an existing secret for ``user_id``."""

    @abstractmethod
    def delete_user_secret(self: Self, user_id: str, secret_name: str) -> None:
        """Delete a secret for ``user_id``."""

    @staticmethod
    def get_usersecretmanager(
        secret_manager_type: str, uri: str = "", **kwargs
    ) -> "UserSecretManager":
        """Get a user secret manager of the given type."""
        UserSecretManager.load_usersecretmanagers()
        if secret_manager_type not in UserSecretManager.usersecretmanagers:
            raise ValueError(
                f"Unknown user secret manager type '{secret_manager_type}'. "
                f"Available: {sorted(UserSecretManager.usersecretmanagers)}"
            )
        return UserSecretManager.usersecretmanagers[secret_manager_type](
            uri=uri, **kwargs
        )

    @staticmethod
    def load_usersecretmanagers() -> None:
        """Auto-discover and register all user secret manager implementations."""
        discover_secret_managers(
            package_file=__file__,
            package_name="gbserver.usersecretmanager",
            base_class=UserSecretManager,
            registry=UserSecretManager.usersecretmanagers,
        )
