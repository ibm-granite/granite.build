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

import importlib
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Optional, Self, Type

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
        if len(SpaceSecretManager.spacesecretmanagers) != 0:
            return
        package_dir = os.path.dirname(__file__)

        for filename in os.listdir(package_dir):
            if (
                filename.endswith(".py")
                and filename != "__init__.py"
                and filename != os.path.basename(__file__)
            ):
                spacesecretmanager_modulename = filename[:-3]
                spacesecretmanager_key_name = spacesecretmanager_modulename[
                    : -len(SpaceSecretManager.__name__)
                ].lower()
                spacesecretmanager_typename = (
                    spacesecretmanager_key_name.capitalize()
                    + SpaceSecretManager.__name__
                )
                try:
                    module = importlib.import_module(
                        f".{spacesecretmanager_modulename}",
                        package="gbserver.spacesecretmanager",
                    )
                    if hasattr(module, spacesecretmanager_typename):
                        handler_class = getattr(module, spacesecretmanager_typename)
                        if isinstance(handler_class, type) and issubclass(
                            handler_class, SpaceSecretManager
                        ):
                            SpaceSecretManager.spacesecretmanagers[
                                spacesecretmanager_key_name
                            ] = handler_class
                        else:
                            logger.error(
                                "Ignoring %s since it is not a subclass of SpaceSecretManager class",
                                spacesecretmanager_typename,
                            )
                    else:
                        logger.error(
                            "Module %s does not contain expected space secret manager type class %s",
                            spacesecretmanager_modulename,
                            spacesecretmanager_typename,
                        )
                except ImportError as e:
                    logger.error(
                        "Error importing module %s: %s", spacesecretmanager_typename, e
                    )
                except Exception as e:
                    logger.error(
                        "Error loading space secret manager type from %s: %s",
                        spacesecretmanager_typename,
                        e,
                    )
