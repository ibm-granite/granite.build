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

"""
Hybrid secret manager that chains multiple secret managers with priority-based fallback.
"""

from typing import Any, Dict, List, Optional, Self

from gbserver.spacesecretmanager.spacesecretmanager import SpaceSecretManager
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class HybridSpaceSecretManager(SpaceSecretManager):
    """Secret manager that chains multiple secret managers with priority-based fallback.

    This manager accepts a list of manager configurations in priority order and attempts
    to fetch secrets from each manager in sequence until one succeeds. For writes, it
    delegates to the first writable manager in the chain.

    Configuration example:
        {
            "type": "hybrid",
            "config": {
                "managers": [
                    {"type": "env", "config": {}},
                    {"type": "local", "config": {"secrets_dir": "~/.gbserver/secrets"}},
                    {"type": "ibmcloud", "config": {"service_url": "..."}}
                ]
            }
        }

    Behavior:
        - get_secret(): Tries each manager until one returns non-empty, stops at first success
        - get_secrets(): Merges all managers with first-wins precedence
        - create_secret(): Writes to first writable manager (that doesn't raise NotImplementedError)
        - Robust error handling: logs exceptions but continues to next manager
    """

    def __init__(
        self: Self, uri: str, managers: List[Dict[str, Any]], **kwargs
    ) -> None:
        """Initialize the hybrid secret manager with a chain of managers.

        Args:
            uri: URI identifier for the secret manager
            managers: List of manager configurations in priority order.
                     Each config is a dict with "type" and optional "config" keys.
                     Example: [{"type": "env", "config": {}}, {"type": "local", "config": {...}}]
            **kwargs: Additional keyword arguments passed to parent class

        Note:
            If a manager fails to initialize, it logs an error but continues with remaining managers.
            If all managers fail to initialize, the chain will be empty and all operations return empty.
        """
        super().__init__(uri=uri, **kwargs)
        self.managers: List[SpaceSecretManager] = []
        self.manager_types: List[str] = []  # Track types for logging

        if not managers:
            logger.warning(
                "HybridSpaceSecretManager initialized with empty manager list"
            )
            return

        logger.info(
            "Initializing HybridSpaceSecretManager with %d manager(s)", len(managers)
        )

        for idx, manager_config in enumerate(managers):
            manager_type = manager_config.get("type")
            if not manager_type:
                logger.error(
                    "Manager configuration at index %d missing 'type' field: %s",
                    idx,
                    manager_config,
                )
                continue

            # Prevent infinite recursion from nested hybrid managers
            if manager_type == "hybrid":
                logger.error(
                    "Nested hybrid managers are not supported (manager #%d). Skipping.",
                    idx + 1,
                )
                continue

            manager_kwargs = manager_config.get("config", {})

            try:
                # Instantiate the manager using the factory method
                manager = SpaceSecretManager.get_spacesecretmanager(
                    secret_manager_type=manager_type, uri=uri, **manager_kwargs
                )
                self.managers.append(manager)
                self.manager_types.append(manager_type)
                logger.info(
                    "Successfully initialized manager %d of type '%s'",
                    idx + 1,
                    manager_type,
                )
            except KeyError:
                logger.error(
                    "Unknown secret manager type '%s' at index %d. Available types: %s",
                    manager_type,
                    idx,
                    list(SpaceSecretManager.spacesecretmanagers.keys()),
                )
            except Exception as e:
                logger.error(
                    "Failed to initialize manager of type '%s' at index %d: %s",
                    manager_type,
                    idx,
                    e,
                    exc_info=True,
                )

        if not self.managers:
            logger.warning(
                "No managers successfully initialized in HybridSpaceSecretManager. "
                "All secret operations will return empty results."
            )
        else:
            logger.info(
                "HybridSpaceSecretManager initialized with %d active manager(s): %s",
                len(self.managers),
                ", ".join(self.manager_types),
            )

    def get_secret(
        self: Self,
        secret_name: str,
        secret_type: str = "arbitrary",
        secret_group_name: str = "",
    ) -> Any:
        """Get a secret by trying each manager in order until one returns a value.

        Stops at the first manager that returns a non-empty dict.

        Args:
            secret_name: Name of the secret to retrieve
            secret_type: Type of secret (passed to underlying managers)
            secret_group_name: Secret group name (passed to underlying managers)

        Returns:
            Dictionary with "value" key containing the secret value from the first
            successful manager, or empty dict if all managers return empty or fail
        """
        if not self.managers:
            logger.debug("No managers available to fetch secret '%s'", secret_name)
            return {}

        for idx, (manager, manager_type) in enumerate(
            zip(self.managers, self.manager_types)
        ):
            try:
                logger.debug(
                    "Trying manager %d (%s) for secret '%s'",
                    idx + 1,
                    manager_type,
                    secret_name,
                )
                result = manager.get_secret(
                    secret_name=secret_name,
                    secret_type=secret_type,
                    secret_group_name=secret_group_name,
                )

                # Check if result is non-empty
                if result:
                    logger.debug(
                        "Secret '%s' found in manager %d (%s)",
                        secret_name,
                        idx + 1,
                        manager_type,
                    )
                    return result
                else:
                    logger.debug(
                        "Secret '%s' not found in manager %d (%s), trying next",
                        secret_name,
                        idx + 1,
                        manager_type,
                    )
            except Exception as e:
                logger.warning(
                    "Manager %d (%s) raised exception while fetching secret '%s': %s. Trying next manager.",
                    idx + 1,
                    manager_type,
                    secret_name,
                    e,
                    exc_info=True,
                )

        # All managers returned empty or failed
        logger.debug(
            "Secret '%s' not found in any of %d manager(s)",
            secret_name,
            len(self.managers),
        )
        return {}

    def get_secrets(
        self: Self, username: Optional[str] = None
    ) -> Optional[Dict[str, str]]:
        """List all secrets by merging results from all managers.

        Uses first-wins precedence: if multiple managers have the same secret key,
        the value from the manager earlier in the chain takes precedence.

        Args:
            username: Username filter (passed to underlying managers)

        Returns:
            Dictionary of secret names to values, merged from all managers with
            first-wins precedence. Returns empty dict if no secrets found or all managers fail.
        """
        if not self.managers:
            logger.debug("No managers available to list secrets")
            return {}

        merged_secrets: Dict[str, str] = {}

        # Iterate managers in reverse order so earlier managers overwrite later ones
        for idx, (manager, manager_type) in reversed(
            list(enumerate(zip(self.managers, self.manager_types)))
        ):
            try:
                logger.debug(
                    "Fetching secrets from manager %d (%s)", idx + 1, manager_type
                )
                secrets = manager.get_secrets(username=username)

                if secrets:
                    # Merge with first-wins precedence (earlier managers override later)
                    # Since we're going in reverse, we update with current manager's secrets
                    # and they will be overwritten by earlier managers in subsequent iterations
                    merged_secrets.update(secrets)
                    logger.debug(
                        "Manager %d (%s) contributed %d secret(s)",
                        idx + 1,
                        manager_type,
                        len(secrets),
                    )
                else:
                    logger.debug(
                        "Manager %d (%s) returned no secrets", idx + 1, manager_type
                    )
            except Exception as e:
                logger.warning(
                    "Manager %d (%s) raised exception while listing secrets: %s. Continuing with other managers.",
                    idx + 1,
                    manager_type,
                    e,
                    exc_info=True,
                )

        logger.debug(
            "Merged %d total secret(s) from %d manager(s)",
            len(merged_secrets),
            len(self.managers),
        )
        return merged_secrets

    def create_secret(
        self: Self,
        secret_name: str,
        secret_value: str,
        secret_type: str = "arbitrary",
        secret_group_name: str = "",
    ) -> None:
        """Create a secret by delegating to the first writable manager.

        Tries each manager in order until one successfully writes the secret without
        raising NotImplementedError (read-only managers).

        Args:
            secret_name: Name of the secret to create
            secret_value: Value of the secret
            secret_type: Type of secret (passed to underlying manager)
            secret_group_name: Secret group name (passed to underlying manager)

        Raises:
            NotImplementedError: If all managers are read-only (raise NotImplementedError)
            Exception: If a manager raises an exception other than NotImplementedError,
                      it's logged and the next manager is tried. If all managers fail,
                      the last exception is raised.
        """
        if not self.managers:
            raise NotImplementedError(
                "Cannot create secret: no managers available in HybridSpaceSecretManager"
            )

        last_exception = None
        all_readonly = True

        for idx, (manager, manager_type) in enumerate(
            zip(self.managers, self.manager_types)
        ):
            try:
                logger.debug(
                    "Attempting to create secret '%s' in manager %d (%s)",
                    secret_name,
                    idx + 1,
                    manager_type,
                )
                manager.create_secret(
                    secret_name=secret_name,
                    secret_value=secret_value,
                    secret_type=secret_type,
                    secret_group_name=secret_group_name,
                )
                logger.info(
                    "Successfully created secret '%s' in manager %d (%s)",
                    secret_name,
                    idx + 1,
                    manager_type,
                )
                return  # Success, stop here
            except NotImplementedError as e:
                logger.debug(
                    "Manager %d (%s) is read-only, trying next manager",
                    idx + 1,
                    manager_type,
                )
                if all_readonly:  # Only track if we haven't seen a real exception yet
                    last_exception = e
                continue
            except Exception as e:
                logger.warning(
                    "Manager %d (%s) failed to create secret '%s': %s. Trying next manager.",
                    idx + 1,
                    manager_type,
                    secret_name,
                    e,
                    exc_info=True,
                )
                all_readonly = False
                last_exception = e
                continue

        # All managers failed or were read-only
        if all_readonly:
            raise NotImplementedError(
                f"Cannot create secret '{secret_name}': all {len(self.managers)} manager(s) "
                f"in the chain are read-only. Manager types: {', '.join(self.manager_types)}"
            )
        else:
            error_msg = (
                f"Failed to create secret '{secret_name}' in any of {len(self.managers)} manager(s). "
                f"Manager types: {', '.join(self.manager_types)}"
            )
            logger.error("%s. Last error: %s", error_msg, last_exception)
            raise RuntimeError(
                f"{error_msg}. Last error: {last_exception}"
            ) from last_exception
