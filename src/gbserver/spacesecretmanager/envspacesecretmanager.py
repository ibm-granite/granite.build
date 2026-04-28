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
Secret manager that reads from environment variables.
"""

import os
from typing import Any, Dict, Optional, Self

from gbserver.spacesecretmanager.spacesecretmanager import SpaceSecretManager
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class EnvSpaceSecretManager(SpaceSecretManager):
    """Secret manager that fetches secrets from environment variables.

    This manager reads secrets from environment variables with a configurable prefix
    (default: GBSERVER_SECRET_). It supports flexible name matching, transforming
    secret names to match environment variable naming conventions.

    Examples:
        - secret_name "api_key" matches "GBSERVER_SECRET_API_KEY"
        - secret_name "api-key" matches "GBSERVER_SECRET_API_KEY"
        - secret_name "api.key" matches "GBSERVER_SECRET_API_KEY"
        - Matching is case-insensitive

    This is a read-only secret manager; create_secret() is not supported.
    """

    def __init__(
        self: Self, uri: str, prefix: str = "GBSERVER_SECRET_", **kwargs
    ) -> None:
        """Initialize the environment variable secret manager.

        Args:
            uri: URI identifier for the secret manager
            prefix: Prefix for environment variable names (default: "GBSERVER_SECRET_")
            **kwargs: Additional keyword arguments passed to parent class
        """
        super().__init__(uri=uri, **kwargs)
        self.prefix = prefix
        logger.info("Initialized EnvSpaceSecretManager with prefix: %s", self.prefix)

    def _normalize_name(self: Self, name: str) -> str:
        """Normalize a secret name for environment variable matching.

        Converts the name to uppercase and replaces dashes and dots with underscores.

        Args:
            name: The secret name to normalize

        Returns:
            Normalized name suitable for environment variable matching
        """
        # Replace dashes and dots with underscores, convert to uppercase
        return name.replace("-", "_").replace(".", "_").upper()

    def get_secret(
        self: Self,
        secret_name: str,
        secret_type: str = "arbitrary",
        secret_group_name: str = "",
    ) -> Any:
        """Get a secret from environment variables.

        Attempts to find an environment variable matching the secret name with
        flexible name matching (case-insensitive, supports underscores/dashes/dots).

        Args:
            secret_name: Name of the secret to retrieve
            secret_type: Type of secret (unused, for compatibility)
            secret_group_name: Secret group name (unused, for compatibility)

        Returns:
            Dictionary with "value" key containing the secret value, or empty dict if not found
        """
        # Normalize the secret name for matching
        normalized_name = self._normalize_name(secret_name)
        env_var_name = f"{self.prefix}{normalized_name}"

        # Try to get the environment variable
        secret_value = os.environ.get(env_var_name)

        if secret_value is not None:
            logger.debug(
                "Found secret '%s' in environment variable '%s'",
                secret_name,
                env_var_name,
            )
            return {"value": secret_value}
        else:
            logger.debug(
                "Secret '%s' not found in environment (looked for '%s')",
                secret_name,
                env_var_name,
            )
            return {}

    def get_secrets(
        self: Self, username: Optional[str] = None
    ) -> Optional[Dict[str, str]]:
        """List all secrets with the configured prefix.

        Returns all environment variables that start with the configured prefix,
        with the prefix removed from the keys.

        Args:
            username: Username filter (unused, for compatibility)

        Returns:
            Dictionary of secret names (without prefix) to values
        """
        secrets = {}
        prefix_len = len(self.prefix)

        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                # Remove the prefix and store the secret
                secret_name = key[prefix_len:]
                secrets[secret_name] = value

        logger.debug(
            "Found %d secrets with prefix '%s'",
            len(secrets),
            self.prefix,
        )
        return secrets

    def create_secret(
        self: Self,
        secret_name: str,
        secret_value: str,
        secret_type: str = "arbitrary",
        secret_group_name: str = "",
    ) -> None:
        """Create a secret (not supported for environment variables).

        Environment variables are managed externally and cannot be created
        through this manager.

        Args:
            secret_name: Name of the secret
            secret_value: Value of the secret
            secret_type: Type of secret
            secret_group_name: Secret group name

        Raises:
            NotImplementedError: Always raised as this is a read-only manager
        """
        raise NotImplementedError(
            "Environment variable secrets are read-only. "
            "Please set environment variables directly in your shell or deployment configuration. "
            f"For secret '{secret_name}', set environment variable: "
            f"{self.prefix}{self._normalize_name(secret_name)}"
        )
