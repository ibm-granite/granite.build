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

"""Read-only per-user secret manager backed by environment variables.

Secrets are read from environment variables of the form
``<prefix><USER>_<NAME>`` (default prefix ``GBSERVER_USER_SECRET_``). Names are
normalized to uppercase with dashes/dots replaced by underscores, so a user
``alice`` secret ``api-key`` is read from ``GBSERVER_USER_SECRET_ALICE_API_KEY``.

This backend is read-only; create/update/delete raise ``NotImplementedError``.
"""

import os
from typing import Dict, List, Optional, Self

from gbserver.usersecretmanager.usersecretmanager import UserSecretManager
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_USER_SECRET_ENV_PREFIX = "GBSERVER_USER_SECRET_"


class EnvUserSecretManager(UserSecretManager):
    """Per-user secret manager that reads from environment variables."""

    def __init__(
        self: Self,
        uri: str = "",
        prefix: str = DEFAULT_USER_SECRET_ENV_PREFIX,
        **kwargs,
    ) -> None:
        super().__init__(uri=uri, **kwargs)
        self.prefix = prefix
        logger.info("Initialized EnvUserSecretManager with prefix: %s", self.prefix)

    @property
    def read_only(self: Self) -> bool:
        return True

    @staticmethod
    def _normalize(name: str) -> str:
        return name.replace("-", "_").replace(".", "_").upper()

    def _user_prefix(self: Self, user_id: str) -> str:
        return f"{self.prefix}{self._normalize(user_id)}_"

    def list_user_secrets(self: Self, user_id: str) -> List[str]:
        user_prefix = self._user_prefix(user_id)
        return [
            key[len(user_prefix) :] for key in os.environ if key.startswith(user_prefix)
        ]

    def get_user_secret(self: Self, user_id: str, secret_name: str) -> Optional[str]:
        env_var = f"{self._user_prefix(user_id)}{self._normalize(secret_name)}"
        return os.environ.get(env_var)

    def get_user_secrets(self: Self, user_id: str) -> Dict[str, str]:
        user_prefix = self._user_prefix(user_id)
        return {
            key[len(user_prefix) :]: value
            for key, value in os.environ.items()
            if key.startswith(user_prefix)
        }

    def _read_only_error(self: Self, secret_name: str) -> NotImplementedError:
        return NotImplementedError(
            "Environment-variable user secrets are read-only. Set the environment "
            f"variable directly for secret '{secret_name}'."
        )

    def create_user_secret(
        self: Self, user_id: str, secret_name: str, secret_value: str
    ) -> None:
        raise self._read_only_error(secret_name)

    def update_user_secret(
        self: Self, user_id: str, secret_name: str, secret_value: str
    ) -> None:
        raise self._read_only_error(secret_name)

    def delete_user_secret(self: Self, user_id: str, secret_name: str) -> None:
        raise self._read_only_error(secret_name)
