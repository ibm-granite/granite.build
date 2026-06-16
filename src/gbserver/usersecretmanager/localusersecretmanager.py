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

"""Editable per-user secret manager backed by the local filesystem.

Each user's secrets live in a single base64-encoded file
``<secrets_dir>/<user_id>.yaml``. This is the default backend in standalone
mode and supports full create/read/update/delete.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Self

from gbserver.usersecretmanager.usersecretmanager import UserSecretManager
from gbserver.utils.logger import get_logger
from gbserver.utils.secretfile import load_secret_file, write_secret_file

logger = get_logger(__name__)

# Only allow filesystem-safe user ids so a user_id can never escape secrets_dir.
_SAFE_USER_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class LocalUserSecretManager(UserSecretManager):
    """Per-user secret manager that persists to ``<dir>/<user_id>.yaml``.

    Writes are read-modify-write on the per-user file without file locking, so
    concurrent writes to the same user could race and lose an update. This is
    acceptable for the standalone single-user use case; multi-client concurrent
    writes would need a file lock (follow-up if that becomes a real scenario).
    """

    def __init__(self: Self, uri: str = "", **kwargs) -> None:
        # The directory comes from the backend's `config: {dir: ...}`; read it via
        # kwargs to avoid shadowing the `dir` builtin in the signature.
        secrets_dir = kwargs.pop("dir", "")
        super().__init__(uri=uri, **kwargs)
        if not secrets_dir:
            raise ValueError(
                "LocalUserSecretManager requires a 'dir' config value "
                "(directory to store per-user secret files)."
            )
        self.dir = Path(secrets_dir)
        logger.info("Initialized LocalUserSecretManager with dir: %s", self.dir)

    def _user_file(self: Self, user_id: str) -> Path:
        if not user_id or not _SAFE_USER_ID.match(user_id):
            raise ValueError(f"Invalid user_id for local secret storage: {user_id!r}")
        return self.dir / f"{user_id}.yaml"

    def _load(self: Self, user_id: str) -> Dict[str, str]:
        target = self._user_file(user_id)
        if not target.exists():
            return {}
        return load_secret_file(target)

    def _save(self: Self, user_id: str, secrets: Dict[str, str]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        write_secret_file(self._user_file(user_id), secrets)

    def list_user_secrets(self: Self, user_id: str) -> List[str]:
        return list(self._load(user_id).keys())

    def get_user_secret(self: Self, user_id: str, secret_name: str) -> Optional[str]:
        return self._load(user_id).get(secret_name)

    def get_user_secrets(self: Self, user_id: str) -> Dict[str, str]:
        return self._load(user_id)

    def create_user_secret(
        self: Self, user_id: str, secret_name: str, secret_value: str
    ) -> None:
        # Create is an upsert: an existing secret of the same name is overwritten
        # (a warning is logged). This matches the prior IBM-backed behavior, where
        # the secret manager does not reject a duplicate name, and keeps POST
        # /(user|space)_secrets idempotent rather than erroring on re-create.
        secrets = self._load(user_id)
        if secret_name in secrets:
            logger.warning(
                "Secret '%s' already exists for user '%s'. Overriding value.",
                secret_name,
                user_id,
            )
        secrets[secret_name] = secret_value
        self._save(user_id, secrets)
        logger.info("Secret '%s' saved for user '%s'", secret_name, user_id)

    def update_user_secret(
        self: Self, user_id: str, secret_name: str, secret_value: str
    ) -> None:
        secrets = self._load(user_id)
        if secret_name not in secrets:
            raise ValueError(
                f"Secret '{secret_name}' does not exist for user '{user_id}'"
            )
        secrets[secret_name] = secret_value
        self._save(user_id, secrets)
        logger.info("Secret '%s' updated for user '%s'", secret_name, user_id)

    def delete_user_secret(self: Self, user_id: str, secret_name: str) -> None:
        secrets = self._load(user_id)
        if secret_name not in secrets:
            raise ValueError(
                f"Secret '{secret_name}' does not exist for user '{user_id}'"
            )
        del secrets[secret_name]
        self._save(user_id, secrets)
        logger.info("Secret '%s' deleted for user '%s'", secret_name, user_id)
