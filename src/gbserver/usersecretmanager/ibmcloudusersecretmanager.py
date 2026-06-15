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

"""Per-user secret manager backed by IBM Cloud Secrets Manager.

This is a thin adapter over the existing ``IbmcloudSpaceSecretManagerAdmin``,
preserving the exact behavior the REST API had before the ``UserSecretManager``
abstraction was introduced (single shared per-environment secret group, with
per-user name prefixing).
"""

from typing import Dict, List, Optional, Self

from gbserver.usersecretmanager.usersecretmanager import UserSecretManager
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class IbmcloudUserSecretManager(UserSecretManager):
    """Per-user secret manager wrapping ``IbmcloudSpaceSecretManagerAdmin``."""

    def __init__(self: Self, uri: str = "", service_url: str = "", **kwargs) -> None:
        super().__init__(uri=uri, **kwargs)
        # Imported lazily so the package loads cleanly without the IBM SDK.
        # pylint: disable=import-outside-toplevel
        from gbserver.spacesecretmanager.ibmcloudspacesecretmanager import (
            IbmcloudSpaceSecretManagerAdmin,
        )

        self._admin = IbmcloudSpaceSecretManagerAdmin(service_url=service_url, **kwargs)
        self._group = self._admin.get_secret_group_for_users()
        if self._group is None:
            raise ValueError("Secret group for users is unavailable")

    def list_user_secrets(self: Self, user_id: str) -> List[str]:
        all_names = self._admin.list_secret_names(self._group)
        return self._admin.filter_user_secrets(user_id, all_names)

    def get_user_secret(self: Self, user_id: str, secret_name: str) -> Optional[str]:
        name_for_user = self._admin.get_secret_name_for_user(user_id, secret_name)
        return self._admin.get_secret_value(self._group, name_for_user, False)

    def get_user_secrets(self: Self, user_id: str) -> Dict[str, str]:
        return self._admin.get_user_secret_values(user_id) or {}

    def create_user_secret(
        self: Self, user_id: str, secret_name: str, secret_value: str
    ) -> None:
        name_for_user = self._admin.get_secret_name_for_user(user_id, secret_name)
        self._admin.create_secret(
            secret_group_name=self._group,
            secret_name=name_for_user,
            secret_value=secret_value,
        )

    def update_user_secret(
        self: Self, user_id: str, secret_name: str, secret_value: str
    ) -> None:
        name_for_user = self._admin.get_secret_name_for_user(user_id, secret_name)
        self._admin.update_secret_value(self._group, name_for_user, secret_value)

    def delete_user_secret(self: Self, user_id: str, secret_name: str) -> None:
        name_for_user = self._admin.get_secret_name_for_user(user_id, secret_name)
        self._admin.delete_secret(self._group, name_for_user)
