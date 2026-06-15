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

"""Convenience factory that builds the configured per-user secret manager.

Centralizes the env-var -> backend selection so the REST API and the build-time
secret resolution path stay consistent.
"""

import json

from gbserver.types.constants import (
    GBSERVER_USER_SECRET_DIR,
    GBSERVER_USER_SECRET_MANAGER,
    GBSERVER_USER_SECRET_MANAGER_CONFIG,
)
from gbserver.usersecretmanager.usersecretmanager import UserSecretManager
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


def get_user_secret_manager() -> UserSecretManager:
    """Build the configured per-user secret manager.

    The backend is selected by GBSERVER_USER_SECRET_MANAGER (ibmcloud / local /
    env). Backend-specific config comes from dedicated env vars, optionally
    overridden by a GBSERVER_USER_SECRET_MANAGER_CONFIG JSON blob.
    """
    manager_type = GBSERVER_USER_SECRET_MANAGER
    config: dict = {}
    if manager_type == "local":
        config["dir"] = GBSERVER_USER_SECRET_DIR
    if GBSERVER_USER_SECRET_MANAGER_CONFIG:
        try:
            config.update(json.loads(GBSERVER_USER_SECRET_MANAGER_CONFIG))
        except json.JSONDecodeError as e:
            logger.error("Invalid GBSERVER_USER_SECRET_MANAGER_CONFIG JSON: %s", e)
    return UserSecretManager.get_usersecretmanager(manager_type, **config)
