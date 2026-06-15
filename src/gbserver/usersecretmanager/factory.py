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
import os

from gbcommon.types.constants import get_gb_home_dir
from gbcommon.types.gbenvconfig import is_standalone
from gbserver.types.constants import (
    ENV_VAR_USER_SECRET_DIR,
    ENV_VAR_USER_SECRET_MANAGER,
    ENV_VAR_USER_SECRET_MANAGER_CONFIG,
)
from gbserver.usersecretmanager.usersecretmanager import UserSecretManager
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


def get_user_secret_manager() -> UserSecretManager:
    """Build the configured per-user secret manager.

    The backend is selected by GBSERVER_USER_SECRET_MANAGER when set; otherwise it
    defaults to the local (file) backend in standalone mode and to ibmcloud
    elsewhere. Backend-specific config comes from dedicated env vars, optionally
    overridden by a GBSERVER_USER_SECRET_MANAGER_CONFIG JSON blob (which, if set,
    must be valid JSON — an invalid value is a hard error rather than a silent
    fall-back to defaults).

    The selection (including the standalone default) and all config are resolved at
    call time, so standalone mode established at runtime — not just at import — picks
    the IBM-free local backend, and overrides are honored regardless of module
    import/reload ordering.
    """
    default_manager = "local" if is_standalone() else "ibmcloud"
    manager_type = os.getenv(ENV_VAR_USER_SECRET_MANAGER, default_manager)
    config: dict = {}
    if manager_type == "local":
        config["dir"] = os.getenv(
            ENV_VAR_USER_SECRET_DIR,
            os.path.join(get_gb_home_dir(), "user_secrets"),
        )
    config_blob = os.getenv(ENV_VAR_USER_SECRET_MANAGER_CONFIG, "")
    if config_blob:
        try:
            config.update(json.loads(config_blob))
        except json.JSONDecodeError as e:
            raise ValueError(
                "Invalid GBSERVER_USER_SECRET_MANAGER_CONFIG: must be valid JSON"
            ) from e
    return UserSecretManager.get_usersecretmanager(manager_type, **config)
