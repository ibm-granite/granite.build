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

import os
from typing import Literal

from pydantic import ConfigDict, Field, field_validator

from gbserver.types.constants import (
    DEFAULT_GH_API_ENDPOINT,
    DEFAULT_ROOT_BUILDWATCHER_WORKSPACE_DIR,
    DEFAULT_ROOT_WORKSPACE_DIR,
    ENV_VAR_DEFAULT_BUILDRUNNER_TYPE,
    MIN_MONITORING_INTERVAL_SECONDS,
)
from gbserver.types.spacesconfig import CLISpacesConfig
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class BuildWatcherConfig(CLISpacesConfig):
    """The build watcher config."""

    # validate_assignment so the monitoring_interval floor (below) is enforced
    # even when the field is set after construction (the only way it is set today
    # — BuildWatcher mutates config.monitoring_interval on an already-built config).
    model_config = ConfigDict(validate_assignment=True)

    lh_max_retries: int = 3
    monitoring_interval: int = 5
    gh_api_endpoint: str = DEFAULT_GH_API_ENDPOINT
    workspace_dir: str = DEFAULT_ROOT_WORKSPACE_DIR
    watcher_workspace_dir: str = DEFAULT_ROOT_BUILDWATCHER_WORKSPACE_DIR
    # Resolved at instantiation (not import) via default_factory so the value
    # tracks the current GBSERVER_DEFAULT_BUILDRUNNER_TYPE env var — e.g. the
    # "thread" default that standalone mode sets after this module is already
    # imported. A frozen import-time default would otherwise stay "job" (k8s) in
    # standalone, since reloading the constants module can't update this default.
    buildrunner_type: Literal["thread", "process", "job"] = Field(
        default_factory=lambda: os.getenv(ENV_VAR_DEFAULT_BUILDRUNNER_TYPE, "job")  # type: ignore
    )

    @field_validator("monitoring_interval")
    @classmethod
    def _floor_monitoring_interval(cls, value: int) -> int:
        """Never poll faster than MIN_MONITORING_INTERVAL_SECONDS.

        A 0 (or sub-second) interval turns the BuildWatcher poll loop into a CPU
        busy-loop that also hammers storage, so clamp it up to the minimum.
        """
        if value < MIN_MONITORING_INTERVAL_SECONDS:
            logger.warning(
                "monitoring_interval %s is below the %ss minimum; using %s",
                value,
                MIN_MONITORING_INTERVAL_SECONDS,
                MIN_MONITORING_INTERVAL_SECONDS,
            )
            return MIN_MONITORING_INTERVAL_SECONDS
        return value
