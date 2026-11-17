#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

import os
from typing import Literal

from pydantic import Field

from gbserver.types.constants import (
    DEFAULT_GH_API_ENDPOINT,
    DEFAULT_ROOT_BUILDWATCHER_WORKSPACE_DIR,
    DEFAULT_ROOT_WORKSPACE_DIR,
    ENV_VAR_DEFAULT_BUILDRUNNER_TYPE,
    GBSERVER_STUCK_BUILD_TIMEOUT_SECONDS,
    MIN_MONITORING_INTERVAL_SECONDS,
)
from gbserver.types.spacesconfig import CLISpacesConfig


class BuildWatcherConfig(CLISpacesConfig):
    """The build watcher config."""

    lh_max_retries: int = 3
    # Floored at the minimum so a 0/negative interval (which would busy-loop the
    # poll loop and hammer storage) is rejected at construction.
    monitoring_interval: int = Field(default=5, ge=MIN_MONITORING_INTERVAL_SECONDS)
    # A build PENDING this long with no live runner is re-armed for re-dispatch.
    stuck_build_timeout_seconds: int = Field(
        default=GBSERVER_STUCK_BUILD_TIMEOUT_SECONDS, ge=1
    )
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
