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

"""Buildwatcherconfig module."""

from typing import Literal

from gbserver.types.constants import (
    DEFAULT_GH_API_ENDPOINT,
    DEFAULT_ROOT_BUILDWATCHER_WORKSPACE_DIR,
    DEFAULT_ROOT_WORKSPACE_DIR,
    GBSERVER_DEFAULT_BUILDRUNNER_TYPE,
)
from gbserver.types.spacesconfig import CLISpacesConfig


class BuildWatcherConfig(CLISpacesConfig):
    """The build watcher config."""

    lh_max_retries: int = 3
    monitoring_interval: int = 5
    gh_api_endpoint: str = DEFAULT_GH_API_ENDPOINT
    workspace_dir: str = DEFAULT_ROOT_WORKSPACE_DIR
    watcher_workspace_dir: str = DEFAULT_ROOT_BUILDWATCHER_WORKSPACE_DIR
    buildrunner_type: Literal["thread", "process", "job"] = GBSERVER_DEFAULT_BUILDRUNNER_TYPE  # type: ignore
