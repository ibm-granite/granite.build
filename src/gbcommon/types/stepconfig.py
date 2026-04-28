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

from enum import StrEnum, auto
from gbcommon.types.config import Config
from pydantic import Field
from typing import Dict, List, Optional


class StepType(StrEnum):
    DATA_PROCESSING = auto()
    DATA_GENERATION = auto()
    TRAINING = auto()
    TUNING = auto()
    CUSTOM = auto()


class StepSetupConfig(Config):
    type: str
    config: Optional[Dict] = Field(default_factory=dict)


class StepLauncherConfig(Config):
    type: str
    setups: Optional[List[str]] = Field(default_factory=list)
    monitors: Optional[List[str]] = Field(default_factory=list)
    config: Optional[Dict] = Field(default=dict)


class StepMonitorConfig(Config):
    type: str
    config: Optional[Dict] = Field(default=dict)


class StepEnvironmentTypeConfig(Config):
    default_launcher: Optional[str] = Field(default=None)
    setups: Optional[Dict[str, StepSetupConfig]] = Field(default_factory=dict)
    launchers: Dict[str, StepLauncherConfig]
    monitors: Optional[Dict[str, StepMonitorConfig]] = Field(default_factory=dict)


class StepConfig(Config):
    name: str = ""
    type: str = "custom"
    config: Optional[Dict] = Field(default_factory=dict)
    environment_configs: Dict[str, StepEnvironmentTypeConfig]
