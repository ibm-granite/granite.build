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
Types related to the K8s environment.
"""

from typing import List, Optional

from pydantic import Field

from gbserver.types.config import Config
from gbserver.types.environment.environment import StepEnvConfig


class EnvironmentVariableConfig(Config):
    """Secrets to fetch from secret manager and expose as an env var"""

    env_name: Optional[str] = None
    secret_name: Optional[str] = None


class StepSecretsConfig(Config):
    """Secrets inside a step"""

    secret_names_to_use_as_pull_secret: List[str] = Field(default_factory=list)
    secret_names_to_use_as_env_variable: List[EnvironmentVariableConfig] = Field(
        default_factory=list
    )


class StepK8sConfig(StepEnvConfig):
    """Wrapper for k8s-specific config inside config, currently only secrets."""

    secrets: StepSecretsConfig = Field(default_factory=StepSecretsConfig)
