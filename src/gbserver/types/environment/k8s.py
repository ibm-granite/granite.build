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

from typing import List

from pydantic import Field

from gbserver.types.environment.environment import (
    EnvironmentVariableConfig,
    StepEnvConfig,
    StepSecretsConfig,
)

# Re-exported for backward compatibility: the shared secret-to-env-var mapping
# type now lives in the environment types module so every environment can share
# it. Imported here so existing ``from gbserver.types.environment.k8s import
# EnvironmentVariableConfig`` call sites keep working.
__all__ = [
    "EnvironmentVariableConfig",
    "StepK8sSecretsConfig",
    "StepK8sConfig",
]


class StepK8sSecretsConfig(StepSecretsConfig):
    """The k8s ``secrets`` section: the shared env-var allow-list plus the
    k8s-only image-pull-secret list (which has no analogue on other clouds)."""

    secret_names_to_use_as_pull_secret: List[str] = Field(default_factory=list)


class StepK8sConfig(StepEnvConfig):
    """Wrapper for k8s-specific config inside config, currently only secrets."""

    secrets: StepK8sSecretsConfig = Field(default_factory=StepK8sSecretsConfig)
