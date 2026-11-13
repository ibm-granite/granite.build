# Copyright Granite.Build Authors
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

"""Types related to the Docker execution environment."""

from typing import Dict, Optional

from pydantic import Field

from gbserver.types.environment.environment import StepEnvConfig, StepSecretsConfig


class StepDockerConfig(StepEnvConfig):
    """Docker-specific config from the step.yaml docker: section.

    ``secrets`` is the shared, declarative secret->env-var allow-list; Docker
    injects *only* these declared secrets into the container (see
    ``Docker._declared_secret_mappings``), as LSF, SkyPilot and K8s do.
    """

    secrets: StepSecretsConfig = Field(default_factory=StepSecretsConfig)
    image: Optional[str] = None
    env: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    registry_auth: Optional[Dict[str, str]] = None
    pull_policy: str = "if-not-present"
