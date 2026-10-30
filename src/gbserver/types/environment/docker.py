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
