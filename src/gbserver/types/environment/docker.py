"""Types related to the Docker execution environment."""

from typing import Dict, Optional

from pydantic import Field

from gbserver.types.environment.environment import StepEnvConfig


class StepDockerConfig(StepEnvConfig):
    """Docker-specific config from the step.yaml docker: section."""

    image: Optional[str] = None
    env: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    registry_auth: Optional[Dict[str, str]] = None
    pull_policy: str = "if-not-present"
