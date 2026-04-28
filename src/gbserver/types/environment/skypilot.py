"""Types related to the SkyPilot environment."""

from typing import Optional

from pydantic import Field

from gbserver.types.environment.environment import StepEnvConfig


class StepSkypilotConfig(StepEnvConfig):
    """Config specific to SkyPilot environments, extracted from step.yaml."""

    resources: dict = Field(default_factory=dict)
    setup: str = ""
    run: str = ""
    envs: dict = Field(default_factory=dict)
    file_mounts: dict = Field(default_factory=dict)
    idle_minutes_to_autostop: int = 10
    image_id: Optional[str] = None
