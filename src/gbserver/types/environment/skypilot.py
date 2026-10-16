"""Types related to the SkyPilot environment."""

from typing import Optional

from pydantic import Field

from gbserver.types.environment.environment import StepEnvConfig, StepSecretsConfig


class StepSkypilotConfig(StepEnvConfig):
    """Config specific to SkyPilot environments, extracted from step.yaml.

    Mirrors the per-cloud step-config section used by the other environments
    (``config.lsf`` / ``config.k8s``): parsed from the step's ``config.skypilot``
    block. ``secrets`` is the shared, declarative secret->env-var allow-list;
    SkyPilot injects *only* these declared secrets into the launched task (see
    ``Skypilot.get_launch_env_vars``), never the whole secret bag.
    """

    secrets: StepSecretsConfig = Field(default_factory=StepSecretsConfig)
    resources: dict = Field(default_factory=dict)
    setup: str = ""
    run: str = ""
    envs: dict = Field(default_factory=dict)
    file_mounts: dict = Field(default_factory=dict)
    idle_minutes_to_autostop: int = 10
    image_id: Optional[str] = None
