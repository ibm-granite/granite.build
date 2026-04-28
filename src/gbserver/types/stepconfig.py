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

"""Types related to the step.yaml"""

from enum import StrEnum, auto
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from gbserver.types.config import Config
from gbserver.types.validation import GBValidationErrors, GBValidatorConfig


class StepType(StrEnum):
    """The type of the step."""

    DATA_PROCESSING = auto()
    DATA_GENERATION = auto()
    TRAINING = auto()
    TUNING = auto()
    CUSTOM = auto()


class StepSetupConfig(Config):
    """Config for a single setup of a step."""

    type: str
    config: Dict = Field(default_factory=dict)


class StepLauncherConfig(Config):
    """Config for a single launcher of a step."""

    type: str
    setups: Optional[List[str]] = Field(default_factory=list)
    monitors: Optional[List[str]] = Field(default_factory=list)
    # Names of env-specific validators to run if this launcher is selected.
    validators: List[str] = Field(default_factory=list)
    config: Optional[Dict] = Field(default_factory=dict)


class StepValidatorConfig(GBValidatorConfig):
    """Config for a step validator."""


class StepMonitorConfig(Config):
    """Config for a single monitor of a step."""

    type: str
    config: Optional[Dict] = Field(default_factory=dict)


class StepEnvironmentTypeConfig(Config):
    """Config for an environment where this step can run."""

    default_launcher: Optional[str] = None
    setups: Optional[Dict[str, StepSetupConfig]] = Field(default_factory=dict)
    launchers: Dict[str, StepLauncherConfig]
    # name -> env-specific validator definition
    validators: Dict[str, StepValidatorConfig] = Field(default_factory=dict)
    monitors: Dict[str, StepMonitorConfig] = Field(default_factory=dict)


class StepIOTypeEnum(StrEnum):
    """The type of input/output to a step."""

    DATASET = auto()  # table
    FILESET = auto()
    MODEL = auto()
    # ARRAY = auto() # Example: run eval on 10 checkpoints


class StepInputsAcceptEnum(StrEnum):
    """The type of sources that an input to a step accepts."""

    URI = auto()
    BINDING = auto()


class StepIODetailsConfig(BaseModel):
    """The details of a particular input/output of the step."""

    type: StepIOTypeEnum
    # only for inputs, empty list means it accepts everything
    accept: List[StepInputsAcceptEnum] = Field(default_factory=list)
    validation: GBValidationErrors = Field(default_factory=GBValidationErrors)


class StepIOConfig(BaseModel):
    """
    The inputs and outputs of the step.

    inputs:
        allow_unknown: false
        required:
            model_to_tune:
                type: model
                accept: ["uri", "binding"]
            tuning_data:
                type: dataset # table?
                accept: ["uri", "binding"]
            # example of a list of inputs
            list_of_lora_checkpoints_to_merge:
                # the step gets executed once for each input?
                type: array
                items:
                    type: model
                    accept: ["uri", "binding"]
        optional:
            validation_data:
                type: dataset # table?
                accept: ["uri", "binding"]
            data_config_path:
                type: fileset
                accept: ["uri", "binding"]
    outputs:
        allow_unknown: true # allow unknown/random artifacts to be created
        required: {}
        optional:
            tuned_checkpoint:
                type: model
    """

    allow_unknown: bool = True
    required: Dict[str, StepIODetailsConfig] = Field(default_factory=dict)
    optional: Dict[str, StepIODetailsConfig] = Field(default_factory=dict)


class StepConfig(Config):
    """The step.yaml file."""

    name: str = ""
    type: str = "custom"
    is_dry_run_compatible: bool = False
    inputs: StepIOConfig = StepIOConfig()
    outputs: StepIOConfig = StepIOConfig()
    config: Optional[Dict] = Field(default_factory=dict)
    # Environment agnostic static validators.
    # These validators will be run before the build starts.
    validators: Dict[str, StepValidatorConfig] = Field(default_factory=dict)
    # to avoid failure when step.yaml does not have environment_configs defined before merge
    environment_configs: Dict[str, StepEnvironmentTypeConfig] = Field(
        default_factory=dict
    )
