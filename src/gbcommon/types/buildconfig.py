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

"""Buildconfig module."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Type

from pydantic import Field

from gbcommon.types.validation import GBValidationErrors, GBValidationErrorType

from .config import Config
from .constants import BUILD_YAML_BASE_KEYS, CURRENT_BUILD_YAML_VERSION

logger = logging.getLogger(__name__)

BUILD_FILENAME = "build.yaml"
BUILD_FILE_BASE_KEY = "granite.build"


class InvalidTarget(Exception):
    """Indicates that the step is invalid."""


class BuildFailure(Exception):
    """Indicates that the build failed."""


class BuildTargetInputConfig(Config):
    """Input Artifact definition"""

    uri: Optional[str] = None
    binding: Optional[str] = None
    # If True, the target will wait for push to be finished before triggering
    wait_for_push: Optional[bool] = False
    event: Optional[str] = None

    def get_binding_parts(self: Self) -> List[str]:
        """
        Given a binding: tunedmodel.tuned_checkpoint
        Returns: ["tunedmodel", "tuned_checkpoint"]
        """
        if self.binding is None:
            raise ValueError("no binding specified")
        return self.binding.split(".")


class BuildTargetOutputConfig(Config):
    """Output Artifact definition"""

    uri: Optional[str] = None


class BuildTargetStepConfig(Config):
    """A single target step in a build file."""

    step_uri: str
    launcher: Optional[str] = Field(default=None)
    config: Optional[dict[str, Any]] = Field(default_factory=dict)
    config_dir: Optional[str] = Field(default=None)


class BuildTargetConfig(Config):
    """A single target in a build file."""

    environment_uri: str
    inputs: Optional[Dict[str, BuildTargetInputConfig]] = Field(default_factory=dict)
    outputs: Optional[Dict[str, BuildTargetOutputConfig]] = Field(default_factory=dict)
    steps: List[BuildTargetStepConfig]
    dependency: Optional[set[str]] = Field(default_factory=set)


class BuildConfig(Config):
    """A single build config."""

    version: str = CURRENT_BUILD_YAML_VERSION
    name: str = ""
    targets: dict[str, BuildTargetConfig]

    @classmethod
    def from_yaml(
        cls: Type[Self],
        path: Path,
        basekey: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Dict,
    ) -> Self:
        if "basekeys" not in kwargs:
            kwargs["basekeys"] = BUILD_YAML_BASE_KEYS
        self = super().from_yaml(path=path, basekey=basekey, context=context, **kwargs)
        validate = kwargs.get("validate", True)
        assert isinstance(validate, bool), f"invalid validate flag: {validate}"
        if validate:
            self.my_validate().raise_if_invalid()
        return self

    def __validate_step_uris(self: Self) -> GBValidationErrors:
        logger.info("validating the step URIs of the build")
        errors = GBValidationErrors()
        for target_name, target in self.targets.items():
            logger.info("checking the env of the target: %s", target_name)
            logger.debug("target: %s", target)  # Avoid blowing travis log limits
            if target.environment_uri == "":
                errors.add(err=f"Target `{target_name}` the env URI is empty")
            logger.info("checking the steps of the target: %s", target_name)
            for i, step in enumerate(target.steps):
                if step.step_uri == "":
                    errors.add(err=f"Target `{target_name}` Step `{i}` URI is empty")
        return errors

    def __validate_target_inputs(self: Self) -> GBValidationErrors:
        logger.info("validating the inputs of the build")
        errors = GBValidationErrors()
        for target_name, target in self.targets.items():
            err_prefix = f"Target `{target_name}`:"
            logger.info("checking inputs of the target: %s", target_name)
            logger.debug("target: %s", target)  # Avoid blowing travis log limits
            if target.outputs is None:
                warning = f"{err_prefix} the target has no outputs"
                logger.warning(warning)
                errors.add_warning(warning=warning)
            if target.inputs is None:
                logger.warning("the target %s has no inputs", target_name)
                continue
            for target_input_name, target_input in target.inputs.items():
                err_prefix = f"Target `{target_name}` Input `{target_input_name}`:"
                logger.info("checking: %s %s", err_prefix, target_input)
                if target_input.uri is None and target_input.binding is None:
                    errors.add(f"{err_prefix} the input has no URI or binding")
                    continue
                if target_input.uri is not None and target_input.binding is not None:
                    errors.add(f"{err_prefix} the input has both a URI and a binding")
                    continue
                if target_input.uri is not None:
                    logger.info("checking the input URI: %s", target_input.uri)
                    if target_input.uri == "":
                        errors.add(f"{err_prefix} the target input URI is empty")
                    continue
                if target_input.binding is not None:
                    logger.info("checking if input binding is valid: %s", target_input.binding)
                    binding_target_name, binding_target_output_name = (
                        target_input.get_binding_parts()
                    )
                    if binding_target_name not in self.targets:
                        errors.add(
                            err=f"{err_prefix} binding to a non-existent target `{binding_target_name}`",
                            type=GBValidationErrorType.NOT_EXIST,
                        )
                        continue
                    build_config_target = self.targets[binding_target_name]
                    if (
                        build_config_target.outputs is None
                        or binding_target_output_name not in build_config_target.outputs
                    ):
                        errors.add(
                            err=f"{err_prefix} binding to a non-existent output `{binding_target_output_name}` of the target `{binding_target_name}`",
                            type=GBValidationErrorType.NOT_EXIST,
                        )
                    continue
        return errors

    def my_validate(self: Self) -> GBValidationErrors:
        """Validate the build config."""
        logger.info("validating the build config")
        errors = GBValidationErrors()
        if len(self.targets) == 0:
            errors.add("the build has no targets specified")
            return errors
        errors.add(self.__validate_step_uris())
        errors.add(self.__validate_target_inputs())
        logger.info("validated the build config and found %d errors", len(errors))
        return errors


class BuildRunConfig(Config):
    """A single build run config."""

    targets_to_run: dict[str, Any]
