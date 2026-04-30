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
The main parser for the build.yaml file.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Type

from pydantic import BaseModel, Field, field_validator, model_validator

from gbserver.types.config import Config
from gbserver.types.constants import (
    BUILD_YAML_BASE_KEYS,
    CURRENT_BUILD_YAML_VERSION,
)
from gbserver.types.validation import GBValidationErrors, GBValidationErrorType
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

BUILD_FILENAME = "build.yaml"
BUILD_RUN_YAML_FILENAME = "run.yaml"


class InvalidTarget(Exception):
    """Indicates that the step is invalid."""


class BuildFailure(Exception):
    """Indicates that the build failed."""


class BuildTargetInputMetadataConfig(Config):
    """Metadata regarding an input artifact."""

    checksum: str = ""
    other_locations: List[str] = Field(default_factory=list)


class BuildTargetInputConfig(Config):
    """Input Artifact definition"""

    uri: Optional[str] = None
    metadata: Optional[BuildTargetInputMetadataConfig] = None
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


class BuildTargetOutputEventSelectorsConfig(BaseModel):
    """
    Select the artifact event(s) that triggers this output.
    We match against the fields of the event's payload.
    """

    field_name: str = ""
    field_value: str = ""
    field_value_regex: str = ""
    # computed fields
    _field_value_regex: Optional[re.Pattern] = None

    def model_post_init(self: Self, context: Any, /) -> None:
        if self.field_name == "":
            raise ValueError("field name is missing")
        if self.field_value == "" and self.field_value_regex == "":
            raise ValueError("neither field value and regex were provided")
        if self.field_value != "" and self.field_value_regex != "":
            raise ValueError("both field value and regex were provided")
        if self.field_value_regex != "":
            self._field_value_regex = re.compile(self.field_value_regex)


class BuildTargetOutputPushConfig(Config):
    """Push configuration for an output artifact (build.yaml ``store_push`` block).

    Attributes:
        mode: The push mode (e.g. ``"hfstore"``).
        config: Mode-specific push configuration (e.g. ``{"hf": {"private": false}}``).
    """

    mode: Optional[str] = None
    config: Dict = Field(default_factory=dict)


class BuildTargetOutputConfig(Config):
    """Output Artifact definition

    Attributes:
        uri: Output artifact URI template.
        event_selectors: Event selector rules for matching artifact events.
        store_push: Optional per-output push configuration from build.yaml.
        space_name: Build space name; set at runtime, not parsed from build.yaml.
    """

    uri: Optional[str] = None
    event_selectors: List[BuildTargetOutputEventSelectorsConfig] = Field(default_factory=list)
    store_push: Optional[BuildTargetOutputPushConfig] = None
    # Populated at runtime from Build.space — not a build.yaml field.
    space_name: Optional[str] = None


class BuildTargetStepConfig(Config):
    """A single target step in a build file."""

    step_uri: Optional[str] = Field(
        default=None
    )  # defaults to "" to support empty step uri if `step_uri` field is present
    launcher: Optional[str] = Field(default=None)
    config: Optional[dict[str, Any]] = Field(default_factory=dict)
    config_dir: Optional[str] = Field(default=None)
    retry_enabled: Optional[bool] = None
    retry_transparently: Optional[bool] = None

    # FIELD VALIDATOR
    # Runs when step_uri as a field EXISTS in YAML, even if empty string or None.
    @field_validator("step_uri", mode="before")
    def apply_default_for_empty_value(cls, v):
        # If value is missing, empty string, or just whitespace -> use base step
        """Apply default for empty value."""
        if v is None or (isinstance(v, str) and v.strip() == ""):
            default_path = Path(os.path.abspath(__file__)).parent.parent / "builtins/steps/gbstep"
            default_step_uri = f"file://{default_path}"
            logger.info(
                f"[FIELD VALIDATOR - BuildTargetStepConfig] EMPTY STEP URI PROVIDED. DEFAULTING TO: {default_step_uri} ======="
            )
            return default_step_uri
        return v

    # MODEL VALIDATOR
    # Runs ALWAYS — even when step_uri field is missing entirely from build.yaml
    @model_validator(mode="after")
    def fill_missing_step_uri(self):
        """Fill missing step uri."""
        if not self.step_uri:
            default_path = Path(os.path.abspath(__file__)).parent.parent / "builtins/steps/gbstep"
            self.step_uri = f"file://{default_path}"
            logger.info(
                f"[MODEL VALIDATOR - BuildTargetStepConfig] STEP URI OMITTED IN BUILD.YAML, DEFAULTING TO {self.step_uri}"
            )
        return self

    # -------------------------------------------------------------------------
    # This utility validates the k8s env section inside the build.yaml config.
    # K8s treats unquoted env VALUE integers differently,
    # numeric must actually be strings, else it throws error.
    # This validator recursively walks the entire k8s config tree and raises errors if
    # k8s.env value is an integer where a string is expected.
    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_k8s_env_section(self):
        """Extract k8s config if present; if absent, nothing to validate
        Validate only the `k8s.env` section: ensure env values are STRINGS
        """
        k8s_cfg = self.config.get("k8s") if self.config else None
        if not k8s_cfg:
            return self

        errors = GBValidationErrors()

        # Nothing to validate if env not present
        if "env" not in k8s_cfg:
            return self

        env_cfg = k8s_cfg["env"]

        # --------------------------------------------------------------------
        # Case 1: dict env: (EXAMPLE)
        # env:
        #   NCCL_TIMEOUT:
        #     value: "10800000"
        # --------------------------------------------------------------------

        if isinstance(env_cfg, dict):
            for env_name, env_val in env_cfg.items():
                if isinstance(env_val, dict) and "value" in env_val:
                    v = env_val["value"]
                    if isinstance(v, int):
                        errors.add(
                            err=(
                                f"k8s env variable `env.{env_name}.value` must be a STRING, "
                                f"not `{type(v).__name__}`."
                            )
                        )

        elif isinstance(env_cfg, list):
            for i, env_val in enumerate(env_cfg):
                if isinstance(env_val, dict) and "value" in env_val:
                    v = env_val["value"]
                    if isinstance(v, int):
                        errors.add(
                            err=(
                                f"k8s env variable `env[{i}].value` must be a STRING, "
                                f"not `{type(v).__name__}`."
                            )
                        )

        if not errors.is_valid():
            raise ValueError(
                "Invalid k8s config found in build.yaml: Kubernetes env requires numeric values to be quoted.\n"
                + "\n".join(str(e) for e in errors)
            )

        return self


class BuildTargetConfig(Config):
    """A single target in a build file."""

    environment_uri: str
    inputs: Optional[Dict[str, BuildTargetInputConfig]] = Field(default_factory=dict)
    outputs: Optional[Dict[str, BuildTargetOutputConfig]] = Field(default_factory=dict)
    steps: List[BuildTargetStepConfig]
    dependency: Optional[set[str]] = Field(default_factory=set)


class BuildRetryConfig(Config):
    """Retry configuration for a build."""

    max_retries: int = 0
    target_reuse_enabled: bool = True


class BuildConfig(Config):
    """A single build config."""

    version: str = CURRENT_BUILD_YAML_VERSION
    name: str = ""
    targets: dict[str, BuildTargetConfig]
    retries: BuildRetryConfig = Field(default_factory=BuildRetryConfig)

    @classmethod
    def from_yaml(
        cls: Type[Self],
        path: Path,
        basekey: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs,
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
