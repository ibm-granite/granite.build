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
The step inside a Target.
"""

import os
import tempfile
from asyncio import Queue
from copy import deepcopy
from enum import Enum
from glob import glob
from pathlib import Path
from typing import Any, Dict, Optional, Self

import yaml

from gbcommon.uri.space import SpaceURI
from gbcommon.uri.uri import URI
from gbserver.build.buildentity import BuildEntity
from gbserver.build.step import STEP_FILE_NAME, Step
from gbserver.environment.environment import Environment
from gbserver.types.buildconfig import BuildTargetConfig, BuildTargetStepConfig
from gbserver.types.constants import (
    GB_ENVIRONMENT,
    GBSERVER_SIDECAR_MONITORING_IMAGE_TAG,
)
from gbserver.types.stepconfig import (
    StepConfig,
    StepEnvironmentTypeConfig,
    StepLauncherConfig,
    StepMonitorConfig,
)
from gbserver.types.validation import GBValidationErrors, GBValidationWarningType
from gbserver.utils.filesystem import (
    fill_templates_in_dir,
    merge_config_dirs,
    merge_dicts,
    sync_or_copy,
)
from gbserver.utils.logger import get_logger
from gbserver.utils.step_image import get_step_image
from gbserver.utils.template import fill_template
from gbserver.utils.utils import random_string
from gbserver.validators.validator import GBValidator

CONFIG_KEY = "config"
LAUNCHER_CONFIG = "launcher_config"
ENVIRONMENT_CONFIG = "environment_config"
ENVIRONMENT_CONFIGS = "environment_configs"
MONITOR_CONFIG = "monitor_config"
LAUNCHER_KEY = "launchers"
STEP_CONFIG_KEY_GB = "gb"
STEP_CONFIG_KEY_USE_BASESTEP = "use_basestep"
MONITORING_IMAGE_TAG = "monitoring_image_tag"
GB_ENV = "gb_environment"
STEP_IMAGE = "step_image"

logger = get_logger(__name__)


def _convert_enums_to_values(obj: Any) -> Any:
    """Recursively convert enum values to their string representation for YAML serialization."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _convert_enums_to_values(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_convert_enums_to_values(item) for item in obj)
    return obj


class TargetStep(BuildEntity):
    """A single step in a target of a build."""

    step: Step
    step_environment_config: StepEnvironmentTypeConfig
    launcher: StepLauncherConfig
    monitors: Dict[
        str, StepMonitorConfig
    ]  # to keep a map of launcher's monitor name to the StepMonitorConfig class
    validator_map: Dict[str, GBValidator]
    parent_target_config: Optional[BuildTargetConfig] = None
    target_step_index: int = -1

    def __init__(
        self: Self,
        build_id: str,
        event_q: Queue,
        targetstep: BuildTargetStepConfig,
        target_name: str,
        environment: Environment,
        build_workspace_dir: Path,
        target_dir: Path,
        username: Optional[str] = "",
        context: Optional[str] = None,
        force_fetch: bool = False,
        parent_target_config: Optional[BuildTargetConfig] = None,
        target_step_index: int = -1,
        **kwargs,
    ) -> None:
        """Checks if a step is valid and loads it"""
        self.validator_map = {}
        self.monitors = {}
        self.parent_target_config = parent_target_config
        # Scope the active env's step-discovery context on SpaceURI so that
        # any `space://steps/...` URIs resolved during step assimilation try,
        # in order: (1) the env's own directory (env-co-located steps),
        # (2) env-class-match against `environment_configs` keys, (3) the
        # env-agnostic fallback location.
        with SpaceURI.with_current_env(environment):
            self.step = Step(
                stepuri=targetstep.step_uri, context=context, force_fetch=force_fetch  # type: ignore[arg-type]
            )
            self.step_uri = targetstep.step_uri
            self.target_name = target_name
            self.target_step_index = target_step_index
            self.environment = environment
            targetstep_dir = target_dir / os.path.basename(self.step.dir) / random_string()  # type: ignore[arg-type]
            self.context = context
            sync_or_copy(str(self.step.dir) + "/", targetstep_dir)
            merge_config_dirs(self.step.dir, targetstep.config_dir, targetstep_dir)  # type: ignore[arg-type]
            self.step = Step(str(targetstep_dir))
        super().__init__(
            build_id=build_id,
            event_q=event_q,
            build_workspace_dir=build_workspace_dir,
            username=username,  # type: ignore[arg-type]
            type="targetstep",
            config=targetstep,
            dir=targetstep_dir,
            force_fetch=force_fetch,
            **kwargs,
        )

    def __get_validator_context(self: Self) -> dict:
        step_dir = self.step.step_yaml_path.parent
        context = {
            "dir": step_dir,
        }
        return context

    def is_dry_run_compatible(self: Self) -> bool:
        """
        Returns True if the step supports dry run.
        """
        step_config = self.step.config
        assert isinstance(step_config, StepConfig)
        return step_config.is_dry_run_compatible

    def init_validators(self):
        step_config = self.step.config
        self.validator_map = {}
        context = self.__get_validator_context()

        if len(step_config.validators) > 0:
            logger.info(
                "initializing %d env agnostic validators", len(step_config.validators)
            )
            for validator_name, validator_config in step_config.validators.items():
                try:
                    if validator_name in self.validator_map:
                        logger.warning(
                            "the validator name %s already exists, replacing...",
                            validator_name,
                        )
                    validator = GBValidator.get_validator(
                        validator_config=validator_config, context=context
                    )
                    self.validator_map[validator_name] = validator
                    logger.info(
                        "initialized validator %s %s", validator_name, validator
                    )
                except Exception as e:
                    raise ValueError(
                        f"failed to load the validator {validator_name} {validator_config} :"
                    ) from e
        # env specific validators
        # Only run if step_environment_config has been resolved (i.e., after merged config)
        if (
            self.step_environment_config is not None
            and len(getattr(self.step_environment_config, "validators", {})) > 0
        ):
            logger.info(
                "initializing %d env specific validators",
                len(self.step_environment_config.validators),
            )
            for (
                validator_name,
                validator_config,
            ) in self.step_environment_config.validators.items():
                try:
                    if validator_name in self.validator_map:
                        logger.warning(
                            "the validator name %s already exists, replacing...",
                            validator_name,
                        )
                    validator = GBValidator.get_validator(
                        validator_config=validator_config, context=context
                    )
                    self.validator_map[validator_name] = validator
                except Exception as e:
                    raise ValueError(
                        f"failed to load the validator {validator_name} {validator_config} :"
                    ) from e

    def load_step_default_config(self, step_default_file_path: Path, full_config: dict):
        """
        Load, render, and return a StepConfig object from a step_default.yaml file.
        This is used when a step.yaml is missing.
        """
        if not step_default_file_path.exists():
            raise FileNotFoundError(
                f"step_default.yaml not found at {step_default_file_path}"
            )

        # --- Render template ---
        raw_text = step_default_file_path.read_text()
        rendered_text = fill_template(raw_text, full_config, strict=False)
        step_cfg_base = yaml.safe_load(rendered_text) or {}

        self.step_default_file_path = step_default_file_path

        return step_cfg_base

    def _resolve_environment_and_launcher(self, step_config):
        """Resolve environment config and launcher based on current environment type."""
        if not getattr(self.environment, "type", None):
            logger.warning(
                "No environment type found — skipping environment resolution."
            )
            self.step_environment_config = None
            self.launcher = None
            self.launcher_name = ""
            return

        env_type = self.environment.type
        env_configs = step_config.environment_configs or {}

        self.step_environment_config = env_configs.get(env_type) or env_configs.get(
            env_type.lower()
        )
        if self.step_environment_config is None:
            logger.warning(
                f"The environment '{env_type}' was not found in step config; "
                f"falling back to default environment config (if any)."
            )
            if "default" in env_configs:
                self.step_environment_config = env_configs["default"]
            elif len(env_configs) > 0:
                first_env = next(iter(env_configs.keys()))
                logger.warning(
                    f"Using first available environment config: '{first_env}'"
                )
                self.step_environment_config = env_configs[first_env]
            else:
                raise ValueError(
                    f"No usable environment configs found for step {self.step_uri}"
                )

        if len(self.step_environment_config.launchers) == 0:
            raise ValueError(
                f"The environment '{env_type}' has no launchers: {self.step_environment_config}"
            )

        self_config = self.config
        assert isinstance(self_config, BuildTargetStepConfig)

        launcher_name = self_config.launcher
        if not launcher_name:
            launcher_name = self.step_environment_config.default_launcher
        if not launcher_name:
            launcher_names = sorted(self.step_environment_config.launchers.keys())
            launcher_name = launcher_names[0]
        if launcher_name not in self.step_environment_config.launchers:
            raise ValueError(
                f"Failed to find the launcher '{launcher_name}' in "
                f"{self.step_environment_config.launchers.keys()}"
            )

        self.launcher_name = launcher_name
        self.launcher = self.step_environment_config.launchers[launcher_name]

    def assimilate(self):
        # Merge step configurations
        self._merge_step_configs()
        step_config = self.step.config

        self._resolve_environment_and_launcher(step_config)

        # Initialize validators after final config is ready
        self.init_validators()

        logger.info(
            "for the target %s step %s using the launcher %s : %s",
            self.target_name,
            self.step_uri,
            self.launcher_name,
            self.launcher,
        )

    def merge_handle_configs(
        self,
        step_default_file_path,
        full_config,
        targetstep_config_unfilled,
        step_config_from_step=None,
        step_file_path=None,
        use_basestep=True,
        step_file_exists=False,
    ):
        """
        Merge configs in order:
            base_step (step_default.yaml) <- step.yaml <- build.yaml
        Returns merged and unfilled step and build config dicts, merged environment configs, and merged step dir and other metadata.
        """

        step_cfg_base = self.load_step_default_config(
            step_default_file_path, full_config
        )

        base_step_config = dict(step_cfg_base)
        base_step_config["config"] = step_cfg_base.get("config", {})

        # --- Merge configs ---
        merged_inner_config = dict(base_step_config.get(CONFIG_KEY, {}))

        # If step config exists, merge base step config with step config
        # (for same keys, priority given to step config)
        if step_config_from_step:
            merged_inner_config = merge_dicts(
                merged_inner_config, step_config_from_step
            )

        # If build config exists, merge base step config + step config with build config
        # (for same keys, priority given to build config)
        if targetstep_config_unfilled:
            merged_inner_config = merge_dicts(
                merged_inner_config, targetstep_config_unfilled
            )

        full_config[CONFIG_KEY] = merged_inner_config
        self._merged_step_build_config_unfilled = merged_inner_config

        # --- Merge step-default + step.yaml (non-config keys) ---
        merged_step_config = dict(base_step_config)
        if step_file_exists and step_file_path:
            with open(step_file_path, "r") as f:
                step_file_data = yaml.safe_load(f) or {}
            for key, val in step_file_data.items():
                if key == "config":
                    continue
                if isinstance(val, dict):
                    merged_step_config[key] = merge_dicts(
                        merged_step_config.get(key, {}), val
                    )
                else:
                    merged_step_config[key] = val

        merged_step_config["config"] = merged_inner_config

        # --- Write merged step.yaml into temp_path ---
        temp_path = Path(tempfile.mkdtemp())
        final_step_yaml_path = temp_path / STEP_FILE_NAME
        with open(final_step_yaml_path, "w") as f:
            yaml.safe_dump(
                _convert_enums_to_values(merged_step_config),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

        # --- Merge environment configs ---
        default_step_env_configs = merged_step_config.get(ENVIRONMENT_CONFIGS, {})

        # Preserve env.yaml configs, but replace step-default part with each new run
        base_env_configs = full_config.get(ENVIRONMENT_CONFIG, {})
        merged_env_configs = dict(
            base_env_configs
        )  # config coming from environment.yaml
        merged_env_configs.update(
            default_step_env_configs
        )  # overlay environment configs from step
        full_config[ENVIRONMENT_CONFIG] = merged_env_configs
        merged_env_configs = full_config[ENVIRONMENT_CONFIG]

        # --- Determine environment type ---
        env_type = None
        if "environment" in full_config and getattr(
            full_config["environment"], "type", None
        ):
            env_type = full_config["environment"].type
        elif merged_env_configs:
            env_type = next(iter(merged_env_configs.keys()))
        if not env_type:
            raise ValueError(
                "No environment type could be resolved (missing 'type' and empty environment_configs)."
            )

        env_config = merged_env_configs.get(env_type) or merged_env_configs.get(
            env_type.lower()
        )
        if not env_config:
            raise ValueError(
                f"Environment config for '{env_type}' not found in environment_configs"
            )

        # --- Prepare step directories ---
        if use_basestep:
            base_step_src = Path(__file__).parent.parent / "builtins/steps/gbstep"
            sync_or_copy(str(base_step_src) + "/", temp_path, delete=False)
            fill_templates_in_dir(
                temp_path,
                full_config,
                ignore_paths=[temp_path / "step_default.yaml"],
                strict=False,
                fill_paths=True,
                fill_files=False,
            )

        if step_file_exists and step_file_path:
            # copies everything present at the step.yaml's parent path to temp_path
            sync_or_copy(str(step_file_path.parent) + "/", temp_path, delete=False)
        else:
            # first takes the subdir (which contains step's repo contents) inside current self.dir
            # then copy contents of subdir into temp_path
            subdir = next(self.dir.iterdir())
            sync_or_copy(str(subdir) + "/", temp_path, delete=False)

        step_yaml_path = temp_path / STEP_FILE_NAME
        ignore_paths = []

        # if fallback is not used -> maintain the default behaviour -> ignore filling step.yaml
        # if fallback is used -> we should NOT ignore filling step.yaml
        # because it is actually default step that is used
        logger.info(
            f"\n==== Step fallback to default step used: {self.step.step_fallback_used}====\n"
        )
        if not (self.step.step_fallback_used):
            ignore_paths = [step_yaml_path] if step_yaml_path.exists() else []

        return {
            "base_step_config": base_step_config,
            "merged_inner_config": merged_inner_config,
            "merged_env_configs": merged_env_configs,
            "env_type": env_type,
            "env_config": env_config,
            "merged_step_dir": temp_path,
            "ignore_paths_final_fill": ignore_paths,
        }

    def _merge_step_configs(self):

        targetstep_step_config = self.step.config

        assert isinstance(targetstep_step_config, StepConfig)
        self.full_config = {
            "step": {"name": targetstep_step_config.name},
            ENVIRONMENT_CONFIG: self.environment.config.config,
            MONITORING_IMAGE_TAG: GBSERVER_SIDECAR_MONITORING_IMAGE_TAG,
            GB_ENV: GB_ENVIRONMENT,
            STEP_IMAGE: get_step_image(),
        }
        targetstep_config = self.config
        assert isinstance(targetstep_config, BuildTargetStepConfig)
        try:

            space_variables = URI.get_space_config()
            if space_variables is not None:
                self.full_config |= space_variables

            # Load build config without filling it yet because of runtime fields - should only merge in targetstep
            targetstep_config_unfilled = deepcopy(targetstep_config.config or {})

            # 3. Load configs from step.yaml and base step
            step_file_paths = glob(
                str(self.dir / "**" / STEP_FILE_NAME), recursive=True
            )
            step_file_exists = len(step_file_paths) > 0
            step_config_from_step = {}
            if step_file_exists:
                step_file_path = Path(step_file_paths[0])
                step_cfg = StepConfig.from_yaml(step_file_path)
                # DO NOT FILL step config at this point — it may refer runtime-only values like `bindings`
                step_config_from_step = step_cfg.config or {}

            # Decide if base step should be used
            use_basestep = True
            if step_file_exists:
                step_cfg_gb = step_config_from_step.get(STEP_CONFIG_KEY_GB, {})
                if (
                    STEP_CONFIG_KEY_USE_BASESTEP in step_cfg_gb
                    and not step_cfg_gb[STEP_CONFIG_KEY_USE_BASESTEP]
                ):
                    use_basestep = False

            if use_basestep:
                step_default_file_path = (
                    Path(os.path.abspath(__file__)).parent.parent
                    / "builtins/steps/gbstep/step_default.yaml"
                )
                self.step_default_file_path = step_default_file_path
                if step_default_file_path.exists():
                    # if step.yaml exists in the step dir
                    if targetstep_step_config is not None:
                        self.is_step_file_exists = True
                        # handle merging the step configs here itself else we do it in targetsteprun
                        # after resolving runtime values (step-default contains runtime placeholders)
                        result = self.merge_handle_configs(
                            step_default_file_path,
                            self.full_config,
                            targetstep_config_unfilled,
                            step_config_from_step=step_config_from_step,
                            step_file_path=step_file_path if step_file_exists else None,
                            use_basestep=use_basestep,
                            step_file_exists=step_file_exists,
                        )
                        self.full_config[CONFIG_KEY] = result["merged_inner_config"]
                        self.step_environment_config = result["env_config"]
                        self.env_type = result["env_type"]
                        self.merged_step_dir = result["merged_step_dir"]
                        self.ignore_paths_final_fill = result["ignore_paths_final_fill"]
                        self._merged_step_build_config_unfilled = result[
                            "merged_inner_config"
                        ]

            logger.warning(
                "Step URI uri: %s contains step.yaml  =  %s",
                self.step_uri,
                self.is_step_file_exists,
            )

        except Exception as e:
            logger.exception("Failed during TargetStep._merge_step_configs()")
            raise

    def _get_validation_context(self: Self) -> Dict:
        """Get context for use with validators."""
        context = {
            "target_name": self.target_name,
            "target_step_index": self.target_step_index,
        }
        # ------------------
        env_asset = self.environment.environment_asset
        if env_asset is None:
            logger.warning("the environment_asset is None")
        else:
            context["environment_uri"] = env_asset.uristr
        # ------------------
        if self.parent_target_config is not None:
            bt_inputs = self.parent_target_config.inputs
            if bt_inputs is not None:
                step_inputs = {}
                for k, v in bt_inputs.items():
                    step_inputs[k] = v.model_dump()
                context["step_inputs"] = step_inputs
            bt_outputs = self.parent_target_config.outputs
            if bt_outputs is not None:
                step_outputs = {}
                for k, v in bt_outputs.items():  # type: ignore[assignment]
                    step_outputs[k] = v.model_dump()
                context["step_outputs"] = step_outputs
        # ------------------
        return context

    def validate(self: Self) -> GBValidationErrors:
        errors = GBValidationErrors()
        self_config = self.config
        assert isinstance(self_config, BuildTargetStepConfig)
        build_yaml_step_config = self_config.config
        if build_yaml_step_config is None:
            logger.warning("there is no step config to validate")
            return errors
        logger.info("validating the step config: %s", build_yaml_step_config)
        # ------------------
        if "gb" in build_yaml_step_config:
            errors.add_warning(
                "overiding internal section 'gb'",
                type=GBValidationWarningType.DEPRECATED,
            )
        # ------------------
        context = self._get_validation_context()
        for validator_name, validator in self.validator_map.items():
            if not validator.is_static():
                logger.info("skipping non-static validator %s", validator_name)
                continue
            logger.info("running validator %s", validator_name)
            curr_errors = validator.validate(
                data=build_yaml_step_config, context=context
            )
            errors.add(curr_errors)
        errors.add(self.step.val_errors)  # propagate warnings from child entities
        return errors
