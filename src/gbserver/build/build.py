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
The build.
"""

import asyncio
import tempfile
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Self

import yaml

from gbcommon.uri.space import SpaceURI
from gbcommon.uri.uri import URI
from gbserver.build.buildentity import BuildEntity
from gbserver.build.space import Space
from gbserver.build.target import Target
from gbserver.types.buildconfig import BUILD_FILENAME, BuildConfig, BuildFailure
from gbserver.types.stepconfig import StepConfig, StepInputsAcceptEnum
from gbserver.types.validation import GBValidationErrors, GBValidationErrorType
from gbserver.utils.filesystem import sync_or_copy
from gbserver.utils.logger import get_logger
from gbserver.utils.utils import get_uuid

logger = get_logger(__name__)

BUILD_DIR = "build"


class Build(BuildEntity):
    """Represents a single build."""

    # instance attributes
    build_id: str
    context: Optional[str] = None
    space: Optional[Space] = None
    event_q: asyncio.Queue
    targets: Dict[str, Target]
    allow_partial_builds: bool = False

    def __init__(
        self: Self,
        build_dir: Optional[Path] = None,
        build_id: Optional[str] = None,
        username: str = "",
        space: Optional[Space] = None,
        workspace_dir: Optional[Path] = None,
        event_q: Optional[asyncio.Queue] = None,
        targets: Optional[List[str]] = None,
        allow_partial_builds: bool = False,
        target_already_run_fn: Optional[
            Callable[[str], Optional[tuple[str, dict[str, list[str]]]]]
        ] = None,
        **kwargs,
    ) -> None:
        if build_id is None:
            build_id = get_uuid()
        if event_q is None:
            logger.info("build %s No event queue was provided, creating...", build_id)
            event_q = asyncio.Queue()
        if targets is None:
            targets = []
        if build_dir is None:
            self.context = None
        self.build_id = build_id
        self.space = space
        self.event_q = event_q
        self.targets = {}
        self.allow_partial_builds = allow_partial_builds
        self.target_already_run_fn: Optional[
            Callable[[str], Optional[tuple[str, dict[str, list[str]]]]]
        ] = target_already_run_fn
        logger.info(
            "build %s self.event_q: %s %s",
            self.build_id,
            id(self.event_q),
            self.event_q,
        )
        build_workspace_dir = (
            Path(tempfile.mkdtemp()) / self.build_id
            if workspace_dir is None
            else workspace_dir / self.build_id
        )
        new_build_dir = build_workspace_dir / BUILD_DIR
        logger.info("build %s final targets: %s", self.build_id, targets)
        try:
            if build_dir is None:
                logger.warning("build_dir was not specified")
            else:
                if not build_dir.is_dir():
                    raise ValueError(f"build_dir {build_dir} is not a valid directory")
                logger.info(
                    "loading build %s from a local directory %s",
                    self.build_id,
                    build_dir,
                )
                copied = sync_or_copy(str(build_dir) + "/", new_build_dir)
                if not copied:
                    raise BuildFailure(
                        f"failed to copy build directory from {build_dir} to {new_build_dir}"
                    )
            build_yaml_path = new_build_dir / BUILD_FILENAME
            logger.info("loading %s from path %s", BUILD_FILENAME, build_yaml_path)
            config = BuildConfig.from_yaml(
                build_yaml_path,
                context=self.context,
            )
            config = self.prune_build(config, targets)
            logger.info("Running with targets : %s", list(config.targets.keys()))
            super().__init__(
                build_id=self.build_id,
                event_q=event_q,
                build_workspace_dir=build_workspace_dir,
                username=username,
                type="build",
                config=config,
                dir=new_build_dir,
                **kwargs,
            )
            logger.info(
                "build %s after super.__init__ self.event_q: %s %s",
                self.build_id,
                id(self.event_q),
                self.event_q,
            )
        except Exception as e:
            logger.error("%s", traceback.format_exc())
            logger.error("error: %s", e)
            raise ValueError(f"Build {self.build_id} failed on creation") from e

    def assimilate(self: Self) -> None:
        """Processes a build"""
        self.targets = {}
        unresolvable_targets = []
        self_config = self.config
        assert isinstance(
            self_config, BuildConfig
        ), f"invalid build_config: {self_config}"
        for target_name, target_config in self_config.targets.items():
            try:
                self.targets[target_name] = Target(
                    build_id=self.build_id,
                    event_q=self.event_q,
                    target_name=target_name,
                    config=target_config,
                    build_workspace_dir=self.build_workspace_dir,
                    space=self.space,
                    username=self.username,
                    context=self.context,
                    force_fetch=self.force_fetch,
                )
            except Exception as e:
                if self.allow_partial_builds:
                    logger.error("%s", traceback.format_exc())
                    logger.error(
                        "failed to load the target %s for build %s : %s",
                        target_name,
                        self.build_id,
                        e,
                    )
                    unresolvable_targets.append(target_name)
                    continue
                raise ValueError(
                    f"failed to load the target {target_name} for build {self.build_id} :"
                ) from e
        if len(unresolvable_targets) > 0:
            logger.error("Unresolvable targets : %s", unresolvable_targets)

    def prune_build(
        self: Self, config: BuildConfig, targets: Optional[List[str]] = None
    ) -> BuildConfig:
        """
        Prunes the build config based on the target and its dependencies.
        """
        if targets is None or len(targets) == 0:
            logger.info("no targets specified, not pruning the build")
            return config
        updated_config = deepcopy(config)
        updated_config.targets = {}
        pruned_targets: set[str] = set()
        to_crawl_targets = set(targets)
        while True:
            if len(to_crawl_targets) == 0:
                break
            target = to_crawl_targets.pop()
            if target in pruned_targets:
                continue
            pruned_targets.add(target)
            dependencies = self.get_dependencies(config, target)
            to_crawl_targets.update(dependencies)
        for target in pruned_targets:
            if target not in config.targets:
                raise ValueError(f"Unknown Target in dependencies : {target}")
            updated_config.targets[target] = config.targets[target]
        return updated_config

    def get_dependencies(self: Self, config: BuildConfig, target_name: str) -> set[str]:
        """Get the targets that the given target depends on."""
        if target_name not in config.targets:
            return set()
        inputs = config.targets[target_name].inputs
        if inputs is None:
            return set()
        dependencies: set[str] = set()
        for t_input in inputs.values():
            if t_input.binding is not None:
                binding_target_name, _ = t_input.get_binding_parts()
                dependencies.add(binding_target_name)
        return dependencies

    def __validate_step_uris(self: Self) -> GBValidationErrors:
        logger.info("validating the step URIs of the build")
        errors = GBValidationErrors()
        build_config = self.config
        assert isinstance(
            build_config, BuildConfig
        ), f"invalid build_config: {build_config}"
        for target_name, target in build_config.targets.items():
            err_prefix = f"Target `{target_name}`:"
            logger.info("checking env of: %s %s", err_prefix, target)
            target_env_uri: Optional[URI] = None
            try:
                target_env_uri = URI.get_uri(target.environment_uri)
                if not target_env_uri.exists():
                    err = f"{err_prefix} the env URI {target.environment_uri} doesn't exist"
                    errors.add(err=err, type=GBValidationErrorType.NOT_EXIST)
                # elif not target_env_uri.is_accessible():
                #     err = f"{err_prefix} the env URI {target.environment_uri} is not accessible"
                #     errors.add(err=err, type=GBValidationErrorType.NOT_EXIST)
            except Exception as e:
                err = (
                    f"{err_prefix} the env URI {target.environment_uri} is invalid: {e}"
                )
                errors.add(err=err)
            # Read the env's `type` so SpaceURI's env-class-match tier picks the
            # right env-keyed step variant during validation.  The validator runs
            # before any TargetStep is instantiated, so without this scope the
            # thread-local has no env class set and `space://steps/<name>` URIs
            # whose only on-disk variants live under env-keyed subdirs would be
            # reported as unresolvable.
            env_class_name = self.__read_env_class_name(target_env_uri)
            logger.info("checking the steps of the target: %s %s", target_name, target)
            with SpaceURI.with_current_env_class_name(env_class_name):
                for i, step in enumerate(target.steps):
                    err_prefix = f"Target `{target_name}` Step `{i}`:"
                    try:
                        target_step_uri = URI.get_uri(step.step_uri)  # type: ignore[arg-type]
                        if not target_step_uri.exists():
                            err = f"{err_prefix} the step URI {step.step_uri} doesn't exist"
                            errors.add(err=err, type=GBValidationErrorType.NOT_EXIST)
                            continue
                        # if not target_step_uri.is_accessible():
                        #     err = f"{err_prefix} the step URI {step.step_uri} is not accessible"
                        #     errors.add(err=err, type=GBValidationErrorType.NOT_EXIST)
                        #     continue
                    except Exception as e:
                        err = f"{err_prefix} the step URI {step.step_uri} is invalid: {e}"
                        errors.add(err=err)
        return errors

    @staticmethod
    def __read_env_class_name(target_env_uri: Optional[URI]) -> Optional[str]:
        """Read the ``type`` field from the target's environment.yaml.

        Returns the env's class name (e.g. ``"K8s"``, ``"Docker"``,
        ``"Skypilot"``) for use as ``SpaceURI.current_env_class_name`` during
        step URI validation.  Returns ``None`` when the env URI is unavailable,
        not a local path, or its yaml can't be parsed — in which case the
        env-class-match tier is silently skipped and validation falls back to
        the existing tiers.

        Lightweight on purpose: skips the full ``Environment.get_environment``
        instantiation (which requires an event_q and runs side effects); the
        validator only needs the class-name string.
        """
        if target_env_uri is None or target_env_uri.uri is None:
            return None
        env_path_str = target_env_uri.uri.path
        if not env_path_str:
            return None
        env_dir = Path(env_path_str)
        env_yaml = env_dir / "environment.yaml"
        if not env_yaml.is_file():
            return None
        try:
            with open(env_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            return None
        if not isinstance(data, dict):
            return None
        type_val = data.get("type")
        return type_val if isinstance(type_val, str) and type_val else None

    def __validate_target_inputs(self: Self) -> GBValidationErrors:
        logger.info("validating the inputs of the build")
        errors = GBValidationErrors()
        build_config = self.config
        assert isinstance(
            build_config, BuildConfig
        ), f"invalid build_config: {build_config}"
        for target_name, target in build_config.targets.items():
            logger.info("checking inputs of the target: %s %s", target_name, target)
            if target.inputs is None:
                continue
            for target_input_name, target_input in target.inputs.items():
                err_prefix = f"Target `{target_name}` Input `{target_input_name}`:"
                logger.info("checking: %s %s", err_prefix, target_input)
                if target_input.uri is None:
                    continue
                logger.info("checking if input URI is valid: %s", target_input.uri)
                try:
                    target_input_uri = URI.get_uri(
                        target_input.uri,
                        secrets=self.space.get_secrets() if self.space else None,
                    )

                    if not target_input_uri.exists():
                        err = f"{err_prefix} the input URI {target_input.uri} doesn't exist"
                        errors.add(err=err, type=GBValidationErrorType.NOT_EXIST)
                        continue
                    # if not target_input_uri.is_accessible():
                    #     err = f"{err_prefix} the input URI {target_input.uri} is not accessible"
                    #     errors.add(err=err, type=GBValidationErrorType.NOT_EXIST)
                    #     continue
                except Exception as e:
                    err = (
                        f"{err_prefix} the input URI {target_input.uri} is invalid: {e}"
                    )
                    errors.add(err=err)
        return errors

    def __validate_step_inputs_and_outputs(self: Self) -> GBValidationErrors:
        logger.info("validating the inputs and outputs to each step")
        errors = GBValidationErrors()
        build_config = self.config
        assert isinstance(
            build_config, BuildConfig
        ), f"invalid build_config: {build_config}"
        for target_name, target in build_config.targets.items():
            logger.info(
                "checking inputs/outputs of the target: %s %s", target_name, target
            )
            err_prefix = f"Target `{target_name}`"
            target_inputs = target.inputs
            target_outputs = target.outputs
            if target_inputs is None and target_outputs is None:
                errors.add_warning(f"Target `{target_name}` has no inputs or outputs")
                continue
            if len(target.steps) == 0:
                errors.add_warning(f"Target `{target_name}` has no steps")
                continue
            curr_target = self.targets[target_name]
            for i, targetstep in enumerate(curr_target.targetsteps):
                step_err_prefix = f"{err_prefix} Step `{i}`"
                step_yaml = targetstep.step.config
                assert isinstance(
                    step_yaml, StepConfig
                ), f"invalid step_yaml: {step_yaml}"
                logger.info("validating against step.yaml inputs")
                for req_input, expected_input in step_yaml.inputs.required.items():
                    err_prefix1 = f"{step_err_prefix} Required input `{req_input}` of type `{expected_input.type}`"
                    if target_inputs is None or req_input not in target_inputs:
                        errors.add(f"{err_prefix1} is missing")
                        continue
                    errors.add(expected_input.validation, prefix=err_prefix1 + " ")
                    actual_input = target_inputs[req_input]
                    if actual_input.uri != "":
                        if StepInputsAcceptEnum.URI not in expected_input.accept:
                            errors.add(f"{err_prefix1} does not accept uri")
                    elif actual_input.binding != "":
                        if StepInputsAcceptEnum.BINDING not in expected_input.accept:
                            errors.add(f"{err_prefix1} does not accept binding")
                for opt_input, expected_input in step_yaml.inputs.optional.items():
                    err_prefix1 = f"{step_err_prefix} Optional input `{opt_input}` of type `{expected_input.type}`"
                    if target_inputs is None or opt_input not in target_inputs:
                        continue
                    errors.add(expected_input.validation, prefix=err_prefix1 + " ")
                    actual_input = target_inputs[opt_input]
                    if actual_input.uri != "":
                        if StepInputsAcceptEnum.URI not in expected_input.accept:
                            errors.add(f"{err_prefix1} does not accept uri")
                    elif actual_input.binding != "":
                        if StepInputsAcceptEnum.BINDING not in expected_input.accept:
                            errors.add(f"{err_prefix1} does not accept binding")
                if not step_yaml.inputs.allow_unknown:
                    if target_inputs is not None:
                        x1 = set(step_yaml.inputs.required.keys())
                        x2 = set(step_yaml.inputs.optional.keys())
                        x3 = x1.union(x2)
                        y1 = set(target_inputs.keys())
                        extras = y1 - x3
                        if len(extras) > 0:
                            errors.add(
                                f"{step_err_prefix} found extra inputs that are not allowed: {extras}"
                            )
                logger.info("validating against step.yaml outputs")
                target_outputs_keys: set[str] = (
                    set() if target_outputs is None else set(target_outputs.keys())
                )
                all_target_outputs = set(target_outputs_keys)
                for req_output, expected_output in step_yaml.outputs.required.items():
                    err_prefix1 = f"{step_err_prefix} Required output `{req_output}` of type `{expected_output.type}`"
                    if req_output not in target_outputs_keys:
                        errors.add(f"{err_prefix1} is missing")
                        continue
                    errors.add(expected_output.validation, prefix=err_prefix1 + " ")
                    all_target_outputs.remove(req_output)
                for opt_output, expected_output in step_yaml.outputs.optional.items():
                    err_prefix1 = f"{step_err_prefix} Optional output `{opt_output}` of type `{expected_output.type}`"
                    if opt_output not in target_outputs_keys:
                        continue
                    errors.add(expected_output.validation, prefix=err_prefix1 + " ")
                    all_target_outputs.discard(opt_output)
                if len(all_target_outputs) > 0:
                    outputs_str = ", ".join(f"`{x}`" for x in all_target_outputs)
                    errors.add_warning(
                        f"{err_prefix} The outputs {outputs_str} are not provided by any of the target's steps."
                        + " This could be because some steps do not have an I/O schema defined.",
                    )
                if not step_yaml.outputs.allow_unknown:
                    if target_outputs is not None:
                        x1 = set(step_yaml.outputs.required.keys())
                        x2 = set(step_yaml.outputs.optional.keys())
                        x3 = x1.union(x2)
                        y1 = set(target_outputs.keys())
                        extras = y1 - x3
                        if len(extras) > 0:
                            errors.add(
                                f"{step_err_prefix} found extra outputs that are not allowed: {extras}"
                            )
        return errors

    def validate(self: Self) -> GBValidationErrors:
        """Validate the build."""
        logger.info("validating the build")
        errors = GBValidationErrors()
        build_config = self.config
        if not isinstance(build_config, BuildConfig):
            errors.add(f"the build config is invalid: {type(build_config)}")
            return errors
        # In case build_config was changed after __init__
        errors.add(build_config.my_validate())
        errors.add(self.__validate_step_uris())
        errors.add(self.__validate_target_inputs())
        errors.add(self.__validate_step_inputs_and_outputs())
        for t in self.targets.values():
            errors.add(t.val_errors)  # propagate warnings from child entities
        logger.info(
            "validated the build and found %d errors %d warnings",
            len(errors),
            len(errors.warnings),
        )
        return errors
