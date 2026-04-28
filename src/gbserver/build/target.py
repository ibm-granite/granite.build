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
The Target.
"""

import asyncio
from asyncio import Queue, Task, TaskGroup
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Tuple

from gbcommon.uri.env import EnvURI
from gbcommon.uri.uri import URI
from gbserver.build.buildentity import BuildEntity
from gbserver.build.space import Space
from gbserver.build.targetstep import TargetStep
from gbserver.environment.environment import Environment
from gbserver.types.buildconfig import (
    BuildTargetConfig,
    BuildTargetOutputConfig,
    BuildTargetStepConfig,
)
from gbserver.types.validation import GBValidationErrors
from gbserver.utils.filesystem import merge_dicts
from gbserver.utils.logger import get_logger

TARGETS_KEY = "targets"

logger = get_logger(__name__)


@dataclass(eq=False, unsafe_hash=False)
class BindingInfo:
    """Info about a binding/input/output of a target."""

    binding_id: str
    target_name: Optional[str] = field(default=None)
    available: bool = field(default=False)
    wait_for_push: bool = field(default=False)
    output: Optional[BuildTargetOutputConfig] = field(default=None)
    input_binding_id: Optional[str] = field(default=None)
    uris: List[str] = field(default_factory=list)
    other_locations: List[str] = field(default_factory=list)

    def __eq__(self, other):
        if not isinstance(other, BindingInfo):
            return NotImplemented
        return (
            self.target_name == other.target_name
            and self.binding_id == other.binding_id
        )

    def __hash__(self):
        if self.target_name is not None:
            return hash(self.target_name + "." + self.binding_id)
        return hash(self.binding_id)


class Target(BuildEntity):
    """A single target in a build."""

    def __init__(
        self: Self,
        build_id: str,
        event_q: Queue,
        target_name: str,
        config: BuildTargetConfig,
        build_workspace_dir: Path,
        space: Optional[Space] = None,
        username: str = "",
        context: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.name = target_name
        self.setup_done = False
        self.setup_lock = asyncio.Lock()
        self.space = space
        self.secrets = {}
        if self.space is not None:
            self.secrets = self.space.get_secrets()
        self.context = context
        self.build_workspace_dir = build_workspace_dir
        self.target_workspace_dir = build_workspace_dir / TARGETS_KEY / self.name
        self.setup_ids: Dict[str, str] = {}  # setup_id : setup_type
        self.setup_config = {}
        super().__init__(
            build_id=build_id,
            event_q=event_q,
            build_workspace_dir=build_workspace_dir,
            username=username,
            type="target",
            config=config,
            dir=self.target_workspace_dir,
            **kwargs,
        )

    def assimilate(self: Self) -> None:
        self_config = self.config
        assert isinstance(
            self_config, BuildTargetConfig
        ), f"invalid self_config: {self_config}"
        self.environment = Environment.get_environment(
            self_config.environment_uri,
            self.event_q,
            context=self.context,
            secrets=self.secrets,
            force_fetch=self.force_fetch,
        )
        assert self.dir is not None, "Target.dir is None"
        self.targetsteps = [
            TargetStep(
                build_id=self.build_id,
                event_q=self.event_q,
                targetstep=targetstep,
                target_name=self.name,
                environment=self.environment,
                build_workspace_dir=self.build_workspace_dir,
                target_dir=self.dir,
                username=self.username,
                context=self.context,
                force_fetch=self.force_fetch,
                parent_target_config=self_config,
                target_step_index=idx,
            )
            for idx, targetstep in enumerate(self_config.steps)
        ]

        self.inputs_status: set[BindingInfo] = set()
        if self_config.inputs is not None:
            for binding_id, t_input in self_config.inputs.items():
                wait_for_push = (
                    False if t_input.wait_for_push is None else t_input.wait_for_push
                )
                if t_input.binding:
                    binding_t_name, binding_t_input = t_input.get_binding_parts()
                    binding_info = BindingInfo(
                        target_name=binding_t_name,
                        binding_id=binding_t_input,
                        input_binding_id=binding_id,
                        wait_for_push=wait_for_push,
                    )
                    if t_input.metadata:
                        binding_info.other_locations = t_input.metadata.other_locations
                        logger.warning(
                            "found some metadata but the input uses a binding"
                            + " for the target '%s' input: %s",
                            self.name,
                            t_input,
                        )
                    self.inputs_status.add(binding_info)
                elif t_input.uri:
                    binding_info = BindingInfo(
                        binding_id=binding_id,
                        available=True,
                        uris=[t_input.uri],
                        wait_for_push=wait_for_push,
                    )
                    if t_input.metadata:
                        binding_info.other_locations = t_input.metadata.other_locations
                        logger.info(
                            "found some metadata for the target '%s' input: %s",
                            self.name,
                            t_input,
                        )
                    self.inputs_status.add(binding_info)

        setups_to_run: set[str] = set()
        self.launchers_in_use = set()
        for targetstep in self.targetsteps:
            # Only consider targetsteps that have a launcher
            if targetstep.launcher and targetstep.launcher.type is not None:
                self.launchers_in_use.add(targetstep.launcher.type)
            if targetstep.launcher and targetstep.launcher.setups:
                setups_to_run = setups_to_run.union(targetstep.launcher.setups)

        self.setups_to_run = []
        for targetstep in self.targetsteps:
            if (
                targetstep.step_environment_config
                and targetstep.step_environment_config.setups
            ):
                for (
                    setup_name,
                    setup,
                ) in targetstep.step_environment_config.setups.items():
                    if setup_name in setups_to_run:
                        self.setups_to_run.append(setup)

    def validate(self: Self) -> GBValidationErrors:
        errors = GBValidationErrors()
        errors.add(
            [t.val_errors for t in self.targetsteps]
        )  # propagate warnings from child entities
        return errors

    def is_dry_run_compatible(self: Self) -> bool:
        """
        Returns True if all the steps of the Targets also support dry run.
        """
        for step in self.targetsteps:
            if not step.is_dry_run_compatible():
                return False
        return True

    def pull_assets(
        self: Self, artifact_configs: List[BindingInfo], tg: TaskGroup
    ) -> List[Task[Tuple[Dict, Optional[BuildTargetStepConfig]]]]:
        """Makes artifacts available and returns tasks for getting the bindings"""
        tasks: List[Task[Tuple[Dict, Optional[BuildTargetStepConfig]]]] = []
        if artifact_configs is None:
            return tasks
        for artifact_config in artifact_configs:
            if len(artifact_config.uris) == 0:
                continue
            uri_to_handle = artifact_config.uris[-1]
            logger.info(
                "artifact_config: %s uri_to_handle: %s", artifact_config, uri_to_handle
            )
            artifact_config_uri = URI.get_uri(uri_to_handle)
            task = self.environment.pullasset(task_group=tg, uri=artifact_config_uri)
            if (
                artifact_config.input_binding_id is None
                or artifact_config.input_binding_id == ""
            ):
                task.binding_id = artifact_config.binding_id
            else:
                task.binding_id = artifact_config.input_binding_id
            if len(artifact_config.other_locations) > 0:
                logger.info(
                    "the input artifact has some 'other_locations' configured: %s",
                    artifact_config,
                )
            should_convert = isinstance(artifact_config_uri, EnvURI)
            if should_convert:
                if len(artifact_config.other_locations) > 0:
                    prev_uri_to_handle = uri_to_handle
                    uri_to_handle = artifact_config.other_locations[-1]
                    logger.info(
                        "detected EnvURI, for lineage changing from '%s' to '%s'",
                        prev_uri_to_handle,
                        uri_to_handle,
                    )
                else:
                    logger.warning(
                        "detected EnvURI '%s' but there are no 'other_locations' configured for lineage",
                        uri_to_handle,
                    )
            task.uri = uri_to_handle
            task.target = self
            tasks.append(task)
        return tasks

    async def setup(self: Self, tg: TaskGroup, **kwargs) -> None:
        """Do some setup before launching the steps that are part of the target."""
        async with self.setup_lock:
            if self.setup_done:
                return
            setup_tasks = set()
            for launcher_in_use in self.launchers_in_use:
                if launcher_in_use in self.environment.setup_types:
                    setup_task = self.environment.setup(
                        launcher_in_use, tg, space_secrets=self.secrets, **kwargs
                    )
                    self.setup_ids[setup_task.setup_id] = launcher_in_use
                    setup_tasks.add(setup_task)
            for setup in self.setups_to_run:
                setup_task = self.environment.setup(
                    setup.type, tg, **setup.config, **kwargs
                )
                self.setup_ids[setup_task.setup_id] = setup.type
                setup_tasks.add(setup_task)
            results = await asyncio.gather(*setup_tasks)
            combined_setup_config = {}
            for result in results:
                combined_setup_config = merge_dicts(combined_setup_config, result)
            self.setup_config = combined_setup_config
            self.setup_done = True

    async def teardown(self: Self) -> None:
        """Do some teardown after all the steps of the target is done."""
        teardown_tasks: set[Task] = set()
        for setup_id, setup_type in self.setup_ids.items():
            if setup_type in self.environment.teardown_types:
                teardown_tasks.add(
                    self.environment.teardown(setup_type, setup_id=setup_id)
                )
        if len(teardown_tasks) > 0:
            await asyncio.wait(teardown_tasks)
