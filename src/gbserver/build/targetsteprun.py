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
The target step run.
"""

import asyncio
import dataclasses
import traceback
from asyncio import Queue, TaskGroup
from copy import deepcopy
from typing import Any, Dict, Optional, Self

from gbserver.build.run import Run
from gbserver.build.target import Target
from gbserver.build.targetstep import TargetStep
from gbserver.types.buildconfig import BuildTargetStepConfig
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventStatusPayload,
    BuildEventType,
    EntityRunMetadata,
)
from gbserver.types.constants import USE_LESS_COMPUTE_ON_DRY_RUN
from gbserver.types.status import STATUS_TO_ICON, Status
from gbserver.types.stepconfig import StepLauncherConfig
from gbserver.utils.filesystem import (
    fill_templates_in_dir,
    sync_or_copy,
)
from gbserver.utils.logger import get_logger
from gbserver.utils.template import fill_objtemplate

logger = get_logger(__name__)

TARGETRUNS_KEY = "targetruns"
BINDINGS_KEY = "bindings"
RUN_METADATA_KEY = "run_metadata"
SETUP_CONFIG = "setup_config"
CONFIG_KEY = "config"
LAUNCHER_CONFIG = "launcher_config"
LAUNCHER_KEY = "launchers"
MONITOR_CONFIG = "monitor_config"
ENVIRONMENT_CONFIG = "environment_config"


class TargetStepRun(Run):
    """Run of a single build target step."""

    targetrun_id: str
    launch_id: str

    def __init__(
        self: Self,
        target: Target,
        targetstep: TargetStep,
        targetrun_id: str,
        event_q: Queue,
        additional_targetsteps_queue: Optional[Queue] = None,
        bindings: Optional[Dict] = None,
        setup_config: dict = None,  # type: ignore[assignment]
        dry_run: bool = False,
    ) -> None:
        try:
            self.target = target
            self.targetrun_id = targetrun_id
            self.launch_id = ""
            super().__init__(
                entity=targetstep,
                event_q=event_q,
                base_dir=targetstep.build_workspace_dir / TARGETRUNS_KEY,
                dry_run=dry_run,
            )
            self.bindings = bindings

            # Full config (without RUN_METADATA and BINDINGS) and MERGED unfilled step-default + step + build config from targetstep
            self.full_config = deepcopy(targetstep.full_config)

            # if step.yaml exists -> merging is already done in targetstep -> do a deepcopy only -> no need to merge again
            merged_step_build_config_unfilled = {}
            if targetstep.is_step_file_exists:
                # unfilled step config + build config
                merged_step_build_config_unfilled = deepcopy(
                    targetstep._merged_step_build_config_unfilled
                )

            # run-specific data
            targetsteprun_runtime_config = {
                BINDINGS_KEY: self.bindings,
                RUN_METADATA_KEY: dataclasses.asdict(self.get_runmetadata()),
                SETUP_CONFIG: setup_config,
            }

            # add the runtime config into the full config
            self.full_config.update(targetsteprun_runtime_config)

            logger.info(
                "FULL CONFIG RUN_METADATA_KEY: %s", self.full_config[RUN_METADATA_KEY]
            )

            logger.info("FULL CONFIG STEP NAME: %s", self.full_config["step"])

            # At this stage, full config CONTAINS RUN_METADATA and BINDINGS + MERGED unfilled step-default + step + build config from targetstep

            # ====== Final filling of config templates after runtime data into the full config ======
            filled_inner = fill_objtemplate(
                merged_step_build_config_unfilled,
                self.full_config,
                strict=False,
                skip_keys={"field_value_template"},
            )
            if isinstance(filled_inner, dict):
                if self.dry_run:
                    logger.info("dry_run: setting mock to True")
                    filled_inner_gb = filled_inner.get("gb", {})
                    filled_inner_gb["mock"] = True
                    filled_inner["gb"] = filled_inner_gb
                    if USE_LESS_COMPUTE_ON_DRY_RUN:
                        if "compute_config" in filled_inner:
                            filled_inner_compute_config = filled_inner["compute_config"]
                            if "num_nodes" in filled_inner_compute_config:
                                if filled_inner_compute_config["num_nodes"] > 2:
                                    filled_inner_compute_config["num_nodes"] = 2
                            if "num_gpus_per_node" in filled_inner_compute_config:
                                if filled_inner_compute_config["num_gpus_per_node"] > 2:
                                    filled_inner_compute_config["num_gpus_per_node"] = 2
                            if "num_cpus_per_node" in filled_inner_compute_config:
                                if (
                                    filled_inner_compute_config["num_cpus_per_node"]
                                    > 16
                                ):
                                    filled_inner_compute_config["num_cpus_per_node"] = (
                                        16
                                    )
                            filled_inner["compute_config"] = filled_inner_compute_config
            else:
                logger.warning(
                    "filled_inner is not a dict: %s %s",
                    type(filled_inner),
                    filled_inner,
                )

            self.full_config[CONFIG_KEY] = filled_inner

            env_config = (
                targetstep.step_environment_config
            )  # returns object of type StepEnvironmentConfig

            logger.info("STEP ENVIRONMENT CONFIG: %s", env_config)
            logger.info(
                "FULL CONFIG ENVIRONMENT CONFIG : %s",
                self.full_config[ENVIRONMENT_CONFIG],
            )
            env_type = targetstep.env_type

            # Populate launcher and monitor config here - different launcher for different runs

            #  --- Launcher config ---
            launchers = env_config.launchers or {}
            if not launchers:
                raise ValueError(f"No launchers found in environment '{env_type}'")

            # Prefer the launcher resolved from the build YAML by targetstep.
            launcher_name = targetstep.launcher_name
            if not launcher_name:
                launcher_name = getattr(env_config, "default_launcher", None)
            if not launcher_name:
                launcher_names = sorted(launchers.keys())
                if not launcher_names:
                    raise ValueError(
                        f"No launchers available in environment '{env_type}'"
                    )
                launcher_name = launcher_names[0]

            if launcher_name not in launchers:
                raise ValueError(
                    f"Failed to find the launcher '{launcher_name}' in {list(launchers.keys())}"
                )

            launcher_cfg = deepcopy(
                launchers[launcher_name]
            )  # returns a StepLauncherConfig object
            filled_launcher_config = fill_objtemplate(
                launcher_cfg, self.full_config, strict=True
            )
            targetstep.launcher = StepLauncherConfig(**filled_launcher_config)

            self.full_config[LAUNCHER_CONFIG] = filled_launcher_config.get(
                CONFIG_KEY, {}
            )
            logger.info(
                "FULL CONFIG LAUNCHER CONFIG: %s", self.full_config[LAUNCHER_CONFIG]
            )

            # --- Monitor Config ---
            self.monitors = {}
            if launcher_cfg.monitors:
                for name in launcher_cfg.monitors:

                    monitors = getattr(env_config, "monitors", None) or {}
                    if name not in monitors:
                        raise ValueError(
                            f"Launcher '{launcher_name}' requires monitor '{name}', "
                            "but it is not defined in environment config."
                        )
                    monitor_obj = monitors[name]
                    self.monitors[name] = monitor_obj

            filled_monitor_configs = {}
            for name, monitor in self.monitors.items():
                monitor_config = deepcopy(monitor.config or {})
                filled_monitor_config = fill_objtemplate(
                    monitor_config,
                    self.full_config,
                    strict=True,
                    skip_keys={"field_value_template"},
                )
                filled_monitor_configs[name] = {
                    "type": monitor.type,
                    "config": filled_monitor_config,
                }

            new_monitor_configs = {}  # type: ignore[var-annotated]
            for monitor_name, monitor_info in filled_monitor_configs.items():
                monitor_type = monitor_info["type"] or "unknown"
                new_monitor_configs.setdefault(monitor_type, []).append(
                    {
                        "name": monitor_name,
                        "config": monitor_info["config"],
                    }
                )

            self.full_config[MONITOR_CONFIG] = new_monitor_configs
            logger.info(
                "FULL CONFIG MONITOR CONFIG: %s", self.full_config[MONITOR_CONFIG]
            )
            logger.info("FULL CONFIG's CONFIG: %s", self.full_config[CONFIG_KEY])

            temp_path = targetstep.merged_step_dir  # merged step path from targetstep

            # Populate merged directory path to pass to the launch
            # in order to copy this final step folder to pod
            self.full_config["merged_dir_path"] = temp_path
            ignore_paths = targetstep.ignore_paths_final_fill
            self.temp_path = temp_path

            logger.info("Ignoring %d paths during template fill", len(ignore_paths))
            targetstep.ignore_paths_final_fill = ignore_paths
            self.ignore_paths = ignore_paths

            fill_templates_in_dir(
                temp_path, self.full_config, ignore_paths=ignore_paths, strict=True
            )

            step_default_yaml = temp_path / "step_default.yaml"
            if step_default_yaml.exists():
                step_default_yaml.unlink()
                logger.info("Removed step_default.yaml before final sync")

            sync_or_copy(str(temp_path) + "/", self.dir, delete=False)
            logger.info("==== Final Step Folder ======: %s", str(temp_path))

        except Exception as e:
            current_err = f"Build `{self.build_id}` Target Step `{self.targetrun_id}` failed on creation."
            full_err_stack = traceback.format_exc()
            status = Status.FAILED
            icon = STATUS_TO_ICON[status]
            msg = f"{icon}  {current_err} error:\n```\n{full_err_stack}\n```\n"
            # logger.error("%s", msg) # TODO: is this necessary?
            run_metadata = self.get_runmetadata()
            payload = BuildEventStatusPayload(status=status, msg=msg)
            fail_event = BuildEvent(
                run_metadata=run_metadata,
                type=BuildEventType.STATUS_EVENT,
                payload=payload,
            )
            self.dispatch_event(fail_event)
            raise ValueError(current_err) from e

    async def _run(self: Self, tg: Optional[TaskGroup] = None, **kwargs) -> Any:
        self_entity = self.entity
        assert isinstance(self_entity, TargetStep)
        logger.info("self.full_config: %s", self.full_config)

        build_config = self_entity.config
        if (
            isinstance(build_config, BuildTargetStepConfig)
            and build_config.retry_enabled is not None
        ):
            self.full_config["retry_enabled"] = build_config.retry_enabled
        if (
            isinstance(build_config, BuildTargetStepConfig)
            and build_config.retry_transparently is not None
        ):
            self.full_config["retry_transparently"] = build_config.retry_transparently

        async with TaskGroup() as tg:
            self.launch_task = self_entity.environment.launch(
                launcher_type=self_entity.launcher.type,
                task_group=tg,
                targetsteprun_asset_dir=self.dir,
                setup_ids=list(self.target.setup_ids.keys()),
                **self.full_config,
            )
            launch_id = self.launch_task.launch_id  # type: ignore[attr-defined]
            assert (
                isinstance(launch_id, str) and launch_id != ""
            ), f"invalid launch_id: {launch_id}"
            self.launch_id = launch_id
            monitor_tasks = set()
            if self_entity.launcher.monitors is not None:
                for monitor in self_entity.launcher.monitors:
                    assert (
                        self_entity.step_environment_config.monitors is not None
                    ), "environment config monitors is None"
                    monitor_config = self_entity.step_environment_config.monitors[
                        monitor
                    ]
                    if monitor_config.config is None:
                        monitor_config.config = {}
                    monitor_tasks.add(
                        self_entity.environment.monitor(
                            type=monitor_config.type,
                            launch_id=self.launch_id,
                            task_group=tg,
                            event_q=self.event_q,
                            entityrun_metadata=self.get_runmetadata(),
                            build_id=self.build_id,
                            **monitor_config.config,
                        )
                    )
            await asyncio.gather(*monitor_tasks)
            await self.launch_task

    async def _cleanup(self: Self, tg: Optional[TaskGroup] = None, **kwargs) -> None:
        """Check status and cleanup after the job is done."""
        logger.debug("Run._cleanup %s start", self.id)
        self_entity = self.entity
        assert isinstance(self_entity, TargetStep)
        assert self.launch_id, f"invalid self.launch_id {self.launch_id}"
        async with TaskGroup() as tg:
            cleanup_task = self_entity.environment.cleanup(
                launch_type=self_entity.launcher.type,
                launch_id=self.launch_id,
                setup_ids=list(self.target.setup_ids.keys()),
                tg=tg,
            )
            if cleanup_task is not None:
                await cleanup_task
        logger.debug("Run._cleanup %s end", self.id)

    def get_runmetadata(self: Self) -> EntityRunMetadata:
        self_entity = self.entity
        assert isinstance(self_entity, TargetStep)
        return EntityRunMetadata(
            build_id=self.build_id,
            username=self_entity.username,
            type=type(self_entity).__name__,
            target_name=self_entity.target_name,
            targetrun_id=self.targetrun_id,
            targetsteprun_id=self.id,
            targetstep_uri=self_entity.step_uri,
            target_step_index=self_entity.target_step_index,
        )
