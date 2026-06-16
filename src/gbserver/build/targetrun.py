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
The target run.
"""

import asyncio
from asyncio import Event, Queue, TaskGroup
from copy import deepcopy
from typing import Dict, Optional, Self

from gbserver.build.run import Run, RunFailed
from gbserver.build.target import Target
from gbserver.build.targetstep import TargetStep
from gbserver.build.targetsteprun import TargetStepRun
from gbserver.environment.environment import Environment
from gbserver.types.buildconfig import BuildTargetStepConfig
from gbserver.types.buildevent import BuildEvent, BuildEventType, EntityRunMetadata
from gbserver.types.status import Status
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class TargetRun(Run):
    """Represents a single target run."""

    target_step_runs: set[TargetStepRun]
    additional_running_steps: set[asyncio.Task]

    def __init__(
        self: Self,
        target: Target,
        event_q: Queue,
        cancel_on_error: bool = False,
        dry_run: bool = False,
        target_hash: str = "",
    ) -> None:
        """Loads a target run"""
        self.bindings: Dict[str, Dict] = {}
        self.target_step_runs = set()
        self.additional_running_steps = set()
        self.inputs_status = deepcopy(target.inputs_status)
        self.cancel_on_error = cancel_on_error
        self.target_hash = target_hash
        super().__init__(
            entity=target, event_q=event_q, base_dir=target.dir, dry_run=dry_run
        )

    async def _run(
        self: Self,
        tg: Optional[TaskGroup] = None,
        additional_targetsteps_queue: Optional[Queue] = None,
        pushes_enqueued: Optional[Event] = None,
        **kwargs,
    ) -> None:
        self.target_step_runs = set()
        self.additional_running_steps = set()
        self_entity = self.entity
        assert isinstance(self_entity, Target)
        async with asyncio.TaskGroup() as tg:
            await self_entity.setup(tg, **kwargs)
            input_uris = {}
            try:
                for binding in self.inputs_status:
                    if binding.wait_for_push:
                        assert len(binding.uris) > 0, "empty binding.uris"
                        uristr = binding.uris[-1]
                        if not hasattr(Environment._thread_local, "asset_events"):
                            logger.error(
                                "No asset_events found in Environment to wait for asset push. Proceeding."
                            )
                        elif uristr not in Environment._thread_local.asset_events:
                            logger.error(
                                "No asset_event found for uri %s for asset push. Proceeding.",
                                uristr,
                            )
                        else:
                            logger.info("Waiting for asset push of %s", uristr)
                            await Environment._thread_local.asset_events[uristr].wait()
                            logger.info("Asset push of %s done. Proceeding.", uristr)
                pending = self_entity.pull_assets(list(self.inputs_status), tg)
                targetstepruntasks = []
                while pending:
                    done, pending = await asyncio.wait(  # type: ignore[assignment]
                        pending, return_when=asyncio.FIRST_COMPLETED
                    )
                    exceptions = []
                    for task in done:
                        exception = task.exception()
                        if exception:
                            exceptions.append(exception)
                            continue
                        binding, targetstep_config = task.result()  # type: ignore[assignment]
                        if isinstance(targetstep_config, BuildTargetStepConfig):
                            targetstep_run = self.get_targetsteprun_from_config(
                                targetstep_config, additional_targetsteps_queue  # type: ignore[arg-type]
                            )
                            if targetstep_run:
                                self.target_step_runs.add(targetstep_run)
                                targetstep_run_task = asyncio.create_task(
                                    targetstep_run.run(tg)
                                )
                                targetstepruntasks.append(targetstep_run_task)
                            else:
                                logger.warning(
                                    "didn't get a targetstep_run, ignoring..."
                                )
                        self.bindings[task.binding_id] = binding  # type: ignore[assignment, attr-defined]
                        input_uris[task.binding_id] = task.uri  # type: ignore[attr-defined]
                    if len(exceptions) > 0:
                        raise RunFailed(status_updated=False, exceptions=exceptions)
                results = await asyncio.gather(
                    *targetstepruntasks, return_exceptions=True
                )
                failed_tasks = [r for r in results if isinstance(r, BaseException)]
                if failed_tasks:
                    raise RunFailed(status_updated=False, exceptions=failed_tasks)
                self.metadata["inputs"] = input_uris
                self.update_status(Status.RUNNING)
            except Exception as e:
                raise ValueError("failed during loading artifacts") from e
            all_targetstep_runs_done = Event()
            asyncio_runner = tg
            if not self.cancel_on_error:
                asyncio_runner = asyncio  # type: ignore[assignment]
            run_additional_targetsteps_task = asyncio_runner.create_task(
                self.run_additional_targetsteps(
                    additional_targetsteps_queue,  # type: ignore[arg-type]
                    all_targetstep_runs_done,
                    (tg if self.cancel_on_error else None),
                )
            )
            for targetstep in self_entity.targetsteps:
                targetstep_run = TargetStepRun(
                    target=self_entity,
                    targetstep=targetstep,
                    targetrun_id=self.id,
                    event_q=self.event_q,
                    additional_targetsteps_queue=additional_targetsteps_queue,
                    bindings=self.bindings,
                    setup_config=self_entity.setup_config,
                    dry_run=self.dry_run,
                )
                self.target_step_runs.add(targetstep_run)
                await targetstep_run.run(tg)
            # All explicit steps are done, so all of this target's output-artifact
            # events have been emitted onto the (FIFO) event queue. Emit a
            # sentinel after them and wait until BuildRun confirms it has
            # processed those events and enqueued every output-push step. Only
            # then signal done — otherwise the additional-steps consumer could
            # observe an empty queue and exit before the push configs arrive,
            # orphaning the queued push steps and leaving artifacts pending.
            await self.event_q.put(
                BuildEvent(
                    run_metadata=self.get_runmetadata(),
                    type=BuildEventType.TARGET_ARTIFACTS_DONE_EVENT,
                )
            )
            if pushes_enqueued is not None:
                await pushes_enqueued.wait()
            all_targetstep_runs_done.set()
            await run_additional_targetsteps_task
            results = await asyncio.gather(
                *self.additional_running_steps, return_exceptions=True
            )
            failed_tasks = [r for r in results if isinstance(r, BaseException)]
            if failed_tasks:
                raise RunFailed(status_updated=False, exceptions=failed_tasks)

    async def _cleanup(self: Self, tg: Optional[TaskGroup] = None) -> None:
        pass

    def get_runmetadata(self: Self) -> EntityRunMetadata:
        self_entity = self.entity
        assert isinstance(self_entity, Target)
        return EntityRunMetadata(
            build_id=self.build_id,
            username=self_entity.username,
            type=type(self_entity).__name__,
            target_name=self_entity.name,
            targetrun_id=self.id,
            target_hash=self.target_hash,
        )

    async def run_additional_targetsteps(
        self: Self,
        additional_targetsteps_queue: Queue,
        all_targetstep_runs_done: Event,
        tg: Optional[TaskGroup] = None,
    ) -> None:
        """Run some additional (usually implicit) steps for pull, push, etc."""
        self_entity = self.entity
        assert isinstance(self_entity, Target)
        while True:
            try:
                queue_task = asyncio.create_task(additional_targetsteps_queue.get())
                event_task = asyncio.create_task(all_targetstep_runs_done.wait())
                done, pending = await asyncio.wait(
                    [queue_task, event_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if queue_task in done:
                    targetstepconfig: BuildTargetStepConfig = queue_task.result()
                    targetstep_run = self.get_targetsteprun_from_config(
                        targetstepconfig, additional_targetsteps_queue
                    )
                    if targetstep_run:
                        self.target_step_runs.add(targetstep_run)
                        targetstep_run_task = asyncio.create_task(
                            targetstep_run.run(tg)
                        )
                        self.additional_running_steps.add(targetstep_run_task)
                    else:
                        logger.warning("didn't get a targetstep_run, ignoring...")
                if (
                    all_targetstep_runs_done.is_set()
                    and additional_targetsteps_queue.empty()
                ):
                    break
            except asyncio.CancelledError as e:
                logger.error("run_additional_targetsteps cancelled : %s", e)
                break

    def get_targetsteprun_from_config(
        self: Self,
        targetstepconfig: BuildTargetStepConfig,
        additional_targetsteps_queue: Queue,
    ) -> Optional[TargetStepRun]:
        """Create a target step run from the given config (usually an implicit step)."""
        self_entity = self.entity
        assert isinstance(self_entity, Target)
        targetstep = TargetStep(
            self.build_id,
            self.event_q,
            targetstepconfig,
            self_entity.name,
            self_entity.environment,
            self_entity.build_workspace_dir,
            self_entity.dir,  # type: ignore[arg-type]
            username=self_entity.username,
            context=self_entity.context,
            force_fetch=self_entity.force_fetch,
            parent_target_config=self_entity.config,  # type: ignore[arg-type]
        )
        if self.dry_run and not targetstep.is_dry_run_compatible():
            logger.warning("dry_run: skip running the incompatible step %s", targetstep)
            return None
        targetstep_run = TargetStepRun(
            target=self_entity,
            targetstep=targetstep,
            targetrun_id=self.id,
            event_q=self.event_q,
            additional_targetsteps_queue=additional_targetsteps_queue,
            bindings=self.bindings,
            setup_config=self_entity.setup_config,
            dry_run=self.dry_run,
        )
        return targetstep_run
