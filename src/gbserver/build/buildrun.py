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
The build run.
"""

import asyncio
import fnmatch
import hashlib
import re
from asyncio import Queue, Task, TaskGroup
from dataclasses import asdict
from typing import Any, Coroutine, Dict, List, Optional, Self, Set

from gbcommon.uri.uri import URI
from gbserver.build.build import Build
from gbserver.build.run import Run, RunFailed
from gbserver.build.target import BindingInfo, Target
from gbserver.build.targetrun import TargetRun
from gbserver.environment.environment import BINDING_KEY, Environment
from gbserver.types.buildconfig import (
    BuildConfig,
    BuildTargetConfig,
    BuildTargetOutputConfig,
)
from gbserver.types.buildevent import (
    ArtifactEventPayload,
    ArtifactPushedEventPayload,
    BuildEvent,
    BuildEventStatusPayload,
    BuildEventType,
    EntityRunMetadata,
    MultiArtifactEventPayload,
)
from gbserver.types.constants import truncate
from gbserver.types.status import Status
from gbserver.utils.logger import get_logger
from gbserver.utils.utils import get_uuid, short_alphanumeric_lower_hash

# Matches {{ target_hash }} with any surrounding whitespace.
_TARGET_HASH_RE = re.compile(r"\{\{\s*target_hash\s*\}\}")

logger = get_logger(__name__)


def get_key_from_dict(k: str, d: Dict) -> Any:
    """
    Get the value of a nested key from a dict. e.g.
    "a.b.c" from {"a": {"b": {"c": 42}}} gives 42
    """
    if k == "":
        return None
    k_parts = k.split(".")
    if len(k_parts) == 0:
        return None
    dd = d
    for k_part in k_parts[:-1]:
        if isinstance(dd, dict):
            if k_part not in dd:
                return None
            dd = dd[k_part]
            continue
        if isinstance(dd, list):
            k_idx = int(k_part, base=10)
            if k_idx >= len(dd):
                return None
            dd = dd[k_idx]
            continue
        return None
    k_part = k_parts[-1]
    if k_part not in dd:
        return None
    return dd[k_part]


def match_event_selectors_against_payload(o: BuildTargetOutputConfig, p: dict) -> bool:
    """Returns True if the event payload matches the event selectors."""
    for es in o.event_selectors:
        pp = get_key_from_dict(es.field_name, p)
        if not isinstance(pp, str):
            continue
        if es.field_value:
            if es.field_value == pp:
                return True
        if es._field_value_regex:
            if es._field_value_regex.search(pp):
                return True
    return False


class BuildRun(Run):
    """Represents a single build run."""

    # instance attributes
    starting_targets: List[Target]
    targetruns: Dict[str, TargetRun]
    targetrun_additionaljobs_queue: Dict[str, Queue]
    targets_queue: Queue[BuildEvent]
    binding_to_target_mapping: Dict[BindingInfo, List[Target]]
    input_status_update_lock: asyncio.Lock
    cancel_on_error: bool

    def __init__(
        self: Self,
        build: Build,
        event_q: Optional[Queue] = None,
        cancel_on_error: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.starting_targets = []
        self.targetruns = {}
        self.targetrun_additionaljobs_queue = {}
        self.targets_queue = Queue()
        self.binding_to_target_mapping = {}
        self.tasks: Set[asyncio.Task] = set()
        self.input_status_update_lock: asyncio.Lock = asyncio.Lock()
        self.cancel_on_error = cancel_on_error
        super().__init__(
            entity=build,
            event_q=event_q,
            base_dir=build.dir,
            id=build.build_id,
            dry_run=dry_run,
        )
        logger.info(
            "BuildRun %s after super().__init__ self.event_q: %s %s",
            self.id,
            id(self.event_q),
            self.event_q,
        )

    async def _run(
        self: Self,
        tg: Optional[TaskGroup] = None,
        **kwargs,
    ) -> None:
        logger.debug("BuildRun._run %s start", self.id)
        self.starting_targets = []
        self.targets_queue = Queue()
        self.binding_to_target_mapping = {}
        self_entity = self.entity
        assert isinstance(self_entity, Build)
        if self.dry_run:
            logger.warning("dry_run flag is set for the build: %s", self_entity)
        logger.info("find targets to start with and binding information")
        for my_target_name, target in self_entity.targets.items():
            target_config = target.config
            assert isinstance(target_config, BuildTargetConfig)
            if target_config.inputs is None or len(target_config.inputs) == 0:
                logger.info(
                    "%s is a starting target because it has no inputs", my_target_name
                )
                if self.dry_run and not target.is_dry_run_compatible():
                    logger.warning(
                        "skipping the target '%s' because it's not dry run compatible: %s",
                        target.name,
                        target.config,
                    )
                else:
                    self.starting_targets.append(target)
                continue
            target_has_dependencies = False
            for binding_id, t_input in target_config.inputs.items():
                if t_input.binding:
                    target_has_dependencies = True
                    binding_target_name, binding_target_output_name = (
                        t_input.get_binding_parts()
                    )
                    binding_info = BindingInfo(
                        target_name=binding_target_name,
                        binding_id=binding_target_output_name,
                        wait_for_push=t_input.wait_for_push,  # type: ignore[arg-type]
                    )
                    if binding_info not in self.binding_to_target_mapping:
                        self.binding_to_target_mapping[binding_info] = []
                    self.binding_to_target_mapping[binding_info].append(target)
                elif t_input.uri:
                    event = asyncio.Event()
                    event.set()
                    if not hasattr(Environment._thread_local, "asset_events"):
                        Environment._thread_local.asset_events = {}  # URI to event map
                    normalized_uri = URI.get_uristr(URI.get_uri(t_input.uri))
                    Environment._thread_local.asset_events[normalized_uri] = event
                    binding_info = BindingInfo(binding_id=binding_id)
                    if binding_info not in self.binding_to_target_mapping:
                        self.binding_to_target_mapping[binding_info] = []
                    self.binding_to_target_mapping[binding_info].append(target)
                elif t_input.event:
                    target_has_dependencies = True
                    logger.error(
                        "unsupported input.event being used: %s", t_input.event
                    )
            if not target_has_dependencies:
                logger.info(
                    "%s is a starting target because it has no dependencies",
                    my_target_name,
                )
                if self.dry_run and not target.is_dry_run_compatible():
                    logger.warning(
                        "skipping the target '%s' because it's not dry run compatible: %s",
                        target.name,
                        target.config,
                    )
                else:
                    self.starting_targets.append(target)
        if tg is not None:
            logger.warning("ignoring the provided task group, we will create a new one")
        if self.cancel_on_error:
            async with TaskGroup() as build_taskgroup:
                await self._run_targets_of_build(tg=build_taskgroup)
        else:
            await self._run_targets_of_build()
        logger.debug("BuildRun._run %s end", self.id)

    def __compute_target_def_hash(self: Self, target: Target) -> str:
        """
        Computes a comprehensive SHA-256 hash of the target's definition:
        environment URI, step URIs, step configs, and resolved input URIs.
        This hash encodes "the exact work this target performs" and is stored
        in gb_targets for skip detection across builds.
        """
        target_config = target.config
        assert isinstance(target_config, BuildTargetConfig)
        env_uri = target_config.environment_uri or ""
        step_uris = [s.step_uri or "" for s in target_config.steps]
        step_configs = [s.model_dump_json() for s in target_config.steps]
        input_uris = sorted(uri for bi in target.inputs_status for uri in bi.uris)
        hash_input = (
            env_uri
            + "|"
            + "|".join(step_uris)
            + "|"
            + "|".join(step_configs)
            + "|"
            + "|".join(input_uris)
        )
        return hashlib.sha256(hash_input.encode()).hexdigest()

    def __propagate_binding(
        self: Self,
        binding_info: BindingInfo,
        uris: list[str],
        tg: Optional[TaskGroup],
    ) -> None:
        """Mark a binding as available for all downstream targets and dispatch any that are ready.

        Must be called while holding self.input_status_update_lock.
        uris must already be normalised URI strings (via URI.get_uristr).
        """
        if binding_info not in self.binding_to_target_mapping:
            return
        for target_to_consider in self.binding_to_target_mapping[binding_info]:
            if binding_info in target_to_consider.inputs_status:
                for bi in target_to_consider.inputs_status:
                    if bi == binding_info:
                        bi.available = True
                        for uri_str in uris:
                            bi.uris.append(uri_str)
            if not all(b.available for b in target_to_consider.inputs_status):
                logger.info(
                    "not all inputs of target %s are available yet",
                    target_to_consider.name,
                )
                continue
            logger.info(
                "all inputs of target %s are available, executing...",
                target_to_consider.name,
            )
            self.__dispatch_target(target_to_consider, tg)

    async def __handle_skipped_target(
        self: Self,
        target: Target,
        skipped_for_prerun_target_id: str,
        resolved_outputs: dict[str, list[str]],
        tg: Optional[TaskGroup],
    ) -> None:
        """
        Handles a target that has been run before with the same configuration:
        skips execution, notifies BuildRunner via STATUS_EVENT (type=Target, status=SUCCESS), and
        pre-resolves the input bindings of any downstream targets so they can proceed.

        resolved_outputs maps binding_id to the list of artifact URIs from the original run.
        """
        target_name = target.name
        logger.info(
            "Skipping target '%s' — previously executed with same configuration",
            target_name,
        )
        target_config = target.config
        assert isinstance(target_config, BuildTargetConfig)

        # Notify BuildRunner so it can persist a StoredTargetRun for this skipped target.
        self_entity = self.entity
        assert isinstance(self_entity, Build)
        skip_event = BuildEvent(
            run_metadata=EntityRunMetadata(
                build_id=self_entity.build_id,
                username=self_entity.username,
                type="Target",
                target_name=target_name,
                targetrun_id=get_uuid(),
                skipped_for_prerun_target_id=skipped_for_prerun_target_id,
            ),
            type=BuildEventType.STATUS_EVENT,
            payload=BuildEventStatusPayload(status=Status.SUCCESS),
        )
        self.dispatch_event(skip_event)

        # Pre-resolve downstream bindings using the original run's artifact URIs.
        async with self.input_status_update_lock:
            for binding_id, uris in resolved_outputs.items():
                out_config = (target_config.outputs or {}).get(binding_id)
                binding_info = BindingInfo(
                    binding_id=binding_id,
                    target_name=target_name,
                    output=out_config,
                )
                self.__propagate_binding(
                    binding_info,
                    [URI.get_uristr(URI.get_uri(uri_str)) for uri_str in uris],
                    tg,
                )

    def __resolve_output_uris(self: Self, target: Target) -> None:
        """
        Substitutes {{ target_hash }} in all output URIs of the target, in-place.
        target_hash is a deterministic hash of the target's step URIs and its resolved input URIs.
        URIs with no {{ are left unchanged. URIs that still contain {{ after substitution
        (because they reference variables other than target_hash) are left as-is; the actual
        artifact URIs from the original run are used for downstream binding pre-resolution.
        """
        target_config = target.config
        assert isinstance(target_config, BuildTargetConfig)
        outputs = target_config.outputs
        if not outputs:
            return
        has_template = any(
            out.uri is not None and "{{" in out.uri for out in outputs.values()
        )
        if not has_template:
            return
        step_uris = [s.step_uri for s in target_config.steps if s.step_uri]
        input_uris = sorted(uri for bi in target.inputs_status for uri in bi.uris)
        hash_input = "|".join(step_uris) + "|" + "|".join(input_uris)
        target_hash = short_alphanumeric_lower_hash(hash_input)
        for out_config in outputs.values():
            if out_config.uri is not None and "{{" in out_config.uri:
                # Use regex instead of template filling here to allow both
                # pre- and post- substitutions in the URI
                out_config.uri = _TARGET_HASH_RE.sub(target_hash, out_config.uri)

    def __dispatch_target(
        self: Self,
        target: Target,
        tg: Optional[TaskGroup],
    ) -> None:
        """
        Dispatches a target: schedules it to be skipped (if the same configuration has
        been run successfully before) or creates a TargetRun and schedules it.
        Adds the task to self.tasks.
        """
        self.__resolve_output_uris(target)
        self_entity = self.entity
        assert isinstance(self_entity, Build)
        target_already_run_fn = self_entity.target_already_run_fn
        target_def_hash = self.__compute_target_def_hash(target)
        skip_result = (
            target_already_run_fn(target_def_hash) if target_already_run_fn else None
        )
        asyncio_runner = tg if tg is not None else asyncio
        if skip_result is not None:
            skipped_for_prerun_target_id, resolved_outputs = skip_result
            self.tasks.add(
                asyncio_runner.create_task(
                    self.__handle_skipped_target(
                        target, skipped_for_prerun_target_id, resolved_outputs, tg
                    )
                )
            )
            return
        targetrun = TargetRun(
            target=target,
            event_q=self.targets_queue,
            cancel_on_error=self.cancel_on_error,
            dry_run=self.dry_run,
            target_hash=target_def_hash,
        )
        self.targetruns[targetrun.id] = targetrun
        if targetrun.id not in self.targetrun_additionaljobs_queue:
            self.targetrun_additionaljobs_queue[targetrun.id] = Queue()
        self.tasks.add(
            asyncio_runner.create_task(
                targetrun.run(
                    tg=tg,
                    additional_targetsteps_queue=self.targetrun_additionaljobs_queue[
                        targetrun.id
                    ],
                )
            )
        )

    async def _run_targets_of_build(self: Self, tg: Optional[TaskGroup] = None) -> None:
        logger.debug("BuildRun._run_targets_of_build %s start", self.id)
        if len(self.starting_targets) == 0:
            logger.warning("there are no starting targets")
        logger.info("run the %d starting targets", len(self.starting_targets))
        self.tasks = set()
        for target in self.starting_targets:
            self.__dispatch_target(target, tg)
        logger.info("loop waiting for targets to finish")
        while True:
            event = None
            try:
                event = await asyncio.wait_for(self.targets_queue.get(), timeout=1.0)
                asyncio_runner = tg if tg is not None else asyncio
                self.tasks.add(
                    asyncio_runner.create_task(self._process_event(event=event, tg=tg))  # type: ignore[arg-type]
                )
            except TimeoutError:
                if all(task.done() for task in self.tasks):
                    logger.info("all tasks are done")
                    break
        exceptions = []
        for task in self.tasks:
            if task.done():
                exception = task.exception()
                if exception:
                    exceptions.append(exception)
        if len(exceptions) > 0:
            raise RunFailed(status_updated=False, exceptions=exceptions)
        logger.debug("BuildRun._run_targets_of_build %s end", self.id)

    async def _process_event(self: Self, event: BuildEvent, tg: TaskGroup) -> None:
        event_str = truncate(str(event))
        logger.info(
            "build %s received event: %s : %s", self.build_id, event.type, event_str
        )
        if event.type is BuildEventType.ARTIFACT_PUSHED_EVENT:
            event_payload = event.payload
            assert isinstance(event_payload, ArtifactPushedEventPayload)
            assert event_payload.uri is not None, "event_payload.uri is None"
            uristr = URI.get_uristr(event_payload.uri)
            if uristr in Environment._thread_local.asset_events:
                Environment._thread_local.asset_events[uristr].set()
            else:
                logger.error("Unable to find asset_event for %s", uristr)
        self.dispatch_event(event=event)
        if not event.type.is_internal_event():
            return
        self_entity = self.entity
        assert isinstance(self_entity, Build)
        self_entity_config = self_entity.config
        assert isinstance(self_entity_config, BuildConfig)
        target_name = event.run_metadata.target_name
        assert target_name is not None, "the event.run_metadata.target_name is None"
        target = self_entity.targets[target_name]
        target_config = target.config
        assert isinstance(target_config, BuildTargetConfig)
        if event.type is BuildEventType.TERMINATE_EVENT:
            event_payload = event.payload
            logger.warning("TERMINATING %s", event_payload)
            self.cancel()
            return
        if event.type is BuildEventType.NEWARTIFACT_IN_ENVIRONMENT_EVENT:
            event_payload = event.payload
            assert isinstance(event_payload, ArtifactEventPayload)
            event = BuildEvent(
                run_metadata=event.run_metadata,
                type=BuildEventType.NEW_MULTIARTIFACT_IN_ENVIRONMENT_EVENT,
                payload=MultiArtifactEventPayload(artifacts=[event_payload]),
            )
        push_tasks: List[Task[URI]] = []
        event_payload = event.payload
        assert isinstance(event_payload, MultiArtifactEventPayload)
        for artifact in event_payload.artifacts:
            targetrun_id = event.run_metadata.targetrun_id
            assert targetrun_id is not None, "expected targetrun_id to not be empty"
            art_bind_id = artifact.binding_id
            self.targetruns[targetrun_id].bindings[art_bind_id] = {
                BINDING_KEY: artifact.binding
            }  # TODO: should we even set this if target_config.outputs is None ?
            if target_config.outputs is None:
                logger.warning(
                    "failed to find output binding %s, the target %s has no outputs. Ignoring.",
                    art_bind_id,
                    target_name,
                )
                continue
            found_key = ""
            found_value: Optional[BuildTargetOutputConfig] = None

            if art_bind_id in target_config.outputs:
                found_key = art_bind_id
                found_value = target_config.outputs[art_bind_id]
            else:
                logger.warning(
                    "failed to find output binding %s, in the outputs of the target %s . trying event selectors...",
                    art_bind_id,
                    target_name,
                )
                for out_name, out in target_config.outputs.items():
                    payload_dict = asdict(artifact)
                    matched = match_event_selectors_against_payload(out, payload_dict)
                    if matched:
                        found_key = out_name
                        found_value = out
                        # art_bind_id = output_name_glob
                        # self.targetruns[targetrun_id].bindings[art_bind_id] = {
                        #     BINDING_KEY: artifact.binding
                        # }
                        logger.info(
                            "matched out_name: %s out: %s",
                            out_name,
                            out,
                        )
                        break
                if found_value is None:
                    logger.warning(
                        "failed to find output binding %s using event selectors in the outputs of the target %s . trying glob...",
                        art_bind_id,
                        target_name,
                    )
                    for output_name_glob, out in target_config.outputs.items():
                        matching = fnmatch.filter(
                            names=[art_bind_id], pat=output_name_glob
                        )
                        if len(matching) > 0:
                            found_key = output_name_glob
                            found_value = out
                            logger.info(
                                "matched glob out_name: %s out: %s",
                                out_name,
                                out,
                            )
                            break
            if found_value is None:
                logger.warning(
                    "failed to find output binding %s, in the outputs of the target %s . Ignoring.",
                    art_bind_id,
                    target_name,
                )
                continue
            _artifact_uri_str = found_value.uri
            _artifact_uri_str = "" if _artifact_uri_str is None else _artifact_uri_str
            additional_targetsteps_queue = self.targetrun_additionaljobs_queue[
                targetrun_id
            ]
            _space_name = (
                self_entity.space.space_config.name if self_entity.space else None
            )
            task = target.environment.pushasset(
                task_group=tg,
                binding=artifact.binding,
                uristr=_artifact_uri_str,
                binding_id=art_bind_id,
                additional_targetsteps_queue=additional_targetsteps_queue,
                run_metadata=event.run_metadata,
                output_config=found_value.model_copy(
                    update={"space_name": _space_name}
                ),
            )
            task.binding_id = art_bind_id  # type: ignore[attr-defined]
            task.binding_id_glob = found_key  # type: ignore[attr-defined]
            push_tasks.append(task)
        uris = await asyncio.gather(*push_tasks, return_exceptions=True)
        if target_config.outputs is None:
            logger.error(
                "the target %s has no outputs, ignoring the uris from push tasks: %s",
                target_name,
                uris,
            )
            return
        async with self.input_status_update_lock:
            exceptions = []
            for task, uri in zip(push_tasks, uris):
                if isinstance(uri, BaseException):
                    exceptions.append(uri)
                    continue
                task_binding_id = task.binding_id  # type: ignore[attr-defined]
                assert isinstance(task_binding_id, str)
                task_binding_id_glob = task.binding_id_glob  # type: ignore[attr-defined]
                assert isinstance(
                    task_binding_id_glob, str
                ), f"invalid task_binding_id_glob: {task_binding_id_glob}"
                binfo_output = target_config.outputs.get(task_binding_id)
                if binfo_output is None:
                    logger.info(
                        "failed to find a binding %s , looking with glob",
                        task_binding_id,
                    )
                    binfo_output = target_config.outputs.get(task_binding_id_glob)
                if binfo_output is None:
                    raise ValueError(
                        f"failed to find a binding {task_binding_id} {task_binding_id_glob}"
                    )
                binding_info = BindingInfo(
                    binding_id=task_binding_id,
                    target_name=event.run_metadata.target_name,
                    output=binfo_output,
                )
                self.__propagate_binding(
                    binding_info,
                    [URI.get_uristr(uri)],
                    tg,
                )
            if len(exceptions) > 0:
                raise RunFailed(status_updated=False, exceptions=exceptions)

    def get_runmetadata(self: Self) -> EntityRunMetadata:
        return EntityRunMetadata(
            build_id=self.build_id,
            username=self.entity.username,
            type=type(self.entity).__name__,
        )

    async def _cleanup(self: Self, tg: Optional[TaskGroup] = None) -> None:
        logger.info("BuildRun._cleanup start")
        self_entity = self.entity
        assert isinstance(self_entity, Build)
        teardown_tasks: set[Task] = set()
        for target in self_entity.targets.values():
            teardown_tasks.add(asyncio.create_task(target.teardown()))
        if len(teardown_tasks) > 0:
            await asyncio.wait(teardown_tasks)
        logger.info("BuildRun._cleanup end")
