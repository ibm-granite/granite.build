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
The environment where the workloads run.
"""

import asyncio
import glob
import hashlib
import importlib
import inspect
import json
import os
import re
import tempfile
import threading
import traceback
from abc import ABC, abstractmethod
from asyncio import Event, Queue, StreamReader, Task, TaskGroup
from asyncio.subprocess import Process
from contextlib import asynccontextmanager
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Self,
    Set,
    Tuple,
    Type,
    Union,
)

if TYPE_CHECKING:
    from gbserver.resilience.retry_handler import RetryHandler, RetryStrategy

from pydantic import Field

from gbcommon.types.testing import is_failure_simulated
from gbcommon.uri.uri import URI
from gbserver.asset.asset import Asset
from gbserver.asset.assetstore import Assetstore
from gbserver.messaging.messaging_base import MessagingBase
from gbserver.types.buildconfig import BuildTargetOutputConfig, BuildTargetStepConfig
from gbserver.types.buildevent import (
    ArtifactEventPayload,
    ArtifactPushedEventPayload,
    BuildEvent,
    BuildEventMessagePayload,
    BuildEventStatusPayload,
    BuildEventType,
    CreatedArtifactEventPayload,
    EntityRunMetadata,
    EventPayload,
)
from gbserver.types.config import Config
from gbserver.types.constants import (
    FULL_CONFIG_RUN_METADATA_KEY,
    GBSERVER_ENABLE_STEP_RETRY,
)
from gbserver.types.environment.environment import StepConfigSection
from gbserver.types.environmentconfig import (
    ENVIRONMENT_FILENAME,
    AssetStoreEnvironmentConfig,
    EnvironmentConfig,
)
from gbserver.types.status import Status
from gbserver.utils.logger import get_logger
from gbserver.utils.template import fill_template
from gbserver.utils.utils import get_uuid

logger = get_logger(__name__)

BINDING_KEY = "binding"


class EventFieldRegexLogParserConfig(Config):
    """Details of a single field in the event payload."""

    field_name: str
    field_value_template: Optional[str] = None
    field_regex: Optional[str] = None
    is_json: Optional[bool] = False
    is_data: Optional[bool] = False
    # Private fields
    _field_pattern: Optional[re.Pattern] = None


class EventLogLineParserConfig(Config):
    """Details on constructing the event payload from the log line."""

    event_type: str = Field(
        ..., description="Event type generated after parsing log line"
    )
    line_regex: str = Field(
        ..., description="Primary regex pattern for extracting text of interest"
    )
    is_json: Optional[bool] = False
    event_fields: List[EventFieldRegexLogParserConfig] = Field(
        default_factory=list,
        description="List of field names and associated regex patterns for further extracting data",
    )
    # Private fields
    _line_pattern: Optional[re.Pattern] = None

    def __init__(self: Self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.compile()

    def compile(self: Self) -> None:
        """Compile the regexes to make sure they are valid"""
        self._line_pattern = re.compile(self.line_regex)
        for event_field in self.event_fields:
            if event_field.field_regex is not None:
                event_field._field_pattern = re.compile(event_field.field_regex)
            else:
                event_field._field_pattern = None


class UnknownEnvironmentType(Exception):
    """Unknown Environment Type exception"""


class RetryArtifactFilterQueue(asyncio.Queue):
    """An asyncio.Queue proxy that filters NEWARTIFACT events in real-time.

    Wraps an outer event_q and forwards events to it as they are put(),
    applying retry_transparently dedup logic inline with no post-iteration drain.

    - ``retry_transparently=True`` (default): deduplicate NEWARTIFACT events by
      the basename of the binding path across all retry iterations; first occurrence
      is forwarded, duplicates are dropped.
    - ``retry_transparently=False``: all events are forwarded without filtering.

    Usage::

        filtering_q = RetryArtifactFilterQueue(event_q=event_q, retry_transparently=retry_transparently)
        while True:
            logfile_monitor = LogFileMonitor(..., event_queue=filtering_q, ...)
            await asyncio.gather(job_monitor.monitor(), logfile_monitor.monitor())
            if retry_complete_event.is_set():
                ...
                continue
            break
    """

    def __init__(self, event_q: Optional[Queue], retry_transparently: bool) -> None:
        super().__init__()
        self._event_q = event_q
        self.retry_transparently = retry_transparently
        self._seen: Set[str] = set()

    def _should_forward(self, ev: Any) -> bool:
        if not self.retry_transparently:
            return True
        if ev.type == BuildEventType.NEWARTIFACT_IN_ENVIRONMENT_EVENT:
            payload = ev.payload
            if isinstance(payload, ArtifactEventPayload):
                binding_path = (payload.binding or {}).get("path", "")
                dedup_key = os.path.basename(binding_path) if binding_path else ""
                if dedup_key and dedup_key in self._seen:
                    logger.info(
                        "Skipping duplicate NEWARTIFACT for binding path basename %s "
                        "(already seen in a previous try)",
                        dedup_key,
                    )
                    return False
                if dedup_key:
                    self._seen.add(dedup_key)
        return True

    async def put(self, item: Any) -> None:  # type: ignore[override]
        if self._should_forward(item) and self._event_q is not None:
            await self._event_q.put(item)

    def put_nowait(self, item: Any) -> None:
        if self._should_forward(item) and self._event_q is not None:
            self._event_q.put_nowait(item)


class Environment(ABC):
    """
    An environment where the workloads can run.
    Supports environment setup and teardown (setup id), and cleanup launch and monitor for individual workloads (launch ids).
    Multiple launches can be issued between setup and teardown. 1 or more launchers can be defined
    within this class idendified by a suffix on the launch_SUFFIX/cleanup_SUFFIX.  These suffixes are then referenced
    in the step.yaml file for a given environment in which the step might run.
    1 or more monitor methods can be defined as monitor_SUFFIX() methods.  These suffixes (usually not the same suffixes
    as are using with luanch and cleanup) are then referenced/set in the environment.yaml, corresponding to the
    sub-class implementation of the Environment class, indicating which monitors should run for which launch suffixes.

    Calls to setup_SUFFIX/teardown_SUFFIX are grouped by a common setup id.
    Calls to cleanup_SUFFIX/launch_SUFFIX/monitor_SUFFIX (from targetsteprun) are grouped by a common launch id.
    For a given setup and launch id, there is an enforced synchronization of these methods.

    * setup_xyz(setup_id) called from setup(setupid) is generally the first function called
    * launch_xyz(launch_id) called from launch(launch_id),  will not be called until setup_xyz() has completed.
    launch_xyz() may return before the workload is completed, but should then call release_monitor(launch_id) to
    enable monitoring.  If launch_xyz() throws an exception, monitoring of that launch_id will not be triggered.
    * cleanup_xyz(launch_id) called from cleanup(launch_id), will not be called until launch_xyz() has completed
    * teardown_xyz(setup_id) called from teardown(setupid), will not be called until all setup, launch and cleanup
    calls have been completed.
    * monitor_abc(launch_id) called from monitor(launch_ic), is not called until release_monitor(launch_id) is called,
    usually by launch_xyz(launch_id).

    launch and monitor methods can make use of the event provided by _get_launch_stopped_event(launch_id)
    to signal that the workload running in the launch is completed.  Often there is 1 monitor that simply
    sets this event when workload completion is detected.  launch_xyz() or other monitors can then use this event
    to determine when they should terminate/return.


    """

    # class attributes
    _thread_local = threading.local()
    environment_types: Dict[str, Type[Self]] = {}
    # instance attributes
    __setup_done_events: Dict[str, Event]
    __launch_ready_events: Dict[str, Event]
    __launch_done_events: Dict[str, Event]
    __launch_failed_events: Dict[str, Event]
    __cleanup_done_events: Dict[str, Event]
    pullasset_types: Dict[str, Callable]
    pushasset_types: Dict[str, Callable]
    __launch_stopped_events: Dict[
        str, asyncio.Event
    ]  # stop events to exit monitoring, one per launch_id

    def __init__(
        self: Self,
        event_q: Queue,
        environment_config: Optional[EnvironmentConfig] = None,
        context: Optional[str] = None,
        secrets: Optional[dict] = None,
        environment_asset: Optional[Asset] = None,
        **kwargs,
    ) -> None:
        self.type = self.__class__.__name__
        self.event_q = event_q
        self.config = environment_config
        self.context = context
        self.secrets = secrets
        self.environment_asset = environment_asset
        self.__setup_done_events: Dict[str, Event] = {}
        self.__launch_done_events: Dict[str, Event] = {}
        self.__launch_failed_events: Dict[str, Event] = {}
        self.__launch_ready_events: Dict[str, Event] = {}
        self.__cleanup_done_events: Dict[str, Event] = {}
        self.__teardown_started_events: Dict[str, Event] = {}
        self.setup_ids: Dict[str, list[str]] = {}
        self.__launch_stopped_events: Dict[str, asyncio.Event] = (
            {}
        )  # stop events to exit monitoring, one per launch_id
        self.asset_bindings: Dict[str, Dict] = {}
        self.step_type_chain: List[str] = self._compute_step_type_chain()
        self.tasks: List[Task] = []
        self.launch_tasks: List[Task] = []
        self.monitor_tasks: List[Task] = []
        self.setup_tasks: List[Task] = []
        self.teardown_tasks: List[Task] = []
        self.launch_types = self._get_fns_with_prefix(prefix="launch_")
        self.cleanup_types = self._get_fns_with_prefix(prefix="cleanup_")
        self.monitor_types = self._get_fns_with_prefix(prefix="monitor_")
        self.setup_types = self._get_fns_with_prefix(prefix="setup_")
        self.teardown_types = self._get_fns_with_prefix(prefix="teardown_")
        self.pullasset_types = self._get_fns_with_prefix(prefix="pullasset_")
        self.pushasset_types = self._get_fns_with_prefix(prefix="pushasset_")
        self.supported_assetstores: Dict[Assetstore, AssetStoreEnvironmentConfig] = {}
        self._load_assetstores()

    def _compute_step_type_chain(self: Self) -> List[str]:
        """Return the normalized step_type chain for this environment.

        Reads ``EnvironmentConfig.step_type`` (which may be a string or a list)
        and produces an ordered list of step_type strings used by the SpaceURI
        resolver to narrow ``space://steps/<name>`` lookups.  Returns an empty
        list when the env config is missing or the field is unset, so callers
        can treat ``not env.step_type_chain`` as "no step_type filtering".

        Returns:
            Ordered list of step_type strings, most-preferred first.  Empty
            list when the env opts out of step_type-narrowing.
        """
        if self.config is None:
            return []
        value = getattr(self.config, "step_type", None)
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    @property
    def environment_dir_uri(self: Self) -> Optional[str]:
        """URI of the directory that contains this env's environment.yaml.

        Used by the SpaceURI resolver to discover env-co-located steps at
        ``<env-dir>/steps/<name>/`` when this env is the active target's env.
        Returned with no trailing slash.  Returns ``None`` when the env was
        constructed without an asset (e.g. some unit tests), in which case
        the resolver skips the env-dir tier and falls through to the
        ``step_type`` chain.
        """
        if self.environment_asset is None:
            return None
        uristr = self.environment_asset.uristr
        if not uristr:
            return None
        return uristr.rstrip("/")

    async def retry_workload(
        self: Self,
        launch_id: str,
        nodes_to_avoid: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """
        Retry a failed workload with optional node avoidance.

        This method should be implemented by subclasses to provide
        environment-specific retry logic. The default implementation
        does nothing (no retry support).

        Args:
            launch_id: The launch identifier for the workload
            nodes_to_avoid: Optional list of node names to avoid (K8s-specific)
            **kwargs: Additional environment-specific parameters

        Raises:
            NotImplementedError: If the environment doesn't support retries
        """
        logger.warning(
            "[Environment %s launch_id %s] retry_workload not implemented, no retry will be attempted",
            self.__class__.__name__,
            launch_id,
        )
        raise NotImplementedError(
            f"retry_workload not implemented for {self.__class__.__name__}"
        )

    def _is_retry_enabled_at_environment_level(self: Self) -> bool:
        """Return True when the environment config has not explicitly disabled retry.

        Retry is enabled by default (backwards compatible). It can be turned off
        by setting ``retry.enabled: false`` in the environment YAML.
        """
        if self.config is None:
            return True
        retry_config = self.config.config.get("retry", {})
        return retry_config.get("enabled", True)

    def _get_default_retry_strategies(self: Self) -> List["RetryStrategy"]:
        """Return the built-in retry strategies for this environment type.

        Called by _get_retry_strategies() when the environment YAML does not
        specify an explicit ``retry.strategies`` list. Override in subclasses
        to provide environment-specific defaults.
        """
        return []

    def _get_retry_strategies(self: Self) -> List["RetryStrategy"]:
        """Return active retry strategies, honouring the environment config.

        Resolution order:
        1. If ``retry.enabled: false``, return [].
        2. If ``retry.strategies`` is present in the YAML, build from that list.
        3. Otherwise fall back to ``_get_default_retry_strategies()``.
        """
        if not self._is_retry_enabled_at_environment_level():
            return []
        if self.config is not None:
            retry_config = self.config.config.get("retry", {})
            if "strategies" in retry_config:
                from gbserver.resilience import build_retry_strategies_from_config

                return build_retry_strategies_from_config(
                    config=retry_config["strategies"],
                    object_types=retry_config.get("object_types"),
                )
        return self._get_default_retry_strategies()

    def _get_retry_max_retries(self: Self) -> int:
        """Return the maximum number of retries for this environment.

        Reads ``retry.max_retries`` from the environment config when present,
        falling back to 3. Subclasses do not need to override this unless they
        require a different fallback.
        """
        if self.config is not None:
            return self.config.config.get("retry", {}).get("max_retries", 3)
        return 3

    def _get_retry_test_scenario(self: Self) -> Optional[str]:
        """Return the scenario name used by _inject_event_to_trigger_retry_when_testing.

        Override in subclasses to enable simulated failure injection during testing.
        Returns None (no injection) by default.
        """
        return None

    def _create_retry_handler_for_launch(
        self: Self,
        launch_id: str,
        event_q: asyncio.Queue,
        build_id: Optional[str] = None,
        node_health_tracker=None,
        entityrun_metadata=None,
    ) -> Optional["RetryHandler"]:
        """Create a RetryHandler for a launch using this environment's strategies.

        Returns None when _get_retry_strategies() returns an empty list.
        """
        strategies = self._get_retry_strategies()
        if not strategies:
            return None
        from gbserver.resilience import RetryHandler  # avoid circular import

        max_retries = self._get_retry_max_retries()
        logger.info(
            "[%s launch_id %s] Creating RetryHandler with %d strategies: %s (max_retries=%d)",
            self.__class__.__name__,
            launch_id,
            len(strategies),
            [s.__class__.__name__ for s in strategies],
            max_retries,
        )
        return RetryHandler(
            launch_id=launch_id,
            downstream_queue=event_q,
            environment=self,
            max_retries=max_retries,
            strategies=strategies,
            node_health_tracker=node_health_tracker,
            build_id=build_id or launch_id,
            entityrun_metadata=entityrun_metadata,
        )

    def _get_step_retry_config(
        self: Self,
        launch_params_for_id: dict,
    ) -> tuple[bool, bool]:
        """Resolve the retry config for a step launch.

        ``GBSERVER_ENABLE_STEP_RETRY`` is the global gate: if False, retries are
        disabled immediately regardless of any per-step or per-run config.

        When the gate is open, priority chains are:

          ``enabled``:
            1. build.yaml per-step ``retry_enabled``
            2. step.yaml ``retry_enabled_default``
            3. False (retries disabled unless explicitly configured at step level)

          ``retry_transparently``:
            1. build.yaml per-step ``retry_transparently``
            2. step.yaml ``retry_transparently_default``

        Args:
            launch_params_for_id: The full params dict for this launch_id
                                  (``self._launch_params[launch_id]`` or
                                  ``self._launch_kwargs[launch_id]``).

        Returns:
            ``(enabled, retry_transparently)`` — the resolved retry enabled flag
            and the resolved retry_transparently flag.
        """
        if not GBSERVER_ENABLE_STEP_RETRY:
            return False, False
        launch_config = launch_params_for_id.get("config", {})
        step_config = StepConfigSection.model_validate(launch_config)
        run_retry_transparently = launch_params_for_id.get("retry_transparently")
        if run_retry_transparently is not None:
            retry_transparently = run_retry_transparently
        else:
            retry_transparently = step_config.retry_transparently_default
        run_retry_enabled = launch_params_for_id.get("retry_enabled")
        if run_retry_enabled is not None:
            enabled = run_retry_enabled
        elif step_config.retry_enabled_default is not None:
            enabled = step_config.retry_enabled_default
        else:
            enabled = False
        return enabled, retry_transparently

    @asynccontextmanager
    async def _with_retry_handler(
        self: Self,
        launch_id: str,
        event_q: Optional[asyncio.Queue],
        build_id: Optional[str] = None,
        node_health_tracker=None,
        enabled: bool = True,
        entityrun_metadata=None,
        retry_transparently: Optional[bool] = None,
    ):
        """Async context manager for the full RetryHandler lifecycle.

        Yields monitor_queue: the RetryHandler input queue, or event_q when retry
        is inactive. Pass this queue to all monitors.

        Args:
            launch_id: The launch identifier
            event_q: Downstream event queue
            build_id: Build ID for tracking (defaults to launch_id)
            node_health_tracker: Optional node health tracker (K8s-specific)
            enabled: When False, disables retry attempts (max_retries=0) and test event injection,
                     but still detects terminal failures and raises WorkloadFailedException.
            entityrun_metadata: Optional metadata for the entity run (used for retry messages).
            retry_transparently: When not None, wraps event_q in a RetryArtifactFilterQueue
                with this flag before creating the handler. Pass None (default) to skip
                wrapping (e.g. when there are no artifact events to filter).

        Usage::

            async with self._with_retry_handler(launch_id, event_q, build_id) as monitor_q:
                # pass monitor_q to all monitors
        """
        if event_q is None:
            raise ValueError(
                f"_with_retry_handler requires an event queue for launch_id={launch_id}"
            )
        downstream: Optional[asyncio.Queue] = event_q
        if retry_transparently is not None:
            downstream = RetryArtifactFilterQueue(
                event_q=event_q, retry_transparently=retry_transparently
            )
        handler = self._create_retry_handler_for_launch(
            launch_id, downstream, build_id, node_health_tracker, entityrun_metadata  # type: ignore[arg-type]
        )
        if handler is None:
            yield downstream
            return
        # When retry is disabled, still create the handler with max_retries=0
        # so that terminal failures (e.g. Failed AppWrapper) are detected and
        # raised as WorkloadFailedException instead of being silently ignored.
        if not enabled:
            handler.max_retries = 0
        task = asyncio.create_task(handler.process_events())
        if enabled:
            await self._inject_event_to_trigger_retry_when_testing(
                handler, self._get_retry_test_scenario()
            )
        try:
            yield handler.get_wrapper_queue()
        finally:
            handler.stop()
            await task

    def _dispatch_event(self: Self, event: Event) -> None:
        """Dispatch an event if the event queue exists."""
        if self.event_q is None:
            logger.debug("Environment._dispatch_event no event_q")
            return
        self.event_q.put_nowait(event)

    def _send_message(self: Self, msg: str, **kwargs) -> None:
        """Create and dispatch a MESSAGE_EVENT with the given text.

        Args:
            msg: Human-readable message text to include in the event payload.
            **kwargs: Must contain 'run_metadata' as a dict; if absent or not a
                dict, the message is silently dropped.
        """
        logger.info("msg: %s", msg)
        _run_metadata = kwargs.get("run_metadata", None)
        if not isinstance(_run_metadata, dict):
            return
        run_metadata = EntityRunMetadata.from_dict(_run_metadata)
        event = BuildEvent(
            type=BuildEventType.MESSAGE_EVENT,
            run_metadata=run_metadata,
            payload=BuildEventMessagePayload(msg=msg),
        )
        self._dispatch_event(event=event)

    def _load_assetstores(self: Self) -> None:
        """Load the asset stores specified in the environment.yaml"""
        if self.config is None or self.config.assetstores is None:
            logger.warning("No asset stores found!")
            return
        for storeenv in self.config.assetstores:
            assetstore = Asset.get_assetstore_from_store_uri(
                store_uri=storeenv.store_uri, context=self.context
            )
            self.supported_assetstores[assetstore] = storeenv

    @classmethod
    def get_environment(
        cls: Type[Self],
        environment_uri: str,
        event_q: Queue,
        context: Optional[str] = None,
        secrets: Optional[dict] = None,
        force_fetch: bool = False,
    ) -> Self:
        """Resolve environment uri to environment"""
        if not hasattr(cls._thread_local, "environments"):
            cls._thread_local.environments = {}
            cls._thread_local.environmentcache_dir = Path(tempfile.mkdtemp())
        if not hasattr(cls._thread_local, "asset_events"):
            cls._thread_local.asset_events = {}  # URI to event map
        if environment_uri in cls._thread_local.environments:
            return cls._thread_local.environments[environment_uri]
        environment_asset = Asset(environment_uri, context=context)
        # Set environment_uri again after any templates get filled
        environment_uri = environment_asset.uristr
        th_env_dir = cls._thread_local.environmentcache_dir
        assert isinstance(th_env_dir, Path)
        environmentasset_dir = th_env_dir / environment_asset.urihash()
        environment_asset.sync(dest=environmentasset_dir, force=force_fetch)
        files = glob.glob(
            str(environmentasset_dir / "**" / ENVIRONMENT_FILENAME), recursive=True
        )
        if len(files) == 0:
            raise UnknownEnvironmentType(
                f"{ENVIRONMENT_FILENAME} not found in {environmentasset_dir}"
            )
        env_yaml_path = Path(files[0])
        if len(files) > 1:
            logger.warning(
                "more than one %s found, using %s", ENVIRONMENT_FILENAME, env_yaml_path
            )
        environment_config = EnvironmentConfig.from_yaml(env_yaml_path, context=context)
        if (
            environment_config.type is None
            or environment_config.type == ""
            or environment_config.type not in cls.environment_types
        ):
            raise UnknownEnvironmentType(
                f"Environment Type `{environment_config.type}` from {environment_uri} is unknown"
            )
        env_class = cls.environment_types[environment_config.type]
        env = env_class(
            event_q=event_q,
            environment_config=environment_config,
            context=context,
            secrets=secrets,
            environment_asset=environment_asset,
        )
        cls._thread_local.environments[environment_uri] = env
        return env

    def __cancel_monitoring(self, launch_id: str):
        """Tells monitors that launch has stopped and they should not proceed."""
        self._get_launch_stopped_event(launch_id).set()

    def launch(
        self: Self,
        launcher_type: str,
        setup_ids: list[str],
        task_group: Optional[TaskGroup],
        **kwargs,
    ) -> Task:
        """Launch the workload in the environment."""
        if task_group is None:
            task_group = asyncio.TaskGroup()
        launch_id = get_uuid()
        self.setup_ids[launch_id] = setup_ids  # Made available for retries
        self._get_launch_ready_event(launch_id)  # Preload this

        async def launch_helper():
            # Let the monitor know it can proceed.
            launch_done_event = self.__get_launch_done_event(launch_id)
            if not self.__any_events_set_from_dict(
                setup_ids, self.__teardown_started_events
            ):
                try:
                    await self.launch_types[launcher_type](self, launch_id, **kwargs)
                except Exception as e:
                    self.__get_launch_failed_event(launch_id).set()
                    raise e
                finally:
                    launch_done_event.set()
            launch_done_event.set()

        task = task_group.create_task(launch_helper())
        task.launch_id = launch_id  # type: ignore[attr-defined]
        return task

    async def _inject_event_to_trigger_retry_when_testing(
        self: Self, retry_handler: Optional["RetryHandler"], scenario: Optional[str]
    ):
        """Can be placed in mainline code when a RetryHandler is installed to trigger the simulation
        of a named scenario that will trigger a retry in the given RetryHandler by placing the simulated
        event on its event queue.

        This will only install 1 event per launch id (attached to the RetryHandler) so can be called
        more than once within an Environment, including the retries that are triggered.

        We are considered to be testing when the GBTEST_SIMULATE_FAILURE_SCENARIO env var is set (only be build tests?)

        Args:
            retry_handler (RetryHandler): may be None
            scenario (str): a scenario from resilience.simulate.py
        """
        if (
            retry_handler
            and scenario
            and GBSERVER_ENABLE_STEP_RETRY
            and is_failure_simulated()
        ):
            from gbserver.resilience.simulate import inject_simulated_failure

            await asyncio.sleep(
                5
            )  # Let the build run for a bit and generate some of its own events.
            await inject_simulated_failure(retry_handler, scenario)

    def __cleanup_gc(self, launch_id: str):
        self.__launch_done_events.pop(launch_id, None)
        # can't gc cleanup_done_events since teardown waits on them

    def cleanup(
        self: Self,
        launch_type: Optional[str] = None,
        launch_id: Optional[str] = None,
        setup_ids: list[str] = [],
        tg: Optional[TaskGroup] = None,
        **kwargs,
    ) -> Optional[Task]:
        """Cleanup the workload from the environment."""
        # self._monitoring_cleanup(launch_id=launch_id)
        assert launch_type
        assert launch_id

        if tg is None:
            tg = asyncio.TaskGroup()

        if launch_type not in self.cleanup_types:
            return None

        async def cleanup_helper():
            launch_event = self.__get_launch_done_event(launch_id)
            logger.debug("Sync waiting on launch done")
            await launch_event.wait()
            logger.debug("Sync got launch done")
            cleanup_event = self.__get_cleanup_done_event(launch_id)
            if not self.__any_events_set_from_dict(
                setup_ids, self.__teardown_started_events
            ):
                try:
                    await self.cleanup_types[launch_type](
                        self, launch_id=launch_id, **kwargs
                    )
                except Exception as e:
                    raise e
                finally:
                    cleanup_event.set()
            cleanup_event.set()
            # self.__cleanup_gc(launch_id)

        task = tg.create_task(cleanup_helper())
        return task

    def _monitoring_cleanup(self: Self, launch_id: str):
        """
        NOTE: this seems deprecated as it is never called from anywhere.
        Make sure the monitoring task is cleaned up.
        """
        logger.info(
            "Setting the stop_event to cleanup monitoring for launch_id %s", launch_id
        )
        stop_event = self._get_launch_stopped_event(launch_id=launch_id)
        stop_event.set()

    def monitor(
        self: Self,
        type: str,
        launch_id: str,
        task_group: TaskGroup,
        event_q: Optional[Queue] = None,
        entityrun_metadata: Optional[EntityRunMetadata] = None,
        build_id: str = "",
        **kwargs,
    ) -> Task:
        """Monitor the launch/workload status and logs.
        Actual monitoring for the launch will not proceed until the _release_monitors()
        has been called by the launch associated launch_id.
        """
        asyncio_runner = task_group if task_group is not None else asyncio
        launch_ready_event = self._get_launch_ready_event(launch_id)
        launch_failed_event = self.__get_launch_failed_event(launch_id)

        async def monitor_helper():
            logger.debug("Sync waiting on launch ready or failed event")
            await self.__wait_any_events([launch_ready_event, launch_failed_event])
            logger.debug("Sync got on launch ready or failed event")
            if launch_failed_event.is_set():
                logger.warning(
                    "Launch failed, aborting monitor for launch_id %s", launch_id
                )
                return  # No need to monitor it.
            await self.monitor_types[type](
                self,
                launch_id=launch_id,
                event_q=event_q,
                entityrun_metadata=entityrun_metadata,
                build_id=build_id,
                **kwargs,
            )

        return asyncio_runner.create_task(monitor_helper())

    def setup(self: Self, type: str, task_group: TaskGroup, **kwargs) -> Task:
        """Setup the environment to run a workload."""
        setup_id = get_uuid()

        async def setup_helper() -> Dict:
            setup_event = self.__get_setup_done_event(setup_id)
            teardown_event = self.__get_teardown_started_event(setup_id)
            try:
                if not teardown_event.is_set():
                    config = await self.setup_types[type](
                        self, setup_id=setup_id, **kwargs
                    )
                else:
                    config = {}
                return config
            finally:
                setup_event.set()  # always unblock teardown, even if setup raised

        task = task_group.create_task(setup_helper())
        task.setup_id = setup_id  # type: ignore[attr-defined]
        return task

    async def __wait_all_events(self, event_dict: dict[str, Event]) -> None:
        task_list = []
        for event in event_dict.values():
            task = asyncio.create_task(event.wait())
            task_list.append(task)
        await asyncio.gather(*task_list)

    async def __wait_any_events(self, event_list: list[Event]) -> None:
        while True:
            if self.__any_events_set_from_list(event_list):
                return
            await asyncio.sleep(1)

    def __any_events_set_from_list(self, event_list: list[Event]) -> bool:
        for event in event_list:
            if event and event.is_set():
                return True
        return False

    def __any_events_set_from_dict(
        self, ids: list[str], event_dict: dict[str, Event]
    ) -> bool:
        event_list = [event_dict[id] for id in ids if id in event_dict]
        return self.__any_events_set_from_list(event_list)

    #        for id in ids:
    #            event = event_dict.get(id,None)
    #            if event and event.is_set():
    #                return True
    #        return False

    def __teardown_gc(self, setup_id: str):
        self.__cleanup_done_events.pop(setup_id, None)
        self.__teardown_started_events.pop(setup_id, None)

    def teardown(self: Self, type: str, setup_id: str, **kwargs) -> Task:
        """Teardown the setup from the environment."""

        async def teardown_helper():
            logger.debug("Sync waiting on setup done")
            await self.__get_setup_done_event(setup_id).wait()
            logger.debug("Sync got on setup done")
            teardown_event = self.__get_teardown_started_event(setup_id)
            teardown_event.set()  # Lock out future launches or cleanups
            logger.debug("Sync waiting on launch done")
            await self.__wait_all_events(self.__launch_done_events)
            logger.debug("Sync got launch done")
            logger.debug("Sync waiting on cleanup done")
            await self.__wait_all_events(self.__cleanup_done_events)
            logger.debug("Sync got cleanup done")
            await asyncio.create_task(
                self.teardown_types[type](self, setup_id=setup_id, **kwargs)
            )
            # self.__teardown_gc(setup_id)

        return asyncio.create_task(teardown_helper())

    def _get_storeconfig(
        self: Self, uri: URI, raise_exceptions: bool = False
    ) -> Tuple[Optional[Assetstore], Optional[AssetStoreEnvironmentConfig]]:
        """Returns the asset store and config that matches the longest prefix of the URI."""
        longest_match = 0
        t1 = None
        t2 = None
        for assetstore, assetstoreenv_config in self.supported_assetstores.items():
            curr_len = assetstore.can_handle_len(uri)
            if curr_len > longest_match:
                longest_match = curr_len
                t1 = assetstore
                t2 = assetstoreenv_config
        if t1 is None or t2 is None:
            if raise_exceptions:
                uristr = URI.get_uristr(uri)
                raise ValueError(
                    f"failed to get an assetstore and config for the URI {uristr}."
                    + f" Check {ENVIRONMENT_FILENAME} file(s) in your space repo."
                )
        return t1, t2

    def pullasset(
        self: Self, task_group: TaskGroup, uri: URI, binding: Optional[Any] = None
    ) -> Task[Tuple[Dict, Optional[BuildTargetStepConfig]]]:
        """Pull an asset and make it available in the environment for the workload."""

        async def pullasset_as_binding() -> (
            Tuple[Dict, Optional[BuildTargetStepConfig]]
        ):
            uristr = URI.get_uristr(uri)
            if uristr in self.asset_bindings:
                return self.asset_bindings[uristr], None
            await Environment._thread_local.asset_events[uristr].wait()
            assetstore, assetstoreenv_config = self._get_storeconfig(
                uri, raise_exceptions=True
            )
            assert assetstore is not None
            assert assetstoreenv_config is not None
            assetstore_type = assetstore.type.lower()
            try:
                storeload_config = (
                    assetstoreenv_config.load[0]
                    if (
                        assetstoreenv_config.load is not None
                        and len(assetstoreenv_config.load) > 0
                    )
                    else None
                )
                if assetstore_type not in self.pullasset_types:
                    la_types = self.pullasset_types.keys()
                    raise ValueError(
                        f"the assetstore type '{assetstore_type}' is not supported"
                        + f", supported ones are '{la_types}'"
                    )
                local_binding, targetrun_config = await self.pullasset_types[
                    assetstore_type
                ](
                    self,
                    uri=uri,
                    binding=binding,
                    storeload_config=storeload_config,
                    assetstore=assetstore,
                    secrets=assetstore.get_secrets(),
                )
                self.asset_bindings[uristr] = local_binding
                return local_binding, targetrun_config
            except Exception as e:
                raise RuntimeError(
                    f"failed to load the asset uri '{uristr}' from the assetstore of type {assetstore.type}"
                ) from e

        asyncio_runner = task_group if task_group is not None else asyncio
        return asyncio_runner.create_task(pullasset_as_binding())

    def pushasset(
        self: Self,
        task_group: TaskGroup,
        binding: Any,
        uristr: str,
        binding_id: Optional[str] = None,
        additional_targetsteps_queue: Optional[Queue] = None,
        run_metadata: Optional[EntityRunMetadata] = None,
        output_config: Optional[BuildTargetOutputConfig] = None,
    ) -> Task[URI]:
        """Push an asset from the environment to an asset store."""
        uristr = URI.get_uristr(uristr)
        config = {BINDING_KEY: binding, FULL_CONFIG_RUN_METADATA_KEY: run_metadata}
        space_variables = URI.get_space_config()
        if space_variables is not None:
            config = config | space_variables
        uri = URI.get_uri(fill_template(uristr, config, strict=True))
        assetstore, assetstoreenv_config = self._get_storeconfig(
            uri=uri, raise_exceptions=True
        )
        assert assetstore is not None
        assert assetstoreenv_config is not None
        assetstore_type = assetstore.type.lower()
        uristr = URI.get_uristr(uri)
        assert run_metadata is not None, "run_metadata is None"
        Environment._thread_local.asset_events[uristr] = Event()
        event = BuildEvent(
            run_metadata=run_metadata,
            type=BuildEventType.ARTIFACT_EVENT,
            payload=CreatedArtifactEventPayload(uri=uristr, binding_id=binding_id),
        )

        async def pushasset_as_uri() -> URI:
            storepush_config = (
                assetstoreenv_config.push[0]
                if (
                    assetstoreenv_config.push is not None
                    and len(assetstoreenv_config.push) > 0
                )
                else None
            )
            result = await self.pushasset_types[assetstore_type](
                self,
                binding=binding,
                binding_id=binding_id,
                storepush_config=storepush_config,
                uri=uri,
                assetstore=assetstore,
                secrets=assetstore.get_secrets(),
                run_metadata=run_metadata,
                output_config=output_config,
            )
            if isinstance(result, BuildTargetStepConfig):
                assert (
                    additional_targetsteps_queue is not None
                ), "additional_targetsteps_queue is None"
                await additional_targetsteps_queue.put(result)
            else:
                Environment._thread_local.asset_events[uristr].set()
            await self.event_q.put(event)
            self.asset_bindings[uristr] = {BINDING_KEY: binding}
            # For inline pushes (no separate push step), emit ARTIFACT_PUSHED_EVENT
            # so the artifact transitions from pending to success.
            if not isinstance(result, BuildTargetStepConfig):
                pushed_event = BuildEvent(
                    run_metadata=run_metadata,
                    type=BuildEventType.ARTIFACT_PUSHED_EVENT,
                    payload=ArtifactPushedEventPayload(
                        uri=uristr, binding_id=binding_id or ""
                    ),
                )
                await self.event_q.put(pushed_event)
            return uri

        asyncio_runner = task_group if task_group is not None else asyncio
        return asyncio_runner.create_task(pushasset_as_uri())

    def _get_fns_with_prefix(self: Self, prefix: str) -> dict[str, Callable]:
        """
        Returns all methods on this instance that
        have names starting with the given prefix.

        NOTE:
        - The dict keys will be the lowercased method name with the prefix removed.
        - The methods are unbound so you will need to pass in self explicitly.

        Returns: a dict of name -> function/method
        """
        logger.info("Environment._get_fns_with_prefix start")
        functions = inspect.getmembers(self.__class__, inspect.isfunction)
        name_to_fn: dict[str, Callable] = {}
        my_name = "_get_fns_with_prefix"
        for fn_name, fn in functions:
            if not fn_name.startswith(prefix) or fn_name == my_name:
                continue
            key = fn_name.removeprefix(prefix).lower()
            name_to_fn[key] = fn
        logger.info("Environment._get_fns_with_prefix end name_to_fn: %s", name_to_fn)
        return name_to_fn

    @classmethod
    def _load_environment_types(cls: Type[Self]) -> None:
        package_dir = os.path.dirname(__file__)

        for filename in os.listdir(package_dir):
            if (
                filename.endswith(".py")
                and filename != "__init__.py"
                and filename != "environment_sync.py"
                and filename != os.path.basename(__file__)
            ):
                environment_type_modulename = filename[:-3]
                environment_type_name = environment_type_modulename.capitalize()
                try:
                    module = importlib.import_module(
                        f".{environment_type_modulename}",
                        package="gbserver.environment",
                    )
                    if hasattr(module, environment_type_name):
                        handler_class = getattr(module, environment_type_name)
                        if isinstance(handler_class, type) and issubclass(
                            handler_class, cls
                        ):
                            cls.environment_types[environment_type_name.lower()] = (
                                handler_class
                            )
                            cls.environment_types[environment_type_name] = handler_class
                        else:
                            logger.warning(
                                "Ignoring %s since it is not a subclass of Environment class",
                                environment_type_name,
                            )
                    else:
                        logger.warning(
                            "Module %s does not contain expected environment type class %s",
                            environment_type_modulename,
                            environment_type_name,
                        )
                except ImportError as e:
                    logger.debug(
                        "Optional environment module %s not available: %s",
                        environment_type_name,
                        e,
                    )
                except Exception as e:
                    logger.error(
                        "Error loading Environment type from %s: %s",
                        environment_type_name,
                        e,
                    )

    async def _read_stream_and_create_event_all(
        self: Self,
        stream: Optional[StreamReader],
        stream_name: str,
        event_q: Queue,
        event_parser_configs: List[EventLogLineParserConfig],
        entityrun_metadata: EntityRunMetadata,
    ):
        if stream is None:
            return
        while True:
            try:
                linebytes = await stream.readline()
                if not linebytes:
                    break
                line = linebytes.decode("utf-8")
                await self.get_events_from_log_line(
                    log_line=line,
                    event_configs=event_parser_configs,
                    event_q=event_q,
                    entityrun_metadata=entityrun_metadata,
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "failed to read and process events from the stream %s: %s",
                    stream_name,
                    e,
                )
                break

    async def _monitor_logs_of_async_subprocess_all(
        self,
        process: Process,
        event_q: Queue,
        event_configs: List[EventLogLineParserConfig],
        entityrun_metadata: EntityRunMetadata,
    ):
        stdout_task = asyncio.create_task(
            self._read_stream_and_create_event_all(
                process.stdout, "stdout", event_q, event_configs, entityrun_metadata
            )
        )
        stderr_task = asyncio.create_task(
            self._read_stream_and_create_event_all(
                process.stderr, "stderr", event_q, event_configs, entityrun_metadata
            )
        )
        await asyncio.gather(stdout_task, stderr_task)

    @staticmethod
    async def get_events_from_log_line(
        log_line: str,
        event_configs: List[EventLogLineParserConfig],
        event_q: Optional[Queue] = None,
        entityrun_metadata: Optional[EntityRunMetadata] = None,
        messenger: Optional[MessagingBase] = None,
        line_num: Optional[int] = None,
    ) -> Union[List[BuildEvent], List[Dict]]:
        """
        Parse events out of the log line according to the given event_configs.
        By default we add the events into the provided event_q.
        We also return the same list of build events that we parsed.
        If a 'messenger' is provided, then it will be used to send
        the events instead of the 'event_q' event queue. Also
        the events will be plain dicts instead of BuildEvent
        to allow sending over the network.
        """
        logger.debug("Running get_events_from_log_line for line: %s", log_line)
        if event_q is None:
            event_q = asyncio.Queue()
        if entityrun_metadata is None:
            entityrun_metadata = EntityRunMetadata()
        build_events: Union[List[BuildEvent], List[Dict]] = []
        for event_config in event_configs:
            try:
                assert (
                    event_config._line_pattern is not None
                ), "event_config._line_pattern is None"
                match = event_config._line_pattern.search(log_line)
                if match is None:
                    continue
                logger.info(
                    "Running get_events_from_log_line for line%s: %s",
                    f" {str(line_num)}" if line_num is not None else "",
                    log_line,
                )
                log_line = match[0]
                event_data: Dict[str, Any] = {}
                event_data["data"] = {}
                if event_config.is_json:
                    event_data["data"] = json.loads(log_line)
                for event_field in event_config.event_fields:
                    if event_field._field_pattern is not None:
                        field_match = event_field._field_pattern.search(log_line)
                        if field_match is not None:
                            value = field_match.group(0)
                            if event_field.is_json:
                                try:
                                    value = json.loads(value)
                                except Exception as e:
                                    raise ValueError(
                                        f"failed to decode the event field '{event_field}' value '{value}' as JSON"
                                    ) from e
                            if event_field.is_data:
                                event_data["data"][event_field.field_name] = value
                            else:
                                event_data[event_field.field_name] = value
                        else:
                            logger.warning(
                                "failed to match event_field._field_pattern: %s against %s",
                                event_field,
                                log_line,
                            )
                for event_field in event_config.event_fields:
                    if event_field.field_value_template is not None:
                        value = ""
                        try:
                            template_data = {"fields": event_data}
                            value = fill_template(
                                event_field.field_value_template,
                                template_data,
                                strict=True,
                            )
                        except Exception as e:
                            raise ValueError(
                                f"failed to fill the {event_field} template with {template_data}"
                            ) from e
                        if event_field.is_json:
                            try:
                                value = json.loads(value)
                            except Exception as e:
                                raise ValueError(
                                    f"failed to decode the event field '{event_field}' value '{value}' as JSON"
                                ) from e
                        if event_field.is_data:
                            event_data["data"][event_field.field_name] = value
                        else:
                            event_data[event_field.field_name] = value
                if messenger is not None:

                    def _make_event_id(
                        event_type: str,
                        event_data: Dict[str, Any],
                        line_num: Optional[int] = None,
                    ) -> str:
                        """Generate an idempotency key for the event"""
                        if line_num is None:
                            line_num = -1
                        canonical_data = json.dumps(
                            event_data, sort_keys=True, separators=(",", ":")
                        )
                        seed = f"{event_type}|{canonical_data}|{line_num}"
                        return hashlib.sha256(seed.encode("utf-8")).hexdigest()

                    event_id = _make_event_id(
                        event_config.event_type, event_data, line_num
                    )
                    build_event = {
                        "type": event_config.event_type,
                        "event_id": event_id,
                        "data": event_data,
                    }
                    build_events.append(build_event)  # type: ignore[arg-type]
                    logger.info(
                        "Publishing build event : %s", json.dumps(build_event, indent=2)
                    )
                    await messenger.publish(
                        payload=build_event, suffix=event_config.event_type
                    )
                else:
                    logger.info("JSON Log Event : %s", event_data)
                    build_event_type = BuildEventType(event_config.event_type.lower())
                    build_event = BuildEvent(  # type: ignore[assignment]
                        run_metadata=entityrun_metadata,
                        type=build_event_type,
                        payload=EventPayload.payload_parser(
                            event_type=build_event_type,
                            data=event_data,
                        ),
                    )
                    build_events.append(build_event)  # type: ignore[arg-type]
                    await event_q.put(build_event)
            except Exception as e:
                logger.error("%s", traceback.format_exc())
                logger.error("for log line '%s' got error: %s", log_line, e)
                continue
        return build_events

    def __get_event(self, id: str, event_dict: dict) -> asyncio.Event:
        event = event_dict.get(id)
        if event is None:
            event = asyncio.Event()
            event_dict[id] = event
        return event

    def __get_setup_done_event(self, setup_id: str) -> asyncio.Event:
        return self.__get_event(setup_id, self.__setup_done_events)

    def __get_teardown_started_event(self, setup_id: str) -> asyncio.Event:
        return self.__get_event(setup_id, self.__teardown_started_events)

    def _get_launch_ready_event(self, launch_id: str) -> asyncio.Event:
        """
        Gets an event for the given launch id that should be set once the launch is ready to be monitored.
        This must be set by the underlying launch_XYZ() method in order for the monitor methods to proceed.
        Once set, the corresponding monitors for the given launch id will be called.

        NOTE: This is deprecated in favor of _release_monitors()
        """
        return self.__get_event(launch_id, self.__launch_ready_events)

    def _release_monitors(self, launch_id: str):
        """
        Set the state to allow the waiting monitors to proceed to begin monitoring the given launch.
        Generally called from a launch_XYZ implementation once the monitor can start monitoring it.
        """
        self._get_launch_ready_event(launch_id).set()

    def __get_launch_done_event(self, launch_id: str) -> asyncio.Event:
        return self.__get_event(launch_id, self.__launch_done_events)

    def __get_launch_failed_event(self, launch_id: str) -> asyncio.Event:
        return self.__get_event(launch_id, self.__launch_failed_events)

    def __get_cleanup_done_event(self, launch_id: str) -> asyncio.Event:
        return self.__get_event(launch_id, self.__cleanup_done_events)

    def _get_launch_stopped_event(self, launch_id: str) -> asyncio.Event:
        """Get an event that should be used by monitors or other mechanism to indicate
        when the launch with the given launch id has completed.
        It is usually set by a monitor that detects the launch is completed.
        This will notify others that the launch is completed
        and other monitors or anything else associated with the launch also should stop.
        """
        return self.__get_event(launch_id, self.__launch_stopped_events)
