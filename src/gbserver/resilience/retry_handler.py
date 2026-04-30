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
RetryHandler for managing workload retry strategies across all environments.

This module provides environment-agnostic retry orchestration for failed workloads,
separating retry logic from both monitoring and execution concerns. It uses a
wrapper queue pattern to intercept BuildEvents emitted by monitors and determine
if retries should be attempted.

Key Design Principles:
- Environment-agnostic: Works with K8s, LSF, local environments, etc.
- Monitor-agnostic: Processes BuildEvents from any monitor type
- Strategy-based: Pluggable retry strategies for different failure patterns
- Decoupled: Delegates actual retry to environment.retry_workload()

The RetryHandler analyzes BuildEvents (which may contain environment-specific
event data like Kubernetes events, LSF job logs, etc.) to detect retryable
conditions, then delegates to the appropriate environment for retry execution.
"""

import asyncio
import json
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Self, Set

from gbserver.types.buildevent import BuildEvent, BuildLogLevel, create_message_event
from gbserver.types.errors import WorkloadFailedException
from gbserver.utils.logger import get_logger

if TYPE_CHECKING:
    from gbserver.environment.environment import Environment
    from gbserver.resilience.node_health_tracker import NodeHealthTracker
    from gbserver.types.buildevent import EntityRunMetadata

logger = get_logger(__name__)


def build_retry_strategies_from_config(
    config: Optional[List[dict]] = None,
    object_types: Optional[List[str]] = None,
) -> List["RetryStrategy"]:
    """
    Build retry strategies from configuration.

    Args:
        config: List of strategy configurations. Each config dict should have:
                - 'type': Strategy type name (e.g., 'UnhealthyInsufficientPods', 'PodEviction')
                - Additional strategy-specific parameters
                If None, returns all available strategies with defaults.
        object_types: Default object types to monitor (e.g., ["AppWrapper", "Job"])
                     Used if not specified in individual strategy configs.

    Returns:
        List[RetryStrategy]: List of configured retry strategies

    Example config:
        [
            {
                "type": "UnhealthyInsufficientPods",
                "object_types": ["AppWrapper", "Job"]
            },
            {
                "type": "PodEviction",
                "object_types": ["AppWrapper"],
                "avoid_eviction_nodes": False
            }
        ]
    """
    from gbserver.resilience.strategies import (
        AsperaRetryStrategy,
        FileNotFoundRetryStrategy,
        LsfTransientErrorRetryStrategy,
        NCCLErrorRetryStrategy,
        PodEvictionRetryStrategy,
        UnhealthyInsufficientPodsRetryStrategy,
    )

    # Default object types if not specified
    default_object_types = object_types or ["AppWrapper"]

    # If no config provided, return all available strategies with defaults
    if config is None:
        logger.info("No retry strategy config provided, using all available strategies")
        return [
            UnhealthyInsufficientPodsRetryStrategy(object_types=default_object_types),
            PodEvictionRetryStrategy(
                object_types=default_object_types,
                avoid_eviction_nodes=False,
            ),
            NCCLErrorRetryStrategy(),
            FileNotFoundRetryStrategy(),
        ]

    # Build strategies from config
    strategies = []
    strategy_map = {
        "UnhealthyInsufficientPods": UnhealthyInsufficientPodsRetryStrategy,
        "PodEviction": PodEvictionRetryStrategy,
        "NCCLError": NCCLErrorRetryStrategy,
        "FileNotFound": FileNotFoundRetryStrategy,
        "LsfTransientError": LsfTransientErrorRetryStrategy,
        "AsperaFailure": AsperaRetryStrategy,
    }

    for strategy_config in config:
        strategy_type = strategy_config.get("type")
        if not strategy_type:
            logger.warning("Strategy config missing 'type' field, skipping: %s", strategy_config)
            continue

        strategy_class = strategy_map.get(strategy_type)
        if not strategy_class:
            logger.warning("Unknown strategy type '%s', skipping", strategy_type)
            continue

        # Extract strategy-specific parameters
        params = {k: v for k, v in strategy_config.items() if k != "type"}

        # Check if this strategy accepts object_types parameter
        # Use the class attribute instead of hardcoded set
        if strategy_class.accepts_object_types:  # type: ignore[attr-defined]
            # Use default object types if not specified
            if "object_types" not in params:
                params["object_types"] = default_object_types
        else:
            # Remove object_types if present for strategies that don't accept it
            params.pop("object_types", None)

        try:
            strategy = strategy_class(**params)
            strategies.append(strategy)
            logger.info(
                "Loaded retry strategy: %s with params: %s",
                strategy_type,
                params,
            )
        except Exception as e:
            logger.error(
                "Failed to create strategy %s with params %s: %s",
                strategy_type,
                params,
                e,
            )

    if not strategies:
        logger.warning("No valid strategies configured, using default")
        strategies = [UnhealthyInsufficientPodsRetryStrategy(object_types=default_object_types)]

    return strategies


class RetryStrategy(ABC):
    """
    Abstract base class for retry strategies.

    Subclasses must implement should_retry() to determine if a retry
    should be attempted based on the failure conditions.

    Class Attributes:
        accepts_object_types: Whether this strategy accepts object_types parameter.
                            Defaults to True. Override to False for strategies that
                            don't filter by Kubernetes object type.
    """

    # Whether this strategy accepts object_types parameter in __init__
    accepts_object_types: bool = True

    @abstractmethod
    def should_retry(
        self: Self,
        event: BuildEvent,
    ) -> bool:
        """
        Determine if a retry should be attempted based on the event.

        The retry count / max-retries limit is enforced by RetryHandler before
        calling this method, so implementations only need to inspect the event.

        Args:
            event: BuildEvent from the monitor

        Returns:
            bool: True if retry should be attempted
        """
        raise NotImplementedError("Subclasses must implement should_retry()")

    def extract_nodes_to_avoid(
        self: Self,
        _event: BuildEvent,
    ) -> Set[str]:
        """
        Extract nodes that should be avoided in the retry.

        Args:
            _event: BuildEvent from the monitor (unused in base implementation)

        Returns:
            Set[str]: Set of node names to avoid
        """
        return set()

    def get_retry_delay(
        self: Self,
        retry_count: int,
    ) -> float:
        """
        Return delay in seconds before retrying.

        Strategies can override this to implement backoff behavior.
        For example, quota exhaustion benefits from exponential backoff
        while node-specific failures (mount issues) can retry immediately.

        Args:
            retry_count: Current retry count (0-based)

        Returns:
            float: Delay in seconds before retry. Default: 0.0 (immediate).
        """
        return 0.0


class RetryHandler:
    """
    Manages retry orchestration for failed workloads using a wrapper queue pattern.

    This class sits between the monitor and the downstream event queue, intercepting
    events to determine if retries should be attempted. It uses pluggable retry
    strategies to determine when and how to retry.

    The flow is:
    Monitor → wrapper_queue → RetryHandler →
      - If retryable: Trigger retry via environment.retry_workload(), suppress event
      - If not retryable: Forward to downstream_queue

    Parameters
    ----------
    launch_id : str
        Unique identifier for the launch
    downstream_queue : asyncio.Queue
        The real event queue where non-retryable events should be forwarded
    environment : Environment
        The environment instance that will handle the retry
    max_retries : int
        Maximum number of retry attempts (default: 3)
    strategies : List[RetryStrategy]
        List of retry strategies to evaluate (default: UnhealthyInsufficientPodsRetryStrategy)
    """

    def __init__(
        self: Self,
        launch_id: str,
        downstream_queue: asyncio.Queue,
        environment: "Environment",
        max_retries: int = 3,
        strategies: Optional[List[RetryStrategy]] = None,
        node_health_tracker: Optional["NodeHealthTracker"] = None,
        build_id: Optional[str] = None,
        entityrun_metadata: Optional["EntityRunMetadata"] = None,
    ) -> None:
        self.launch_id = launch_id
        self.build_id = build_id or launch_id  # Use launch_id as fallback
        self.entityrun_metadata = entityrun_metadata
        self.downstream_queue = downstream_queue
        self.environment = environment
        self.max_retries = max_retries
        self.retry_count = 0
        self.nodes_to_avoid: Set[str] = set()

        # Wrapper queue that monitors will publish to
        self.wrapper_queue: asyncio.Queue = asyncio.Queue()

        # Default to UnhealthyInsufficientPodsRetryStrategy if no strategies provided
        if strategies is None or not strategies:
            from gbserver.resilience.strategies import (
                NCCLErrorRetryStrategy,
                UnhealthyInsufficientPodsRetryStrategy,
            )

            self.strategies: List[RetryStrategy] = [
                UnhealthyInsufficientPodsRetryStrategy(),
                NCCLErrorRetryStrategy(),
            ]
            if strategies is not None:  # Empty list was explicitly provided
                logger.warning(
                    "[RetryHandler launch_id %s] Empty strategies list provided, using default",
                    launch_id,
                )
        else:
            self.strategies = strategies

        # Node health tracker for observability
        self.node_health_tracker = node_health_tracker

        # Flag to stop the processor
        self.stop_processing = False

    def get_wrapper_queue(self: Self) -> asyncio.Queue:
        """
        Get the wrapper queue that monitors should publish to.

        Returns:
            asyncio.Queue: The wrapper queue for event interception
        """
        return self.wrapper_queue

    async def process_events(self: Self) -> None:
        """
        Process events from the wrapper queue.

        This method runs as a background task, processing events from monitors
        and determining if they should trigger retries or be forwarded downstream.
        """
        logger.info(
            "[RetryHandler launch_id %s] Started event processing",
            self.launch_id,
        )

        while not self.stop_processing:
            try:
                # Wait for events with a timeout to allow checking stop_processing
                event = await asyncio.wait_for(
                    self.wrapper_queue.get(),
                    timeout=1.0,
                )

                # Evaluate if this event should trigger a retry
                retry_triggered = await self._evaluate_and_retry(event)

                # Check if this is a terminal failure
                is_terminal_failure = self._is_terminal_failure_event(event)

                # Always forward the event downstream, but enrich with retry metadata
                if retry_triggered:
                    # Add retry metadata to the event payload
                    if event.payload and hasattr(event.payload, "data"):
                        if event.payload.data is None:
                            event.payload.data = {}
                        event.payload.data["retry_triggered"] = True
                        event.payload.data["retry_count"] = self.retry_count
                        event.payload.data["max_retries"] = self.max_retries
                        event.payload.data["nodes_to_avoid"] = list(self.nodes_to_avoid)
                    logger.info(
                        "[RetryHandler launch_id %s] Event triggered retry %d/%d, forwarding downstream with retry metadata",
                        self.launch_id,
                        self.retry_count,
                        self.max_retries,
                    )

                # Forward to downstream queue
                await self.downstream_queue.put(event)

                # If terminal failure and no retry was triggered, raise exception to stop the build
                if is_terminal_failure and not retry_triggered:
                    error_message = self._extract_failure_message(event)
                    logger.error(
                        "[RetryHandler launch_id %s] Terminal failure detected with no retry possible. Raising exception.",
                        self.launch_id,
                    )
                    raise WorkloadFailedException(error_message)

            except asyncio.TimeoutError:
                # No event received, continue loop to check stop_processing
                continue
            except WorkloadFailedException:
                # Re-raise WorkloadFailedException to stop the build
                raise
            except Exception as e:
                logger.error(
                    "[RetryHandler launch_id %s] Error processing event: %s",
                    self.launch_id,
                    e,
                )
                # Forward the event downstream to avoid silently dropping it
                # The event variable may not be set if the error occurred during get()
                if "event" in dir() and event is not None:
                    try:
                        await self.downstream_queue.put(event)
                    except Exception as forward_err:
                        logger.error(
                            "[RetryHandler launch_id %s] Failed to forward event after error: %s",
                            self.launch_id,
                            forward_err,
                        )

        logger.info(
            "[RetryHandler launch_id %s] Stopped event processing",
            self.launch_id,
        )

    async def _record_node_failures(
        self: Self,
        failed_nodes: Set[str],
        strategy_name: str,
        event: BuildEvent,
    ) -> None:
        """
        Record node failures in the health tracker.

        Args:
            failed_nodes: Set of node names that failed
            strategy_name: Name of the retry strategy that detected the failure
            event: The build event that triggered the retry
        """
        if not self.node_health_tracker:
            return

        for node_name in failed_nodes:
            try:
                should_alert = await self.node_health_tracker.record_failure(
                    node_name=node_name,
                    build_id=self.build_id,
                    launch_id=self.launch_id,
                    failure_type=strategy_name,
                    retry_count=self.retry_count,
                    metadata={
                        "strategy": strategy_name,
                        "event_type": (
                            event.type.value if hasattr(event.type, "value") else str(event.type)
                        ),
                    },
                    namespace=getattr(self.environment, "namespace", ""),
                    cluster=getattr(self.environment, "kube_context", ""),
                )
                if should_alert:
                    logger.error(
                        "[RetryHandler launch_id %s] ALERT: Node %s has "
                        "exceeded failure threshold!",
                        self.launch_id,
                        node_name,
                    )
            except Exception as e:
                logger.warning(
                    "[RetryHandler launch_id %s] Failed to record node "
                    "failure in health tracker: %s",
                    self.launch_id,
                    e,
                )

    async def _evaluate_and_retry(
        self: Self,
        event: BuildEvent,
    ) -> bool:
        """
        Evaluate if a retry should be attempted for this event and retry if indicated.

        Args:
            event: BuildEvent from the monitor

        Returns:
            bool: True if a retry was triggered, False otherwise
        """
        if self.retry_count >= self.max_retries:
            logger.debug(
                "[RetryHandler launch_id %s] Maximum retries reached, will not retry",
                self.launch_id,
            )
            return False

        # Check if any strategy recommends a retry
        for strategy in self.strategies:
            if strategy.should_retry(event=event):
                # Extract nodes to avoid from this strategy
                failed_nodes = strategy.extract_nodes_to_avoid(event)
                self.nodes_to_avoid.update(failed_nodes)

                # Record failures in health tracker
                await self._record_node_failures(failed_nodes, strategy.__class__.__name__, event)

                logger.warning(
                    "[RetryHandler launch_id %s] Strategy %s recommends retry %d/%d, "
                    "avoiding nodes: %s",
                    self.launch_id,
                    strategy.__class__.__name__,
                    self.retry_count + 1,
                    self.max_retries,
                    self.nodes_to_avoid,
                )

                # Apply backoff delay if the strategy requests it
                delay = strategy.get_retry_delay(self.retry_count)
                if delay > 0:
                    logger.info(
                        "[RetryHandler launch_id %s] Waiting %.1f seconds before retry "
                        "(backoff from %s)",
                        self.launch_id,
                        delay,
                        strategy.__class__.__name__,
                    )
                    await asyncio.sleep(delay)

                # Execute the retry
                try:
                    return await self._execute_retry()
                except Exception as e:
                    logger.error(
                        "[RetryHandler launch_id %s] Retry execution failed, forwarding event: %s",
                        self.launch_id,
                        e,
                    )
                    # Return False to indicate retry was not successful,
                    # so the event gets forwarded downstream
                    return False

        return False

    async def _execute_retry(self: Self) -> bool:
        """
        Execute the retry via environment.retry_workload().

        Returns:
            bool: True if retry was initiated successfully

        Raises:
            Exception: If the retry fails, the exception is propagated to the caller
        """
        self.retry_count += 1

        step_label = (
            self.entityrun_metadata.targetstep_uri
            if self.entityrun_metadata and self.entityrun_metadata.targetstep_uri
            else self.launch_id
        )
        message = (
            f"Retrying workload for {step_label} "
            f"(attempt {self.retry_count}/{self.max_retries})"
        )
        logger.info(message)
        retry_event = create_message_event(
            source="retry_handler",
            build_id=self.build_id,
            level=BuildLogLevel.INFO,
            message=message,
        )
        await self.downstream_queue.put(retry_event)

        await self.environment.retry_workload(
            launch_id=self.launch_id,
            nodes_to_avoid=list(self.nodes_to_avoid) if self.nodes_to_avoid else None,
        )
        logger.info(
            "[RetryHandler launch_id %s] Retry %d/%d completed successfully",
            self.launch_id,
            self.retry_count,
            self.max_retries,
        )
        return True

    def stop(self: Self) -> None:
        """Stop processing events."""
        self.stop_processing = True

    def _is_terminal_failure_event(self: Self, event: BuildEvent) -> bool:
        """
        Determine if an event represents a terminal workload failure.

        A terminal failure is one where the AppWrapper has entered a Failed state.
        This is detected by parsing the event message for state information.

        Args:
            event: BuildEvent from the monitor

        Returns:
            bool: True if this is a terminal failure event
        """
        if not event.payload or not hasattr(event.payload, "msg"):
            return False

        msg = event.payload.msg
        if not msg:
            return False

        # Parse JSON from markdown code blocks
        try:
            # Extract JSON from ```json ... ``` blocks
            json_match = re.search(r"```json\s*\n(.*?)\n```", msg, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                state = data.get("state", "")
                # Terminal states that should stop the build
                return state == "Failed" or state.startswith("Exception:")
        except (json.JSONDecodeError, AttributeError) as e:
            logger.debug(
                "[RetryHandler launch_id %s] Could not parse event message as JSON: %s",
                self.launch_id,
                e,
            )

        return False

    def _extract_failure_message(self: Self, event: BuildEvent) -> str:
        """
        Extract the failure error message from the event.

        Args:
            event: BuildEvent from the monitor

        Returns:
            str: Error message describing the failure
        """
        if not event.payload or not hasattr(event.payload, "msg"):
            return f"Workload failed for launch_id {self.launch_id}"

        msg = event.payload.msg
        if not msg:
            return f"Workload failed for launch_id {self.launch_id}"

        # Try to extract more context from JSON
        try:
            json_match = re.search(r"```json\s*\n(.*?)\n```", msg, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                appwrapper = data.get("appwrapper", "unknown")
                state = data.get("state", "Failed")
                return (
                    f"[RetryHandler launch_id {self.launch_id}] {appwrapper} is in a {state} state. "
                    + "Build will stop because of an appwrapper workload error. "
                    + "The `failed_pods` and `events` sections in the message above have more error details."
                )
        except (json.JSONDecodeError, AttributeError):
            pass

        return f"[RetryHandler launch_id {self.launch_id}] Workload failed. See event message for details."
