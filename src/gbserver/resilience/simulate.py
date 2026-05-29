"""
Local debugging helper for RetryHandler.

Inject a synthetic failure event into a RetryHandler's wrapper queue to trigger
retry_workload without needing a real Kubernetes failure.

Set the GBTEST_SIMULATE_FAILURE_SCENARIO environment variable (or pass the scenario
name directly) to inject a failure once the RetryHandler is ready.

Supported scenario names:
  unhealthy_insufficient_pods   – triggers UnhealthyInsufficientPodsRetryStrategy
  pod_eviction                  – triggers PodEvictionRetryStrategy
  nccl_error                    – triggers NCCLErrorRetryStrategy
  file_not_found                – triggers FileNotFoundRetryStrategy
"""

import json
from typing import TYPE_CHECKING, Optional

from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventMessagePayload,
    BuildEventType,
    EntityRunMetadata,
)
from gbserver.utils.logger import get_logger

if TYPE_CHECKING:
    from gbserver.resilience.retry_handler import RetryHandler

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Per-launch injection tracking – ensures each launch is only injected once.
# Without this, monitor_appwrapper_only would re-inject on every retry cycle
# (retry reuses the same launch_id).
# ---------------------------------------------------------------------------

_injected_launch_ids: set[str] = set()

# ---------------------------------------------------------------------------
# Canned event payloads – one per strategy type
# ---------------------------------------------------------------------------

_PAYLOADS: dict[str, str] = {
    "lsf_transient_error": "Cannot open your job file",
    "unhealthy_insufficient_pods": json.dumps(
        {
            "state": "Failed",
            "previous_state": "Running",
            "events": [
                {
                    "object_type": "AppWrapper",
                    "reason": "Unhealthy",
                    "message": "insufficient pods: 0/1 nodes are available",
                    "object_name": "sim-appwrapper",
                }
            ],
            "pod_placement": {},
        }
    ),
    "pod_eviction": json.dumps(
        {
            "state": "Failed",
            "previous_state": "Running",
            "events": [
                {
                    "object_type": "AppWrapper",
                    "reason": "Unhealthy",
                    "message": "pod evicted due to resource pressure",
                    "object_name": "sim-appwrapper",
                },
                {
                    "object_type": "Pod",
                    "reason": "Evicted",
                    "message": "The node was low on resource: memory",
                    "object_name": "sim-pod-0",
                },
            ],
            "pod_placement": {"sim-pod-0": "sim-node-1"},
        }
    ),
    # NOTE: The "message" string must match one of NCCLErrorRetryStrategy's
    # regex patterns (see strategies/nccl_error.py NCCL_ERROR_PATTERNS), or
    # the strategy votes "no retry" and the synthetic event passes through
    # without firing retry_workload. The strategy scans `event.payload.msg`
    # as a flat string — JSON structure doesn't matter, only that the regex
    # matches somewhere in the JSON-serialized message body.
    "nccl_error": json.dumps(
        {
            "state": "Failed",
            "previous_state": "Running",
            "events": [
                {
                    "object_type": "AppWrapper",
                    "reason": "Unhealthy",
                    "message": "RuntimeError: NCCL Error 3: internal error",
                    "object_name": "sim-appwrapper",
                }
            ],
            "pod_placement": {},
        }
    ),
    "file_not_found": json.dumps(
        {
            "state": "Failed",
            "previous_state": "Running",
            "events": [
                {
                    "object_type": "AppWrapper",
                    "reason": "Unhealthy",
                    "message": "FileNotFoundError: [Errno 2] No such file or directory",
                    "object_name": "sim-appwrapper",
                }
            ],
            "pod_placement": {},
        }
    ),
}


def _build_event(scenario: str, build_id: str) -> Optional[BuildEvent]:
    """Build a synthetic BuildEvent for the given scenario name."""
    payload_msg = _PAYLOADS.get(scenario.strip().lower())
    if payload_msg is None:
        logger.warning(
            "[simulate] Unknown scenario %r. Valid values: %s",
            scenario,
            ", ".join(_PAYLOADS),
        )
        return None
    metadata = EntityRunMetadata(build_id=build_id)
    payload = BuildEventMessagePayload(level="ERROR", msg=payload_msg)
    return BuildEvent(
        run_metadata=metadata,
        type=BuildEventType.MESSAGE_EVENT,
        payload=payload,
        source="simulate",
    )


async def inject_simulated_failure(
    handler: "RetryHandler",
    scenario: Optional[str],
) -> None:
    """
    Inject one synthetic failure event into *handler*'s wrapper queue.

    Args:
        handler:  The RetryHandler whose wrapper_queue will receive the event.
        scenario: One of the keys in _PAYLOADS, or None/empty to skip injection.
    """
    if not scenario:
        return

    if handler.launch_id in _injected_launch_ids:
        logger.info(
            "[simulate] Skipping injection for launch_id %s — already injected for this launch",
            handler.launch_id,
        )
        return

    logger.warning(
        "[simulate] Injecting synthetic '%s' failure event for launch_id %s",
        scenario,
        handler.launch_id,
    )
    event = _build_event(scenario, handler.build_id)
    if event is None:
        return
    await handler.wrapper_queue.put(event)
    _injected_launch_ids.add(handler.launch_id)
