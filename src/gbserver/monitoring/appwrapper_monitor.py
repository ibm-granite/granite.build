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
Monitor AppWrappers in K8s clusters.
"""

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional, Self, Set

import aiohttp
from kubernetes_asyncio import client

from gbserver.environment.k8s import AtomicApiClient
from gbserver.monitoring.monitor_base import MonitorBase
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventType,
    EntityRunMetadata,
    EventPayload,
)
from gbserver.types.constants import (
    GBSERVER_API_FAILURE_TIMEOUT,
    GBSERVER_MONITORING_GRACE_PERIOD,
)
from gbserver.types.metrics import Metric, MetricMetadata, MetricName, MetricUnits
from gbserver.utils.logger import get_logger
from gbserver.utils.utils import get_utc_time

logger = get_logger(__name__)


# Timeout for Kubernetes API calls (in seconds)
API_CALL_TIMEOUT = 30

# "Normal" events that actually indicate problems
PROBLEMATIC_NORMAL_EVENTS = {
    # Pod lifecycle issues
    "Killing",  # Pod is being killed
    "Evicted",  # Pod was evicted (resource pressure, etc.)
    "BackOff",  # Container back-off restarting failed container
    "FailedKillPod",  # Failed to kill pod
    # Node issues
    "NodeNotReady",  # Node became not ready
    "NodeReady",  # Flood of these events after NodeNotReady could signal node instability/reboots
    "NodeRebooted",  # Node was rebooted
    "DeletingNode",  # Node is being deleted
    # Container/Health issues
    "Unhealthy",  # Liveness/readiness probe failed
    "ProbeWarning",  # Probe produced a warning
    "FailedPostStartHook",  # PostStart hook failed
    "FailedPreStopHook",  # PreStop hook failed
    # Resource issues
    "FailedCreatePodContainer",  # Failed to create pod sandbox
    "FailedSync",  # Failed to sync pod
    "InspectFailed",  # Failed to inspect pod
    # Volume issues
    "FailedAttachVolume",  # Failed to attach volume
    "FailedMount",  # Failed to mount volume (often Normal type!)
    "VolumeFailedRecycle",  # Failed to recycle volume
    # Scheduling issues (sometimes marked as Normal)
    "FailedScheduling",  # Pod failed to schedule
    "Preempted",  # Pod was preempted
    # Network issues
    "NetworkNotReady",  # Network not ready
    # Image issues (sometimes Normal)
    "InvalidImageName",  # Invalid image name
    "ErrImagePull",  # Error resolving, accessing, or pulling image from registry
    "ImagePullBackOff",  # A possibly transient version of ErrImagePull
    # Resource limits
    "OOMKilling",  # Container is being OOM killed
    "OOMKilled",  # Container is being OOM killed
}


# ---------------------- Kubernetes AppWrapper monitor ---------------------
class AppWrapperMonitor(MonitorBase):
    """
    - Watches an AppWrapper CR (workload.codeflare.dev/v1beta2) until its
      .status.state moves from Running to Succeeded|Failed, then fires stop_event.
    - Publishes a Build Event every time the AppWrapper `.status.state` field changes
      - The build event contains:
        - The status of the workloads associated with the appwrapper
        - The kubernetes events (all for appwrapper, abnormal for associated pods) issued since the last appwrapper state change
    Parameters
    ----------
    event_queue : asyncio.Queue
        Event queue to which Build Events are pushed.
    name, namespace : str
        Identify the AppWrapper (name + namespace).
    poll : float
        Seconds between status polls when no watch event is available.
    stop_event : asyncio.Event
        Shared cooperative-cancellation flag.
    event_queue: asyncio.Queue
        Queue where the monitor sends the build events issued when appwrapper status changes
    entityrun_metadata : EntityRunMetadata
        EntityRunMetadata object for the step delivered by the AppWrapper.
    """

    def __init__(
        self: Self,
        name: str,
        namespace: str,
        poll: float,
        kube_config: Optional[str] = None,
        kube_context: Optional[str] = None,
        ssl_verification: Optional[bool] = True,
        launch_id: str = "",
        entityrun_metadata: Optional[EntityRunMetadata] = None,
        event_queue: Optional[asyncio.Queue] = None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        super().__init__(
            launch_id=launch_id,
            entityrun_metadata=entityrun_metadata,
            event_queue=event_queue,
            stop_event=stop_event,
        )
        self.name = name
        self.ns = namespace
        self.poll = poll
        self.event_q = event_queue
        self._last_state: str = "NotStarted"
        self._seen_event_uids: Set[str] = set()
        self.kube_config = kube_config
        self.kube_context = kube_context
        self.ssl_verification = ssl_verification
        self.v1: Optional[client.CoreV1Api] = None
        self.custom_api: Optional[client.CustomObjectsApi] = None
        self.additional_appwrapper_state_info = {}  # type: ignore[var-annotated]
        self.latest_events = []  # type: ignore[var-annotated]
        self.failed_pods = {}  # type: ignore[var-annotated]
        self.launched_pods = {}  # type: ignore[var-annotated]
        self._api_failure_start_time: Optional[float] = None
        self._run_event = asyncio.Event()
        self._run_event.set()  # running by default; cleared by pause()

    def pause(self: Self) -> None:
        """
        Pause the monitor loop.  Clears _run_event so the monitor suspends at
        the top of its next iteration and skips all K8s API calls until unpause().
        """
        logger.info(
            "[AWMonitor launch_id %s] Pausing monitor for AppWrapper %s",
            self.launch_id,
            self.name,
        )
        self._run_event.clear()

    def unpause(self: Self) -> None:
        """
        Resume/unpausethe monitor state for a new workload (e.g., after retry)
        after a call to pause().

        This method should be called after a workload retry to ensure the monitor
        starts with clean state for the new AppWrapper instance, preventing
        contamination from the previous failed workload. For the resume to
        work as expected, the retried workload must have the same name (self.name).
        """
        logger.info(
            "[AWMonitor launch_id %s] Unpausing monitor state for new workload",
            self.launch_id,
        )
        self._last_state = "NotStarted"
        self._seen_event_uids = set()
        self.additional_appwrapper_state_info = {}
        self.latest_events = []
        self.failed_pods = {}
        self.launched_pods = {}
        self._api_failure_start_time = None
        self._run_event.set()

    def _is_api_failure_timeout(self: Self) -> bool:
        """Check if sustained API failures have exceeded the configured timeout."""
        if self._api_failure_start_time is None:
            return False
        elapsed = time.monotonic() - self._api_failure_start_time
        return elapsed > GBSERVER_API_FAILURE_TIMEOUT

    def _record_api_failure(
        self: Self, exc: Optional[Exception] = None
    ) -> None:
        """Record an API failure, setting the start time if this is the first in a streak."""
        if self._api_failure_start_time is None:
            self._api_failure_start_time = time.monotonic()
        elapsed = time.monotonic() - self._api_failure_start_time
        logger.warning(
            "[AWMonitor launch_id %s] API failure for AppWrapper %s "
            "(sustained for %.0fs / %ds timeout): %s",
            self.launch_id,
            self.name,
            elapsed,
            GBSERVER_API_FAILURE_TIMEOUT,
            exc if exc else "unknown",
        )

    async def _check_pod_liveness(self: Self) -> bool:
        """
        Check if any pods owned by this AppWrapper are still Running.

        Returns True if at least one pod is in Running phase, False otherwise.
        Used as a secondary liveness check before declaring fatal API failure —
        the core v1 Pod API may still be reachable even if the AppWrapper CRD
        API is not.
        """
        label_selector = f"workload.codeflare.dev/appwrapper={self.name}"
        assert self.v1 is not None, "CoreV1Api not initialized"
        try:
            pod_list = await asyncio.wait_for(
                self.v1.list_namespaced_pod(
                    namespace=self.ns, label_selector=label_selector
                ),
                timeout=API_CALL_TIMEOUT,
            )
            if not pod_list.items:
                logger.warning(
                    "[AWMonitor launch_id %s] No pods found for AppWrapper %s",
                    self.launch_id,
                    self.name,
                )
                return False
            for pod in pod_list.items:
                if pod.status.phase == "Running":
                    logger.info(
                        "[AWMonitor launch_id %s] Pod %s is still Running — "
                        "workload appears healthy despite AppWrapper API failure",
                        self.launch_id,
                        pod.metadata.name,
                    )
                    return True
            return False
        except Exception as e:
            logger.warning(
                "[AWMonitor launch_id %s] Pod liveness check also failed: %s",
                self.launch_id,
                e,
            )
            return False

    async def _handle_api_failure_timeout(self: Self) -> bool:
        """
        Handle the case where sustained API failures have exceeded the timeout.

        Performs a secondary pod liveness check. If pods are still running, resets
        the failure timer (extends grace). Otherwise, publishes a fatal failure event
        and stops the monitor.

        Returns:
            True if the monitor should stop (fatal failure), False if grace was extended.
        """
        pods_alive = await self._check_pod_liveness()
        if pods_alive:
            logger.warning(
                "[AWMonitor launch_id %s] AppWrapper API unreachable for >%ds "
                "but pods still Running — extending grace period",
                self.launch_id,
                GBSERVER_API_FAILURE_TIMEOUT,
            )
            self._api_failure_start_time = None  # reset timer
            return False

        # Fatal failure — publish Failed event and stop
        error_message = (
            f"[AWMonitor launch_id {self.launch_id}] Failed to retrieve AppWrapper {self.name} status "
            f"for over {GBSERVER_API_FAILURE_TIMEOUT} seconds. "
            f"The AppWrapper may have been deleted or the Kubernetes API is unreachable. "
            f"Treating this as a fatal error."
        )
        logger.error("%s", error_message)
        payload = json.dumps(
            {
                "appwrapper": self.name,
                "state": "Failed",
                "previous_state": self._last_state,
                "error": error_message,
            },
            indent=4,
        )
        build_event = BuildEvent(
            run_metadata=self.entityrun_metadata,
            type=BuildEventType.MESSAGE_EVENT,
            payload=EventPayload.payload_parser(
                event_type=BuildEventType.MESSAGE_EVENT,
                data={"msg": f"\n```json\n{payload}\n```\n"},
            ),
        )
        if self.event_q is not None:
            await self.event_q.put(build_event)
        await asyncio.sleep(GBSERVER_MONITORING_GRACE_PERIOD)
        self.stop()
        return True

    # ------------------------ override monitor() ------------------------
    async def monitor(self: Self) -> None:
        """Poll (or watch) the AppWrapper; publish on every state change."""
        logger.info(
            "[AWMonitor launch_id %s] Watching AppWrapper %s in namespace %s",
            self.launch_id,
            self.name,
            self.ns,
        )
        # Create Appwrapper monitor which generates Build Events every time appwrapper state changes
        async with await AtomicApiClient.create_api_client(
            kube_config_string=self.kube_config,
            kube_context=self.kube_context,
            ssl_verification=self.ssl_verification,
        ) as api:
            self.v1 = client.CoreV1Api(api_client=api)
            self.custom_api = client.CustomObjectsApi(api_client=api)
            counter: int = 0
            state = "STARTING"
            while not self.stop_event.is_set():
                # Suspend here while paused (e.g. during helm uninstall/reinstall).
                # retry_workload() calls pause() before uninstalling and unpause()
                # resumes us once the new workload is ready.
                await self._run_event.wait()
                if self.stop_event.is_set():
                    break

                counter += 1
                logger.debug(
                    "[AWMonitor launch_id %s] stop_event is not set", self.launch_id
                )
                try:
                    state = await self._get_appwrapper_status()
                    logger.info(f"launch_id={self.launch_id} state={state}")
                    # If we were paused mid-iteration, discard the stale state
                    # (it reflects the old AppWrapper being torn down, not the new one).
                    if not self._run_event.is_set():
                        await asyncio.sleep(self.poll)
                        continue

                    # Check if sustained API failures have exceeded the timeout
                    if self._is_api_failure_timeout():
                        should_stop = await self._handle_api_failure_timeout()
                        if should_stop:
                            return

                    await self._get_appwrapper_failed_pods()
                    if state and state != self._last_state:
                        await self._publish_state_change(
                            new_state=state,
                            get_state_for_exception=(
                                state == "Failed" or state.startswith("Exception:")
                            ),
                        )
                        self._last_state = state
                    if state == "Succeeded":
                        logger.info(
                            "[AWMonitor launch_id %s] %s completed (%s); stopping",
                            self.launch_id,
                            self.name,
                            state,
                        )
                        await asyncio.sleep(GBSERVER_MONITORING_GRACE_PERIOD)
                        self.stop()
                    elif state == "Failed" or state.startswith("Exception:"):
                        error_state = state if state == "Failed" else state[10:]
                        error_message = (
                            f"[AWMonitor launch_id {self.launch_id}] {self.name} is in a {error_state} state. "
                            + "Build will stop because of an appwrapper workload error. "
                            + "The `failed_pods` and `events` sections in the message above have more error details."
                        )
                        logger.error("%s", error_message)
                        await asyncio.sleep(GBSERVER_MONITORING_GRACE_PERIOD)
                        self.stop()
                        # Note: Do NOT raise WorkloadFailedException here
                        # The RetryHandler will evaluate if this failure should trigger a retry
                        # or raise an exception to fail the build
                    elif state != "Running":
                        logger.warning(
                            "[AWMonitor launch_id %s] Appwrapper %s is in a %s state",
                            self.launch_id,
                            self.name,
                            state,
                        )
                    else:
                        if counter % 100 == 0:
                            logger.info(
                                "[AWMonitor launch_id %s] Appwrapper %s is in a %s state",
                                self.launch_id,
                                self.name,
                                state,
                            )
                except Exception as e:
                    if not self._run_event.is_set():
                        # Exception from a K8s call that raced with pause() — safe to ignore.
                        logger.debug(
                            "[AWMonitor launch_id %s] Ignoring exception during paused cleanup: %s",
                            self.launch_id,
                            e,
                        )
                    else:
                        raise
                await asyncio.sleep(self.poll)
            if self.stop_event.is_set():
                logger.warning(
                    "[AWMonitor launch_id %s] stop event has been set, stopping appwrapper monitoring...",
                    self.launch_id,
                )
            logger.info(
                "[AWMonitor launch_id %s] %s in namespace %s has completed. Final state: %s",
                self.launch_id,
                self.name,
                self.ns,
                state,
            )

    async def _get_appwrapper_status(self: Self) -> str:
        """Fetch the status of the AppWrapper (displayed before terminating monitoring)."""
        rstatus = "Unset"
        assert self.custom_api is not None, "CustomObjectsApi not initialized"
        try:
            response = await asyncio.wait_for(
                self.custom_api.get_namespaced_custom_object(
                    group="workload.codeflare.dev",
                    version=os.getenv("K8S_APPWRAPPER_VERSION", "v1beta2"),
                    namespace=self.ns,
                    plural="appwrappers",
                    name=self.name,
                ),
                timeout=API_CALL_TIMEOUT,
            )
            current_resets = response.get("status", {}).get("resettingCount", 0)
            max_retries = (
                response.get("metadata", {})
                .get("annotations", {})
                .get("workload.codeflare.dev.appwrapper/retryLimit", "1")
            )
            self.additional_appwrapper_state_info["current_resets"] = current_resets
            self.additional_appwrapper_state_info["max_retries"] = max_retries
            res_status = response.get("status", {}).get("phase", "Unknown")
            logger.info("Appwrapper %s status is %s", self.name, res_status)
            # Reset failure tracking on successful API call
            self._api_failure_start_time = None
            rstatus = res_status
        except asyncio.TimeoutError as e:
            self._record_api_failure(e)
            rstatus = "Running"
        except client.ApiException as e:
            logger.error(
                "Failed to retrieve AppWrapper %s status - %s: %s",
                self.name,
                type(e).__name__,
                e,
            )
            if e.reason == "Not Found" or e.status == 404:
                logger.warning("Appwrapper %s not found: %s", self.name, e)
                rstatus = f"Exception: Appwrapper {self.name} does not exist any longer in namespace {self.ns}"
            elif "Cannot connect to host" in str(e) or e.status in [
                403,
                408,
                429,
                500,
                502,
                503,
                504,
            ]:
                self._record_api_failure(e)
                rstatus = "Running"
            else:
                rstatus = f"Exception: Failed to get status for appwrapper {self.name} in namespace {self.ns}; client.ApiException {str(e)}, status = {e.status}"
        except aiohttp.ClientError as aio_ce:
            self._record_api_failure(aio_ce)
            rstatus = "Running"
        except AssertionError as ae:
            self._record_api_failure(ae)
            rstatus = "Running"
        except Exception as e:
            logger.error(
                "Failed to retrieve AppWrapper %s status - %s: %s",
                self.name,
                type(e).__name__,
                e,
            )
            if "Cannot connect to host" in str(e):
                self._record_api_failure(e)
                rstatus = "Running"
            else:
                status = getattr(e, "status", "N/A")
                rstatus = f"Exception: Failed to get status for appwrapper {self.name} in namespace {self.ns}; {type(e).__name__} {str(e)}, status = {status}"
                logger.warning(rstatus)

        return rstatus

    # ------------------ helpers ------------------------------
    async def _workloads_for_appwrapper_by_ownerref(
        self: Self, aw_name: str, namespace: str = "default"
    ) -> List[str]:
        """Return Workload names whose ownerReference.name == aw_name."""
        assert self.custom_api is not None, "CustomObjectsApi not initialized"
        wl_list = await self.custom_api.list_namespaced_custom_object(
            group="kueue.x-k8s.io",
            version=os.getenv("K8S_WORKLOAD_VERSION", "v1beta1"),
            namespace=namespace,
            plural="workloads",
        )

        result = []
        for wl in wl_list.get("items", []):
            for ref in wl["metadata"].get("ownerReferences", []):
                if (
                    ref.get("apiVersion", "").startswith("workload.codeflare.dev/")
                    and ref.get("name") == aw_name
                ):
                    result.append(wl["metadata"]["name"])
                    break
        return result

    async def _get_workload_status(self: Self) -> List:
        """Return the status of the Workloads owned by this AppWrapper."""
        appwrapper_workload_list = await self._workloads_for_appwrapper_by_ownerref(
            self.name, self.ns
        )
        if not appwrapper_workload_list:
            return []
        workload_status_list = []
        for workload in appwrapper_workload_list:
            assert self.custom_api is not None, "CustomObjectsApi not initialized"
            workload_status = await self.custom_api.get_namespaced_custom_object_status(
                group="kueue.x-k8s.io",
                version=os.getenv("K8S_WORKLOAD_VERSION", "v1beta1"),
                namespace=self.ns,
                plural="workloads",
                name=workload,
            )
            workload_status_list.append(
                {
                    "workload_name": workload,
                    "workload_status": workload_status.get("status", {}),
                }
            )
        return workload_status_list

    async def _get_appwrapper_failed_pods(self: Self) -> None:
        """Update the dictionary of failed pods owned by this AppWrapper with their status and logs."""
        pod_list = []
        label_selector = f"workload.codeflare.dev/appwrapper={self.name}"
        assert self.v1 is not None, "CoreV1Api not initialized"
        try:
            pod_list = await asyncio.wait_for(
                self.v1.list_namespaced_pod(
                    namespace=self.ns, label_selector=label_selector
                ),
                timeout=API_CALL_TIMEOUT,
            )
            if not pod_list.items:  # type: ignore[attr-defined]
                logger.warning(
                    "No pods found with the specified label %s in namespace %s",
                    label_selector,
                    self.ns,
                )
                return

            for pod in pod_list.items:  # type: ignore[attr-defined]
                pod_name = pod.metadata.name
                # update the list of all the pods ever launched by this appwrapper
                self.launched_pods[pod_name] = pod
                is_failed, failure_reason = self._analyze_pod_status(pod)
                if is_failed:
                    logger.info(
                        "Pod %s has failed (reason: %s), collecting logs...",
                        pod_name,
                        failure_reason,
                    )
                    pod_status = self.failed_pods.get(pod_name, {})
                    if failure_reason:
                        pod_status["failure-reason"] = failure_reason
                    existing_logs = pod_status.get("logs", {})
                    logs = await self._collect_pod_logs(pod_name, pod, existing_logs)
                    pod_status["logs"] = logs
                    self.failed_pods[pod_name] = pod_status
        except asyncio.TimeoutError:
            logger.warning(
                "Timeout fetching pods for AppWrapper %s after %d seconds",
                self.name,
                API_CALL_TIMEOUT,
            )
            return
        except client.ApiException as e:
            logger.error("Error fetching pods: %s", str(e))
            return
        if self.failed_pods:
            logger.warning(
                "Found %d failed pods for appwrapper %s: %s",
                len(self.failed_pods),
                self.name,
                str(self.failed_pods.keys()),
            )

    def _analyze_pod_status(self, pod):
        is_failed = False
        failure_reason_list = []
        # pod_name = pod.metadata.name
        phase = pod.status.phase
        if phase in ["Failed", "Unknown"]:
            if pod.status.reason:
                failure_reason_list.append(pod.status.reason)
            is_failed = True

        # Check container statuses for more specific information
        # container_statuses = []
        if pod.status.container_statuses:
            for container_status in pod.status.container_statuses:
                if container_status.state.terminated:
                    terminated = container_status.state.terminated
                    exit_code = terminated.exit_code
                    reason = terminated.reason
                    if exit_code != 0:
                        is_failed = True
                        failure_reason_list.append(
                            f"{container_status.name} failed with exit code {exit_code}; reason: {reason}"
                        )
        return is_failed, "\n".join(failure_reason_list)

    async def _collect_pod_logs(self, pod_name, pod, existing_logs=None, tail_lines=50):
        """
        Collect last 50 lines of logs from all containers in a pod.
        Only overwrites existing logs if new non-empty logs are retrieved.

        Args:
            pod_name: Name of the pod
            pod: Pod object
            existing_logs: Dictionary of previously collected logs (optional)
            tail_lines: number of lines to return

        Returns:
            dict: Dictionary mapping container names to their logs
        """
        # Start with existing logs if provided, otherwise empty dict
        logs_dict = existing_logs.copy() if existing_logs else {}

        # Function to fetch logs for a single container
        async def fetch_container_logs(container_name: str):
            try:
                assert self.v1 is not None, "CoreV1Api not initialized"
                logs = await asyncio.wait_for(
                    self.v1.read_namespaced_pod_log(
                        name=pod_name,
                        namespace=self.ns,
                        container=container_name,
                        tail_lines=tail_lines,
                    ),
                    timeout=API_CALL_TIMEOUT,
                )
                # Split into lines and filter empty lines
                log_lines = [line for line in logs.strip().split("\n") if line]
                return container_name, log_lines
            except asyncio.TimeoutError:
                logger.warning(
                    "Timeout fetching logs for container %s after %d seconds",
                    container_name,
                    API_CALL_TIMEOUT,
                )
                return container_name, []
            except client.ApiException as e:
                # Return error info if logs can't be fetched
                logger.warning(
                    "Failed to get logs for container %s: %s", container_name, e.reason
                )
                return container_name, []

        # Collect logs from pod containers
        if pod.spec.containers:
            # Get all container names from the pod
            container_names = [container.name for container in pod.spec.containers]
            # Fetch logs from all containers concurrently
            tasks = [fetch_container_logs(name) for name in container_names]
            container_logs = await asyncio.gather(*tasks)
            # Convert list of tuples to dictionary
            collected_logs = dict(container_logs)
            for container_name in container_names:
                log_lines = collected_logs.get(container_name)
                if log_lines:
                    logs_dict[container_name] = log_lines
                # If logs are empty/None and key doesn't exist yet, don't add it
                # If logs are empty/None but key exists, keep the old value (don't overwrite)

        return logs_dict

    async def _get_new_events(self, appwrapper_pod_list: List[str]) -> List[Dict]:
        """
        Returns:
        - ALL events for the AppWrapper whose name == self.name
        - PLUS events with type != "Normal" for the pods associated with the appwrapper workflows
        """
        events = []
        assert self.v1 is not None, "CoreV1Api not initialized"
        # get events for appwrapper
        try:
            appwrapper_events = await asyncio.wait_for(
                self.v1.list_namespaced_event(
                    namespace=self.ns,
                    field_selector=(
                        f"involvedObject.kind=AppWrapper,"
                        f"involvedObject.name={self.name}"
                    ),
                ),
                timeout=API_CALL_TIMEOUT,
            )
            events.extend(appwrapper_events.items)
        except asyncio.TimeoutError:
            logger.warning(
                "Timeout fetching events for AppWrapper %s after %d seconds",
                self.name,
                API_CALL_TIMEOUT,
            )
        except Exception as e:
            logger.warning("Error fetching events for AppWrapper %s: %s", self.name, e)

        # get abnormal events for all the pods launched by this appwrapper,
        # not only the failed pods, so that we can catch significant events
        # (Evicted, ErrImagePull) that occur even if the pod is not necessarily
        # marked as failed
        appwrapper_pod_list = self.launched_pods.keys()  # type: ignore[assignment]
        abnormal_pod_events = []
        if appwrapper_pod_list is not None:
            for pod_name in appwrapper_pod_list:
                try:
                    field_selector = f"involvedObject.name={pod_name}"
                    # Retrieve events for the specific pod
                    pod_events = await asyncio.wait_for(
                        self.v1.list_namespaced_event(
                            namespace=self.ns,
                            field_selector=field_selector,
                        ),
                        timeout=API_CALL_TIMEOUT,
                    )
                    pod_events_list = pod_events.items
                    abnormal_events = []
                    if pod_events_list:
                        for event in pod_events_list:
                            event_type = (event.type or "").strip() or "Unknown"
                            event_reason = (event.reason or "").strip() or "Unknown"
                            if (event_type != "Normal") or (
                                event_reason in PROBLEMATIC_NORMAL_EVENTS
                            ):
                                abnormal_events.append(event)
                        logger.info(
                            "Found %d events (%d abnormal) for pod %s",
                            len(pod_events_list),
                            len(abnormal_events),
                            pod_name,
                        )
                    abnormal_pod_events.extend(abnormal_events)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timeout fetching events for pod %s after %d seconds",
                        pod_name,
                        API_CALL_TIMEOUT,
                    )
                except client.ApiException as e:
                    logger.error(
                        "Error fetching events for pod %s: %s", pod_name, str(e)
                    )
        events.extend(abnormal_pod_events)

        fresh_events = []
        for event in events:
            event_type = (event.type or "").strip() or "Unknown"
            inv = event.involved_object
            inv_name = inv.name or ""
            inv_kind = inv.kind or ""
            uid = event.metadata.uid
            if uid not in self._seen_event_uids:
                fresh_events.append(
                    {
                        "object_type": inv_kind,
                        "object_name": inv_name,
                        "reason": event.reason,
                        "message": event.message,
                        "type": event_type,
                        "time": (
                            str(event.last_timestamp)
                            if event.last_timestamp
                            else str(event.event_time)
                        ),
                    }
                )
                self._seen_event_uids.add(uid)
        self.latest_events = fresh_events
        return fresh_events

    async def _get_state_change(
        self: Self, new_state: str, get_state_for_exception: bool = False
    ) -> str:
        payload: dict[str, Any] = {
            "appwrapper": self.name,
            "state": new_state,
            "previous_state": self._last_state,
            "current_resets": self.additional_appwrapper_state_info.get(
                "current_resets", 0
            ),
            "max_retries": self.additional_appwrapper_state_info.get(
                "max_resets", "unlimited"
            ),
            "workload_status": await self._get_workload_status(),
        }
        pods_placement = {
            pod_name: pod.spec.node_name for pod_name, pod in self.launched_pods.items()
        }
        payload["pod_placement"] = pods_placement
        await self._get_appwrapper_failed_pods()
        if get_state_for_exception:
            payload["failed_pods"] = self.failed_pods
        payload["events"] = await self._get_new_events(self.failed_pods.keys())  # type: ignore[arg-type]
        return f"\n```json\n{json.dumps(payload, indent=4)}\n```\n"

    async def _publish_state_change(
        self: Self, new_state: str, get_state_for_exception: bool = False
    ) -> None:
        payload = await self._get_state_change(
            new_state=new_state, get_state_for_exception=get_state_for_exception
        )
        build_event = BuildEvent(
            run_metadata=self.entityrun_metadata,
            type=BuildEventType.MESSAGE_EVENT,
            payload=EventPayload.payload_parser(
                event_type=BuildEventType.MESSAGE_EVENT,
                data={"msg": payload},
            ),
        )
        if self.event_q is not None:
            await self.event_q.put(build_event)
        logger.info(
            "[AWMonitor launch_id %s] Published state change for %s:%s",
            self.launch_id,
            self.name,
            payload,
        )
        # publish metrics as an event
        if self.event_q is None:
            return
        metric = Metric(
            name=MetricName.APPWRAPPER_STATUS_CHANGE_TIMESTAMP,
            value=get_utc_time(),
            units=MetricUnits.TIMESTAMP,
            metadata=MetricMetadata(
                username=self.entityrun_metadata.username or "",
                build_id=self.entityrun_metadata.build_id or "",
                targetrun_id=self.entityrun_metadata.targetrun_id or "",
                targetsteprun_id=self.entityrun_metadata.targetsteprun_id or "",
                targetstep_uri=self.entityrun_metadata.targetstep_uri or "",
                target_name=self.entityrun_metadata.target_name or "",
                launch_id=self.launch_id or "",
                k8s_resource_type="appwrapper",
                k8s_resource_name=self.name,
                k8s_resource_namespace=self.ns,
                k8s_resource_status=new_state,
            ),
        )
        metrics_event = BuildEvent(
            run_metadata=self.entityrun_metadata,
            type=BuildEventType.METRICS_EVENT,
            payload=EventPayload.payload_parser(
                event_type=BuildEventType.METRICS_EVENT,
                data={"metrics": [metric]},
            ),
        )
        self.event_q.put_nowait(metrics_event)
