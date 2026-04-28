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
Retry strategy for NCCL GPU errors.

Detects NCCL internal errors and illegal memory access errors on GPU nodes,
enabling automatic retry with node avoidance.

Related Issues:
- #1609: NCCL internal errors on specific GPU nodes
"""

import json
import re
from typing import Self, Set

from gbserver.resilience.retry_handler import RetryStrategy
from gbserver.types.buildevent import BuildEvent, BuildEventType
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class NCCLErrorRetryStrategy(RetryStrategy):
    """
    Retry strategy for NCCL GPU errors.

    Detects GPU-related hardware failures in distributed training workloads:
    - NCCL Error 3: internal error
    - NCCL Error 2: unhandled system error
    - CUDA illegal memory access errors (hardware-related)
    - CUDNN internal errors

    When detected, extracts the node name from pod placement info
    and recommends retry with node anti-affinity to avoid the problematic
    GPU node.

    Note: The following are NOT retried as retry won't help:
    - CUDA out of memory (configuration issue: model/batch size too large)
    - CUDA launch failures (software bug: invalid kernel configuration)
    - CUDA illegal instruction (software bug: wrong architecture/compilation)
    - CUDA misaligned address (software bug: pointer alignment error)

    Example error messages detected:
        RuntimeError: NCCL Error 3: internal error
        RuntimeError: CUDA error: an illegal memory access was encountered
        RuntimeError: Worker failed with error 'Triton Error [CUDA]: an illegal memory access was encountered'
        cudaErrorIllegalAddress (from vLLM workers)

    Related Issues:
        - #1691: vLLM CUDA illegal address errors on bad GPUs
        - #1779: Triton/vLLM wrapped CUDA errors not detected
    """

    # This strategy doesn't filter by Kubernetes object type
    accepts_object_types = False

    # NCCL error patterns to detect
    NCCL_ERROR_PATTERNS = [
        r"RuntimeError: NCCL Error \d+: internal error",
        r"RuntimeError: NCCL Error \d+: unhandled (?:system|cuda) error",
        r"NCCL error in:.*internal error",
        r"NCCL failure.*internal error",
    ]

    # CUDA error patterns to detect - hardware failures only
    # Software bugs (launch failure, illegal instruction, misalignment) are excluded
    CUDA_ERROR_PATTERNS = [
        # "CUDA error: an illegal memory access" — matches regardless of wrapper:
        #   RuntimeError: CUDA error: an illegal memory access was encountered
        #   torch.AcceleratorError: CUDA error: an illegal memory access was encountered
        r"CUDA error: an illegal memory access",
        r"CUDNN_STATUS_INTERNAL_ERROR",
        r"CUDA error: device-side assert triggered",
        # CUDA error code from Issue #1691 (hardware failure)
        r"cudaErrorIllegalAddress",  # Illegal memory address
        # Triton/vLLM wrapper format — vLLM workers report CUDA errors
        # wrapped in Triton error format instead of direct "CUDA error:" prefix
        r"Triton Error \[CUDA\].*illegal memory access",
        r"Worker failed with error.*illegal memory access",
    ]

    def __init__(self: Self) -> None:
        """Initialize NCCL error retry strategy with compiled regex patterns."""
        super().__init__()
        self._nccl_patterns = [re.compile(p) for p in self.NCCL_ERROR_PATTERNS]
        self._cuda_patterns = [re.compile(p) for p in self.CUDA_ERROR_PATTERNS]

    def should_retry(
        self: Self,
        event: BuildEvent,
    ) -> bool:
        """
        Check if event contains NCCL/CUDA GPU errors warranting retry.

        Args:
            event: BuildEvent from the monitor
            retry_count: Current retry count
            max_retries: Maximum allowed retries

        Returns:
            bool: True if NCCL/CUDA error detected and retry is warranted
        """
        # Only process MESSAGE_EVENT types (from log monitors)
        if event.type != BuildEventType.MESSAGE_EVENT:
            return False

        try:
            if not hasattr(event.payload, "msg"):
                return False

            msg = event.payload.msg

            # Check for NCCL errors
            for pattern in self._nccl_patterns:
                if pattern.search(msg):
                    logger.warning(
                        "[NCCLErrorRetryStrategy] Detected NCCL error in event: %s",
                        msg[:200],
                    )
                    return True

            # Check for CUDA errors
            for pattern in self._cuda_patterns:
                if pattern.search(msg):
                    logger.warning(
                        "[NCCLErrorRetryStrategy] Detected CUDA error in event: %s",
                        msg[:200],
                    )
                    return True

        except AttributeError as e:
            logger.debug(
                "[NCCLErrorRetryStrategy] Event missing expected attributes: %s", e
            )
        except Exception as e:
            logger.debug("[NCCLErrorRetryStrategy] Could not parse event: %s", e)

        return False

    def extract_nodes_to_avoid(
        self: Self,
        _event: BuildEvent,
    ) -> Set[str]:
        """
        Extract the node name where the GPU error occurred.

        Looks for node placement info in the event metadata.
        K8s monitoring embeds node names in pod events.

        Args:
            _event: BuildEvent from the monitor

        Returns:
            Set[str]: Set of node names to avoid in retry
        """
        nodes = set()

        try:
            # Check if event has metadata with node info
            if hasattr(_event, "metadata") and _event.metadata:
                node_name = _event.metadata.get("node_name")
                if node_name:
                    nodes.add(node_name)
                    logger.info(
                        "[NCCLErrorRetryStrategy] Extracted node to avoid: %s",
                        node_name,
                    )
                    return nodes

            # Try to extract from JSON payload (K8s monitor format)
            if (
                hasattr(_event, "payload")
                and _event.payload
                and hasattr(_event.payload, "msg")
            ):
                msg = _event.payload.msg

                # Parse JSON if present
                data = None
                if "```json" in msg:
                    json_start = msg.find("```json") + 7
                    json_end = msg.find("```", json_start)
                    json_str = msg[json_start:json_end].strip()
                    data = json.loads(json_str)
                else:
                    # Try parsing entire message as JSON
                    try:
                        data = json.loads(msg)
                    except json.JSONDecodeError:
                        pass

                if data:
                    # Look for node name in various fields
                    node_name = (
                        data.get("node_name")
                        or data.get("nodeName")
                        or data.get("spec", {}).get("nodeName")
                    )

                    if node_name:
                        nodes.add(node_name)
                        logger.info(
                            "[NCCLErrorRetryStrategy] Extracted node from payload: %s",
                            node_name,
                        )

                    # Check pod_placement from AppWrapperMonitor
                    # Format: {"pod_name": "node_name", ...}
                    pod_placement = data.get("pod_placement", {})
                    if pod_placement and not nodes:
                        for pod_name, placed_node in pod_placement.items():
                            if placed_node:
                                nodes.add(placed_node)
                                logger.info(
                                    "[NCCLErrorRetryStrategy] Extracted node from "
                                    "pod_placement (%s): %s",
                                    pod_name,
                                    placed_node,
                                )

        except json.JSONDecodeError as e:
            logger.debug(
                "[NCCLErrorRetryStrategy] Could not parse JSON from payload: %s", e
            )
        except Exception as e:
            logger.warning(
                "[NCCLErrorRetryStrategy] Could not extract node name: %s", e
            )

        if not nodes:
            logger.warning(
                "[NCCLErrorRetryStrategy] No node name found in event metadata. "
                "Retry will not use node anti-affinity."
            )

        return nodes
