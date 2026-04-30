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
Retry strategy for FileNotFoundError on NFS-mounted volumes.

Detects FileNotFoundError in pod logs that indicate flaky NFS mounts
rather than genuinely missing files, enabling automatic retry with
node avoidance.

Related Issues:
- #1829: vLLM eval FileNotFoundError on files that exist on NFS mount
"""

import json
import re
from typing import Self, Set

from gbserver.resilience.retry_handler import RetryStrategy
from gbserver.types.buildevent import BuildEvent, BuildEventType
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class FileNotFoundRetryStrategy(RetryStrategy):
    """
    Retry strategy for FileNotFoundError caused by flaky NFS mounts.

    Detects FileNotFoundError in pod log messages. These errors typically
    occur when NFS mounts are degraded on specific nodes — the files exist
    but cannot be accessed. Retrying on a different node resolves the issue.

    Example error messages detected:
        FileNotFoundError: No such file or directory: '/gb-lakehouse-prod-read-only/.../model.safetensors'
        FileNotFoundError: [Errno 2] No such file or directory: '/path/to/file'

    Related Issues:
        - #1829: vLLM eval FileNotFoundError on NFS-mounted safetensors files
    """

    # This strategy doesn't filter by Kubernetes object type
    accepts_object_types = False

    FILE_NOT_FOUND_PATTERNS = [
        r"FileNotFoundError: No such file or directory",
        r"FileNotFoundError: \[Errno 2\]",
    ]

    def __init__(self: Self) -> None:
        """Initialize FileNotFound retry strategy with compiled regex patterns."""
        super().__init__()
        self._patterns = [re.compile(p) for p in self.FILE_NOT_FOUND_PATTERNS]

    def should_retry(
        self: Self,
        event: BuildEvent,
    ) -> bool:
        """
        Check if event contains FileNotFoundError warranting retry.

        Args:
            event: BuildEvent from the monitor
            retry_count: Current retry count
            max_retries: Maximum allowed retries

        Returns:
            bool: True if FileNotFoundError detected and retry is warranted
        """
        if event.type != BuildEventType.MESSAGE_EVENT:
            return False

        try:
            if not hasattr(event.payload, "msg"):
                return False

            msg = event.payload.msg  # type: ignore[union-attr]

            for pattern in self._patterns:
                if pattern.search(msg):
                    logger.warning(
                        "[FileNotFoundRetryStrategy] Detected FileNotFoundError in event: %s",
                        msg[:200],
                    )
                    return True

        except AttributeError as e:
            logger.debug("[FileNotFoundRetryStrategy] Event missing expected attributes: %s", e)
        except Exception as e:
            logger.debug("[FileNotFoundRetryStrategy] Could not parse event: %s", e)

        return False

    def extract_nodes_to_avoid(
        self: Self,
        _event: BuildEvent,
    ) -> Set[str]:
        """
        Extract the node name where the FileNotFoundError occurred.

        Looks for node placement info in the event metadata or JSON payload.

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
                        "[FileNotFoundRetryStrategy] Extracted node to avoid: %s",
                        node_name,
                    )
                    return nodes

            # Try to extract from JSON payload
            if hasattr(_event, "payload") and _event.payload and hasattr(_event.payload, "msg"):
                msg = _event.payload.msg

                # Parse JSON if present
                data = None
                if "```json" in msg:
                    json_start = msg.find("```json") + 7
                    json_end = msg.find("```", json_start)
                    json_str = msg[json_start:json_end].strip()
                    data = json.loads(json_str)
                else:
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
                            "[FileNotFoundRetryStrategy] Extracted node from payload: %s",
                            node_name,
                        )

                    # Check pod_placement from AppWrapperMonitor
                    pod_placement = data.get("pod_placement", {})
                    if pod_placement and not nodes:
                        for pod_name, placed_node in pod_placement.items():
                            if placed_node:
                                nodes.add(placed_node)
                                logger.info(
                                    "[FileNotFoundRetryStrategy] Extracted node from "
                                    "pod_placement (%s): %s",
                                    pod_name,
                                    placed_node,
                                )

        except json.JSONDecodeError as e:
            logger.debug("[FileNotFoundRetryStrategy] Could not parse JSON from payload: %s", e)
        except Exception as e:
            logger.warning("[FileNotFoundRetryStrategy] Could not extract node name: %s", e)

        if not nodes:
            logger.warning(
                "[FileNotFoundRetryStrategy] No node name found in event metadata. "
                "Retry will not use node anti-affinity."
            )

        return nodes
