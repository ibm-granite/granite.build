#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

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


class AsperaRetryStrategy(RetryStrategy):
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

    ASPERA_ERROR_PATTERNS = [r"Aspera transfer failed"]

    def __init__(self: Self) -> None:
        """Initialize Aspera retry strategy with compiled regex patterns."""
        super().__init__()
        self._patterns = [re.compile(p) for p in self.ASPERA_ERROR_PATTERNS]

    def should_retry(
        self: Self,
        event: BuildEvent,
    ) -> bool:
        """
        Check if event contains FileNotFoundError warranting retry.

        Args:
            event: BuildEvent from the monitor

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
                        "[AsperaRetryStrategy] Detected FileNotFoundError in event: %s",
                        msg[:200],
                    )
                    return True

        except AttributeError as e:
            logger.debug(
                "[AsperaRetryStrategy] Event missing expected attributes: %s", e
            )
        except Exception as e:
            logger.debug("[AsperaRetryStrategy] Could not parse event: %s", e)

        return False

    def extract_nodes_to_avoid(
        self: Self,
        _event: BuildEvent,
    ) -> Set[str]:
        """ """
        nodes = set()  # type: ignore[var-annotated]
        return nodes
