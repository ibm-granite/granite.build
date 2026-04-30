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
Lazy singleton accessor for the process-wide NodeHealthTracker.

Provides get_node_health_tracker() which creates the tracker on first access
using get_admin_storage() and create_alert_handler_from_env(). This decouples
the tracker from any specific process (e.g., BuildWatcher) — any consumer
can access it and it will self-initialize.
"""

import threading
from typing import Optional

from gbserver.resilience.alert_handlers import create_alert_handler_from_env
from gbserver.resilience.node_health_tracker import NodeHealthTracker
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

_node_health_tracker: Optional[NodeHealthTracker] = None
_tracker_lock = threading.Lock()
_tracker_initialized = False  # Distinguishes "None because failed" from "not yet tried"


def get_node_health_tracker() -> Optional[NodeHealthTracker]:
    """Get or lazily create the process-wide NodeHealthTracker singleton.

    On first call, creates the tracker using:
    - node_failure_storage from get_admin_storage()
    - alert_handler from create_alert_handler_from_env()

    Returns None (with a warning) if initialization fails. All consumers
    already handle None gracefully (RetryHandler._record_node_failures
    returns early, K8s stores None and passes it through).

    Thread-safe via double-checked locking.
    """
    global _node_health_tracker, _tracker_initialized

    if _tracker_initialized:
        return _node_health_tracker

    with _tracker_lock:
        if _tracker_initialized:
            return _node_health_tracker

        try:
            from gbserver.storage.singleton_storage import get_admin_storage

            admin_storage = get_admin_storage()
            alert_handler = create_alert_handler_from_env()

            _node_health_tracker = NodeHealthTracker(
                node_failure_storage=admin_storage.node_failure_storage,
                alert_handler=alert_handler,
            )

            logger.info(
                "[NodeHealthTracker] Lazy singleton initialized " "(alert_handler=%s, storage=%s)",
                type(alert_handler).__name__ if alert_handler else "None",
                type(admin_storage.node_failure_storage).__name__,
            )
        except Exception as e:
            logger.warning(
                "[NodeHealthTracker] Failed to initialize lazy singleton: %s. "
                "Node health monitoring will be disabled.",
                e,
            )
            _node_health_tracker = None

        _tracker_initialized = True
        return _node_health_tracker


def set_node_health_tracker(tracker: Optional[NodeHealthTracker]) -> None:
    """Override the NodeHealthTracker singleton. Primarily for testing."""
    global _node_health_tracker, _tracker_initialized
    with _tracker_lock:
        _node_health_tracker = tracker
        _tracker_initialized = True


def reset_node_health_tracker() -> None:
    """Reset the singleton so the next get_node_health_tracker() re-initializes.

    Primarily for testing to ensure clean state between tests.
    """
    global _node_health_tracker, _tracker_initialized
    with _tracker_lock:
        _node_health_tracker = None
        _tracker_initialized = False
