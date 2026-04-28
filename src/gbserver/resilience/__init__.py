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
Resilience module for gbserver.

This module provides resilience and recovery mechanisms for handling
failures in distributed workload execution.
"""

from gbserver.resilience.alert_handlers import (
    AlertHandler,
    CompositeAlertHandler,
    LoggingAlertHandler,
    NodeHealthAlert,
    RetryableAlertHandler,
    SlackAlertHandler,
    WebhookAlertHandler,
    create_alert_handler_from_env,
)
from gbserver.resilience.nht_singleton import (
    get_node_health_tracker,
    reset_node_health_tracker,
    set_node_health_tracker,
)
from gbserver.resilience.node_health_tracker import (
    NodeHealthTracker,
)
from gbserver.resilience.retry_handler import (
    RetryHandler,
    RetryStrategy,
    build_retry_strategies_from_config,
)
from gbserver.resilience.strategies import (
    FileNotFoundRetryStrategy,
    LsfTransientErrorRetryStrategy,
    NCCLErrorRetryStrategy,
    PodEvictionRetryStrategy,
    UnhealthyInsufficientPodsRetryStrategy,
)

__all__ = [
    # Retry handling
    "RetryHandler",
    "RetryStrategy",
    "build_retry_strategies_from_config",
    # Retry strategies
    "FileNotFoundRetryStrategy",
    "LsfTransientErrorRetryStrategy",
    "NCCLErrorRetryStrategy",
    "PodEvictionRetryStrategy",
    "UnhealthyInsufficientPodsRetryStrategy",
    # Node health tracking
    "NodeHealthTracker",
    "get_node_health_tracker",
    "set_node_health_tracker",
    "reset_node_health_tracker",
    # Alerting
    "AlertHandler",
    "NodeHealthAlert",
    "LoggingAlertHandler",
    "WebhookAlertHandler",
    "SlackAlertHandler",
    "CompositeAlertHandler",
    "RetryableAlertHandler",
    "create_alert_handler_from_env",
]
