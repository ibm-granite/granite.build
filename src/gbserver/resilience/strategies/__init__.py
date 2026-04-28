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
Retry strategies for the resilience module.

This package contains concrete implementations of RetryStrategy for different
failure patterns and scenarios.
"""

from gbserver.resilience.strategies.aspera_failure import AsperaRetryStrategy
from gbserver.resilience.strategies.file_not_found import FileNotFoundRetryStrategy
from gbserver.resilience.strategies.lsf_transient_error import (
    LsfTransientErrorRetryStrategy,
)
from gbserver.resilience.strategies.nccl_error import NCCLErrorRetryStrategy
from gbserver.resilience.strategies.pod_eviction import PodEvictionRetryStrategy
from gbserver.resilience.strategies.unhealthy_insufficient_pods import (
    UnhealthyInsufficientPodsRetryStrategy,
)

__all__ = [
    "FileNotFoundRetryStrategy",
    "LsfTransientErrorRetryStrategy",
    "NCCLErrorRetryStrategy",
    "PodEvictionRetryStrategy",
    "UnhealthyInsufficientPodsRetryStrategy",
    "AsperaRetryStrategy",
]
