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
Retry strategy for LSF transient errors.

This strategy handles the 'Cannot open your job file' transient LSF error,
which can occur when the LSF system temporarily fails to open the job script.
"""

from typing import Self

from gbserver.resilience.retry_handler import RetryStrategy
from gbserver.types.buildevent import BuildEvent, BuildEventType
from gbserver.types.constants import GBSERVER_LSF_TRANSIENT_ERROR_RETRY_DELAY
from gbserver.types.errors import ERR_LSF_CANNOT_OPEN_JOB_FILE
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class LsfTransientErrorRetryStrategy(RetryStrategy):
    """
    Retry strategy for LSF transient 'Cannot open your job file' errors.

    This strategy triggers a retry when the LSF bsub monitor detects the
    transient error string in its output. The error is typically temporary
    and resolves on retry.
    """

    accepts_object_types: bool = False

    def should_retry(
        self: Self,
        event: BuildEvent,
    ) -> bool:
        if event.type != BuildEventType.MESSAGE_EVENT:
            return False

        msg = getattr(event.payload, "msg", "") or ""
        should = ERR_LSF_CANNOT_OPEN_JOB_FILE in msg

        if should:
            logger.warning("LsfTransientErrorRetryStrategy: detected transient LSF error (%s)", msg)

        return should

    def get_retry_delay(self: Self, retry_count: int) -> float:
        return float(GBSERVER_LSF_TRANSIENT_ERROR_RETRY_DELAY)
