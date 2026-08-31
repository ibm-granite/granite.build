# coding=utf-8
# Copyright 2023-present International Business Machines Corporation
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

import logging
import os
import sys

_CONFIGURED = False
LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_level=logging.INFO):
    """Configure the root logger for the current process.

    Safe to call multiple times — the second call is a no-op.
    Designed to work in both the main process and Ray worker processes.

    In worker processes, if AUTOTUNE_JOB_ID and AUTOTUNE_ENDPOINT_URL
    environment variables are set, a BufferedLogHandler is created so
    that worker logs are flushed to the API/DB endpoint.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    root = logging.getLogger()
    root.setLevel(log_level)

    # stderr handler using the real fd (avoids PrintLogger re-entry
    # in the main process where sys.stderr may be replaced).
    stderr_handler = logging.StreamHandler(sys.__stderr__)
    stderr_handler.setFormatter(logging.Formatter(LOG_FMT, datefmt=LOG_DATEFMT))
    root.addHandler(stderr_handler)

    # If job_id and endpoint_url are available via env vars, create a
    # BufferedLogHandler so worker logs reach the API/DB.
    job_id = os.environ.get("AUTOTUNE_JOB_ID")
    endpoint_url = os.environ.get("AUTOTUNE_ENDPOINT_URL")
    if job_id and endpoint_url:
        from autotune.callbacks.logging_service import BufferedLogHandler

        handler = BufferedLogHandler(
            job_id=job_id,
            endpoint_url=endpoint_url,
            flush_interval=10.0,
        )
        root.addHandler(handler)
