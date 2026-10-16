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

# Bridge settings a worker process needs to build its own log handler.
BRIDGE_ENV_KEYS = ("AUTOTUNE_JOB_ID", "AUTOTUNE_ENDPOINT_URL")


def bridge_env_vars():
    """The bridge env vars currently set, as a dict (empty when logging is off).

    Worker processes on *other* nodes do not inherit the driver's environment,
    so these have to travel explicitly as a job-level Ray ``runtime_env``.
    Without them a worker builds no handler and its logs can only reach the
    backend via the driver's forwarded stdout, where they are job-level and
    cannot be attributed to a trial.
    """
    return {k: os.environ[k] for k in BRIDGE_ENV_KEYS if os.environ.get(k)}


def bridge_runtime_env():
    """``ray.init`` kwargs that carry the bridge settings to workers.

    Returns an empty dict when bridge logging is off (the default), so the
    Ray runtime_env machinery is only engaged for runs that actually need it.
    """
    env = bridge_env_vars()
    return {"runtime_env": {"env_vars": env}} if env else {}


def bind_trial_id(trial_id):
    """Attribute this process's log records to ``trial_id``.

    Called by each driver once it knows its trial. A trial worker process
    serves one trial at a time, so setting the handler default here is correct
    and — unlike a context-scoped id — covers records emitted from any thread
    in the process (HF Trainer, dataloader workers, DeepSpeed). The driver
    process must never do this: every trial's forwarded output converges on its
    handler, so a process-wide id there is exactly the bug this avoids.
    """
    if not trial_id:
        return
    from autotune.callbacks.logging_service import BufferedLogHandler

    for h in logging.getLogger().handlers:
        if isinstance(h, BufferedLogHandler):
            h.set_trial_id(trial_id)


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
