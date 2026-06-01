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
Monitor the job submitted via bsub.
"""

import asyncio
import json
import shlex
from pathlib import Path
from typing import Any, Optional, Self

from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from gbserver.monitoring.monitor_base import MonitorBase
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventMessagePayload,
    BuildEventType,
    EntityRunMetadata,
)
from gbserver.types.constants import GBSERVER_MONITORING_GRACE_PERIOD
from gbserver.types.errors import (
    ERR_LSF_CANNOT_OPEN_JOB_FILE,
    ErrLSFCannotOpenJobFile,
    ErrSSHConnectionError,
)
from gbserver.utils.logger import get_logger
from gbserver.utils.ssh_tunnel import SshTunnel
from gbserver.utils.utils import cmd_safe_join

logger = get_logger(__name__)

JOB_LOG_STDOUT_FILENAME = "job_log.out"


class BJobRecord(BaseModel):
    """A bjob record."""

    JOBID: str
    STAT: str
    EXIT_CODE: str
    EXIT_REASON: str


class BJobOutput(BaseModel):
    """Output of bjob."""

    COMMAND: str
    JOBS: int
    RECORDS: list[BJobRecord]


class LSFBsubMonitor(MonitorBase):
    """Monitor the status of a job launched with bsub."""

    def __init__(
        self: Self,
        lsf: Any,  # Lsf Environment
        job_id: str,
        launch_id: str,
        entityrun_metadata: Optional[EntityRunMetadata] = None,
        event_queue: Optional[asyncio.Queue] = None,
        stop_event: Optional[asyncio.Event] = None,
        monitor_interval: int = 5,
    ) -> None:
        super().__init__(
            launch_id=launch_id,
            entityrun_metadata=entityrun_metadata,
            event_queue=event_queue,
            stop_event=stop_event,
        )
        self.lsf = lsf
        self.job_id = job_id
        self.launch_id = launch_id
        self.monitor_interval = monitor_interval
        self.monitor_command = ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(5),
        retry=retry_if_exception_type(ErrSSHConnectionError),
    )
    async def _check_for_transient_lsf_error(
        self: Self,
        fallback_output_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Check the job_log.out file for transient LSF errors that should trigger a retry.

        Gets OUTPUT_FILE from bjobs and reads the file content in a single SSH command.
        Falls back to fallback_output_path if bjobs fails.
        Retries on SSH/connection errors using tenacity.

        Args:
            fallback_output_path: Optional path to use if bjobs fails to return OUTPUT_FILE.

        Returns the error message if a transient error is found, None otherwise.
        Raises ErrSSHConnectionError if SSH errors persist after retries.
        """
        # Build combined command: get OUTPUT_FILE from bjobs, then tac the file
        # Using -noheader to get just the path value without JSON parsing
        # Falls back to fallback_output_path if bjobs returns empty
        combined_cmd = (
            f'OUTPUT_FILE=$(bjobs -a -o "OUTPUT_FILE" -noheader {self.job_id} 2>/dev/null | tr -d " "); '
            f'[ -z "$OUTPUT_FILE" ] && OUTPUT_FILE="{fallback_output_path}"; '
            f"tac \"$OUTPUT_FILE\" | awk '!flag; /Sender: LSF System/{{flag=1}};' | tac"
        )

        if self.lsf.use_ssh:
            from gbserver.environment.lsf import Lsf

            assert isinstance(self.lsf, Lsf)
            ssh_cmd = await self.lsf.create_ssh_base_cmd()
            ssh_cmd.append(shlex.quote(combined_cmd))
            read_command = " ".join(ssh_cmd)
        else:
            read_command = combined_cmd

        logger.warning(
            "[LSFBsubMonitor %s] Checking for transient errors: %s",
            self.launch_id,
            read_command,
        )

        proc = await asyncio.create_subprocess_shell(
            read_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        stderr_str = stderr.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            # Check for SSH/connection errors that should trigger a retry
            if ErrSSHConnectionError.matches_error_str(stderr_str):
                logger.warning(
                    "[LSFBsubMonitor %s] SSH/connection error: %s",
                    self.launch_id,
                    stderr_str,
                )
                raise ErrSSHConnectionError(stderr_str)

            logger.warning(
                "[LSFBsubMonitor %s] Command failed: %s",
                self.launch_id,
                stderr_str,
            )
            return None

        job_log_content = stdout.decode("utf-8", errors="replace")
        logger.warning(job_log_content)

        # Check for the "Cannot open your job file" transient error
        if ErrLSFCannotOpenJobFile.matches_error_str(job_log_content):
            logger.warning(
                "[LSFBsubMonitor %s] Found transient LSF error: Cannot open your job file",
                self.launch_id,
            )
            return job_log_content

        return None

    async def monitor(self: Self) -> None:
        bare_bjobs_command = " ".join(
            [
                "bjobs",
                "-o",
                '"jobid stat exit_code exit_reason"',
                "-json",
                self.job_id,
            ]
        )
        ssh_tunnel = self.lsf.get_ssh_tunnel()
        if (
            ssh_tunnel
        ):  # Should always be the case except maybe when not lsf.use_ssh during debugging?
            self.monitor_command = bare_bjobs_command
        else:
            ssh_cmd = await self.lsf.create_ssh_base_cmd()
            assert isinstance(ssh_cmd, list), f"invalid ssh_cmd: {ssh_cmd}"
            ssh_cmd.append(bare_bjobs_command)
            self.monitor_command = cmd_safe_join(ssh_cmd)
        logger.info("running the ssh cmd for monitoring: %s", self.monitor_command)
        returncode = -1
        while not self.stop_event.is_set():
            try:
                await asyncio.sleep(self.monitor_interval)
                logger.info("monitor_command: %s", self.monitor_command)
                if ssh_tunnel:
                    returncode, stdout, stderr = await ssh_tunnel.run_remote(
                        self.monitor_command, raise_on_error=False
                    )
                else:
                    proc = await asyncio.create_subprocess_shell(
                        self.monitor_command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    await proc.wait()
                    stdout, stderr = await proc.communicate()
                    returncode = -1 if proc.returncode is None else proc.returncode
                is_error = returncode != 0
                if is_error:
                    logger.error(
                        "proc: %s returncode: %s", self.monitor_command, returncode
                    )
                    logger.error("stdout: %s", stdout)
                    logger.error("stderr: %s", stderr)
                    continue
                bjobs_output = BJobOutput.model_validate_json(stdout)
                logger.info("bjobs_output: %s", bjobs_output)
                if len(bjobs_output.RECORDS) == 0:
                    raise ValueError(f"failed to find the bsub job '{self.job_id}'")
                record = bjobs_output.RECORDS[0]
                logger.info("BSubLauncher.monitor_logs record: %s", record)
                if record.STAT not in ("PEND", "RUN"):
                    returncode = int(record.EXIT_CODE or "0", base=10)
                    # If stop was requested externally (e.g. retry_workload signalled us
                    # to stop before bkilling the job), treat this as a clean exit rather
                    # than a real failure — we may have just detected the bkill's exit code.
                    if self.stop_event.is_set():
                        returncode = 0
                    break
            except Exception as e:
                logger.error("failed to get the status of the job. error: %s", e)
                continue
        if self.stop_event.is_set():
            logger.warning(
                "[LSFBsubMonitor %s] stop event has been set, stopping bsub monitoring...",
                self.launch_id,
            )
            if returncode < 0:
                # Stop was requested externally before any terminal status was detected
                # (e.g. retry_workload signalled us to stop before bkilling the job).
                logger.info(
                    "[LSFBsubMonitor %s] Externally stopped before job completion; returning cleanly",
                    self.launch_id,
                )
                return
        logger.info(
            "[LSFBsubMonitor %s] BSubLauncher.monitor_logs returncode: %d",
            self.launch_id,
            returncode,
        )
        is_error = returncode != 0
        await asyncio.sleep(GBSERVER_MONITORING_GRACE_PERIOD)
        self.stop()
        if is_error:
            error_message = f"Job {self.job_id} failed with return code {returncode}"
            logger.error("[LSFBsubMonitor %s] %s", self.launch_id, error_message)

            # Check for transient LSF errors that should trigger a retry
            # Compute fallback path from log paths in case bjobs fails
            fallback_path = None
            log_path = self.lsf.get_log_path(self.launch_id, default="")
            if log_path:
                fallback_path = str(Path(log_path).parent / JOB_LOG_STDOUT_FILENAME)
            try:
                transient_error_content = await self._check_for_transient_lsf_error(
                    fallback_output_path=fallback_path
                )
            except ErrSSHConnectionError as e:
                logger.warning(
                    "[LSFBsubMonitor %s] SSH/connection error after retries: %s",
                    self.launch_id,
                    e,
                )
                transient_error_content = None
            if transient_error_content is not None:
                logger.warning(
                    "[LSFBsubMonitor %s] Emitting transient LSF error event for retry",
                    self.launch_id,
                )
                if self.event_queue is not None:
                    await self.event_queue.put(
                        BuildEvent(
                            run_metadata=self.entityrun_metadata,
                            type=BuildEventType.MESSAGE_EVENT,
                            payload=BuildEventMessagePayload(
                                level="ERROR",
                                msg=(
                                    f"LSF transient error: {ERR_LSF_CANNOT_OPEN_JOB_FILE}. "
                                    f"Job {self.job_id} failed with return code {returncode}"
                                ),
                            ),
                        )
                    )
                return

            # Publish a terminal failure event so RetryHandler can detect it
            # and raise WorkloadFailedException to fail the build.
            # Uses JSON format with state="Failed" so _is_terminal_failure_event matches.
            if self.event_queue is not None:
                payload = json.dumps(
                    {
                        "job_id": self.job_id,
                        "state": "Failed",
                        "error": error_message,
                    },
                    indent=4,
                )
                await self.event_queue.put(
                    BuildEvent(
                        run_metadata=self.entityrun_metadata,
                        type=BuildEventType.MESSAGE_EVENT,
                        payload=BuildEventMessagePayload(
                            level="ERROR",
                            msg=f"\n```json\n{payload}\n```\n",
                        ),
                    )
                )
            return
