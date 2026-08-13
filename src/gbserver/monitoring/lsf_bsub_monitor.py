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
import re
import shlex
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Dict, Optional, Self

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
    BuildEventStatusPayload,
    BuildEventType,
    EntityRunMetadata,
)
from gbserver.types.constants import GBSERVER_MONITORING_GRACE_PERIOD
from gbserver.types.errors import (
    ERR_LSF_CANNOT_OPEN_JOB_FILE,
    ErrLSFCannotOpenJobFile,
    ErrSSHConnectionError,
)
from gbserver.types.status import Status
from gbserver.utils.logger import get_logger
from gbserver.utils.ssh_tunnel import SshTunnel
from gbserver.utils.utils import cmd_safe_join

logger = get_logger(__name__)

JOB_LOG_STDOUT_FILENAME = "job_log.out"


class LsfStateClass(StrEnum):
    """How a native LSF ``STAT`` maps onto the terminal decision for a step.

    Deliberately not a gbserver ``Status``: monitors never set status (they emit
    events, and the step's status follows from whether anything raised). This
    only answers "is the job over, and did it pass".
    """

    ACTIVE = auto()  # the job is alive; keep polling
    SUCCEEDED = auto()  # terminal -> step SUCCESS
    FAILED = auto()  # terminal -> step FAILED


# Native LSF STAT -> terminal decision. Source: `man bjobs` on IBM Spectrum LSF
# Advanced 10.1.0.15. Any STAT absent from this table is treated as ACTIVE; see
# LSFBsubMonitor._classify_state for why that direction is the safe one.
LSF_STATE_CLASS: Dict[str, LsfStateClass] = {
    # -- not terminal: the job has NOT finished, so keep polling ---------------
    "PEND": LsfStateClass.ACTIVE,  # queued; PEND_REASON says why
    "FWD_PEND": LsfStateClass.ACTIVE,  # pending forward to a remote cluster
    "WAIT": LsfStateClass.ACTIVE,  # chunk-job member waiting to start
    "PROV": LsfStateClass.ACTIVE,  # execution host still provisioning
    "RUN": LsfStateClass.ACTIVE,
    "PSUSP": LsfStateClass.ACTIVE,  # suspended while pending (bstop / bsub -H)
    "SSUSP": LsfStateClass.ACTIVE,  # suspended by LSF (load, preemption)
    "USUSP": LsfStateClass.ACTIVE,  # suspended by user/admin after dispatch
    "UNKWN": LsfStateClass.ACTIVE,  # mbatchd lost contact with sbatchd; transient
    # -- terminal, succeeded --------------------------------------------------
    "DONE": LsfStateClass.SUCCEEDED,
    "POST_DONE": LsfStateClass.SUCCEEDED,  # post-exec completed successfully
    # -- terminal, failed ----------------------------------------------------
    # EXIT_CODE/EXIT_REASON only *describe* these states; they never decide them.
    "EXIT": LsfStateClass.FAILED,
    "ZOMBI": LsfStateClass.FAILED,  # job lost with an unreachable exec host
    "POST_ERR": LsfStateClass.FAILED,  # post-exec failed
}

# Non-terminal LSF state -> the granite.build step status it should report.
#
# Only RUN maps to RUNNING. A job queued, held, or suspended in LSF is NOT
# running, and reporting it as RUNNING is the false signal this mapping exists to
# remove: the step status should never claim progress the scheduler isn't making.
# granite.build has no SUSPENDED status, so the suspend states collapse onto
# PENDING and the native state + reason in the event stream carries the precision
# the enum cannot.
LSF_ACTIVE_STATE_TO_GB_STATUS: Dict[str, Status] = {
    "RUN": Status.RUNNING,
    "PEND": Status.PENDING,  # queued for dispatch
    "FWD_PEND": Status.PENDING,  # pending forward to a remote cluster
    "WAIT": Status.PENDING,  # chunk-job member waiting to start
    "PROV": Status.PENDING,  # execution host still provisioning
    "PSUSP": Status.PENDING,  # suspended while pending; never dispatched
    "SSUSP": Status.PENDING,  # suspended by LSF after dispatch
    "USUSP": Status.PENDING,  # suspended by user/admin after dispatch
    "UNKWN": Status.PENDING,  # exec host unreachable; progress unknown
}

# States for which PEND_REASON is meaningful and worth re-emitting when it changes.
LSF_STATES_WITH_PEND_REASON = frozenset(
    {"PEND", "FWD_PEND", "WAIT", "PROV", "PSUSP", "SSUSP", "USUSP"}
)

# A job LSF calls failed must never report success, even when EXIT_CODE is blank:
# on a real cluster 103,256 of 315,504 EXIT jobs report EXIT_CODE="" (102,208 of
# those TERM_OWNER, i.e. killed), and every DONE job reports "" as well -- so
# EXIT_CODE cannot decide the verdict.
LSF_UNSPECIFIED_FAILURE_RETURNCODE = 1

_PEND_REASON_VOLATILE_NUMBERS = re.compile(r"\d+")


def _pend_reason_key(reason: str) -> str:
    """Collapse the live counters LSF embeds in PEND_REASON.

    LSF re-renders the numbers every scheduling cycle, e.g.
    ``"...(Resource: ngpus_physical): 340 hosts;"`` becomes ``"... 339 hosts;"``.
    Comparing raw text would emit an event on every poll for a long PEND, so we
    dedupe on a digit-collapsed key but always display the raw reason.
    """
    return _PEND_REASON_VOLATILE_NUMBERS.sub("#", (reason or "").strip())


class BJobRecord(BaseModel):
    """A bjob record.

    Every field is defaulted on purpose. This model is validated inside the poll
    loop's broad ``except Exception: continue``, so a ValidationError does not
    surface as an error -- it silently costs a poll cycle, and a *persistently*
    invalid record becomes an infinite loop. Defaults degrade to "unrecognized
    STAT" (which warns and keeps polling) instead, and keep payloads from callers
    that predate a new field valid.
    """

    JOBID: str = ""
    STAT: str = ""
    EXIT_CODE: str = ""
    EXIT_REASON: str = ""
    # `bjobs -o pend_reason`, LSF 10.1. NOTE: `susp_reason` does NOT exist on
    # this LSF ("not a valid field name"); PEND_REASON already covers
    # suspended-while-pending, e.g. "Job was suspended by the user while pending;".
    PEND_REASON: str = ""


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
        # On-change-only state surfacing. None means "nothing emitted yet", which
        # is distinct from "" (a state that carries no reason).
        self._last_lsf_state: Optional[str] = None
        self._last_pend_reason_key: Optional[str] = None
        self._last_gb_status: Optional[Status] = None
        # The record we broke the poll loop on, so the failure message can name
        # the native LSF state and exit reason.
        self._terminal_record: Optional[BJobRecord] = None
        # True iff this run handed an error event (transient or terminal) to the
        # RetryHandler for adjudication. monitor_bsub_monitor reads this to know
        # it must wait for that out-of-band decision before deciding success,
        # rather than racing it (which would orphan a relaunched job).
        self.emitted_error_event = False

    def _classify_state(self: Self, stat: str) -> LsfStateClass:
        """Classify a native LSF STAT. Never raises.

        An unrecognized STAT is treated as ACTIVE (keep polling), not terminal.
        Assuming "anything I don't recognize means the job is over" is precisely
        how a suspended job came to be reported as a successful build: those
        states carry EXIT_CODE="", so the old code scored them as returncode 0.
        Failing safe means the worst case is a visible hang rather than a silent
        wrong SUCCESS, and it survives an LSF upgrade that adds a state.
        """
        state = (stat or "").strip().upper()
        state_class = LSF_STATE_CLASS.get(state)
        if state_class is None:
            logger.warning(
                "[LSFBsubMonitor %s] job %s reported unrecognized LSF STAT %r; "
                "treating as non-terminal and continuing to poll",
                self.launch_id,
                self.job_id,
                stat,
            )
            return LsfStateClass.ACTIVE
        return state_class

    @staticmethod
    def _terminal_returncode(record: BJobRecord, state_class: LsfStateClass) -> int:
        """Return a process-style returncode for a terminal LSF state. Never raises.

        STAT decides pass/fail; EXIT_CODE only refines a failure's code. EXIT_CODE
        cannot decide the verdict -- it is empty on every DONE job and on a third
        of EXIT jobs -- so a failure with a blank or zero code still returns
        non-zero.
        """
        if state_class is LsfStateClass.SUCCEEDED:
            return 0
        try:
            # `bjobs -json` renders unset as ""; the -noheader table form uses "-".
            code = int((record.EXIT_CODE or "").strip(), base=10)
        except ValueError:
            code = 0
        return code if code != 0 else LSF_UNSPECIFIED_FAILURE_RETURNCODE

    @staticmethod
    def _exit_detail(record: BJobRecord) -> str:
        """Render EXIT_CODE/EXIT_REASON as supplementary detail (may be empty)."""
        bits = []
        if (record.EXIT_CODE or "").strip() not in ("", "-"):
            bits.append(f"exit_code={record.EXIT_CODE.strip()}")
        if (record.EXIT_REASON or "").strip() not in ("", "-"):
            bits.append(f"exit_reason={record.EXIT_REASON.strip()}")
        return f" ({', '.join(bits)})" if bits else ""

    def _format_terminal_failure_detail(self: Self) -> str:
        """Name the native LSF state and exit reason on a failure.

        EXIT_REASON has always been fetched and never read. It is the only field
        that separates TERM_MEMLIMIT / TERM_RUNLIMIT / TERM_OWNER / TERM_ADMIN --
        i.e. "ask for more memory" from "ask for more time" from "someone killed
        it" from "the admin drained the host".
        """
        record = self._terminal_record
        if record is None:
            return ""
        bits = [f"LSF state {(record.STAT or '').strip().upper()}"]
        if (record.EXIT_REASON or "").strip() not in ("", "-"):
            bits.append(f"exit_reason={record.EXIT_REASON.strip()}")
        return f" ({', '.join(bits)})"

    def _format_lsf_state_change(
        self: Self, record: BJobRecord, state_class: LsfStateClass
    ) -> str:
        """Build the one-line message for an LSF state or reason change.

        Deliberately a single plain line, NOT a fenced ```json block: the CLI
        strips backticks and drops this string into a two-column
        ``tabulate(tablefmt="plain")`` table (the `gb build status --show-events`
        build history), which a multi-line blob would wreck. Carrying no json
        fence also means ``RetryHandler._is_terminal_failure_event`` cannot match
        an informational event, so surfacing state can never fail the build.
        """
        state = (record.STAT or "").strip().upper() or "UNKNOWN"
        reason = (record.PEND_REASON or "").strip().rstrip(";")
        detail = f" - {reason}" if reason else ""
        previous = self._last_lsf_state
        if previous is None or previous == state:
            # First sighting, or the state held and only the reason moved on --
            # rendering "PEND -> PEND" for a reason-only change reads as a bug.
            transition = f"LSF job {self.job_id} is {state}"
        else:
            transition = f"LSF job {self.job_id}: {previous} -> {state}"

        if state in ("PEND", "FWD_PEND", "WAIT", "PROV"):
            return f"⏳ {transition}{detail}"
        if state in ("PSUSP", "SSUSP", "USUSP"):
            return (
                f"⏸️ {transition}{detail}. The job is suspended, not finished; "
                f"granite.build will keep monitoring until it resumes or exits "
                f"(bresume {self.job_id} to resume, or cancel the build)."
            )
        if state == "UNKWN":
            return (
                f"❓ {transition}. LSF has lost contact with the execution host; "
                f"this is usually transient and granite.build will keep monitoring."
            )
        if state == "RUN":
            return f"⚡ {transition}"
        if state_class is LsfStateClass.SUCCEEDED:
            return f"✅ {transition}"
        if state_class is LsfStateClass.FAILED:
            return f"❌ {transition}{self._exit_detail(record)}"
        return (
            f"⚠️ {transition} - granite.build does not recognize this LSF state "
            f"and will keep monitoring rather than assume the job is over."
        )

    async def _publish_gb_status(self: Self, state: str, msg: str) -> None:
        """Report the granite.build step status implied by a non-terminal LSF state.

        A job queued or suspended in LSF is not running, so the step must not sit
        at RUNNING while it waits -- that is the false signal this mapping exists
        to remove. Emitting a STATUS_EVENT lets BuildRunner write the mapped
        status (and the reason, as status_msg) onto the step record, so
        `gb build status` shows PENDING with the pend reason attached.

        Only fires when the mapped status actually changes, which is far rarer
        than an LSF state change (PEND -> PSUSP is the same gb status). Terminal
        states are left alone: SUCCESS/FAILED are owned by the existing
        returncode path and by Run.update_status.
        """
        gb_status = LSF_ACTIVE_STATE_TO_GB_STATUS.get(state)
        if gb_status is None:
            # Terminal, or a state we do not recognize. In the unrecognized case
            # we genuinely do not know whether the job is progressing, so leave
            # the step status untouched rather than assert something false.
            return
        if gb_status is self._last_gb_status:
            return
        if self.event_queue is not None:
            await self.event_queue.put(
                BuildEvent(
                    run_metadata=self.entityrun_metadata,
                    type=BuildEventType.STATUS_EVENT,
                    payload=BuildEventStatusPayload(status=gb_status, msg=msg),
                )
            )
        logger.info(
            "[LSFBsubMonitor %s] LSF %s -> granite.build step status %s",
            self.launch_id,
            state,
            gb_status,
        )
        self._last_gb_status = gb_status

    async def _publish_lsf_state_change(
        self: Self, record: BJobRecord, state_class: LsfStateClass
    ) -> None:
        """Emit a MESSAGE_EVENT when the LSF state or its reason changes.

        On change only -- not once, and not on every poll: a job can sit in PEND
        for hours at a 5s cadence.

        Best effort. This runs inside the poll loop's broad ``except Exception``,
        so it swallows its own errors rather than risk delaying or hiding the
        terminal verdict below it.
        """
        try:
            state = (record.STAT or "").strip().upper()
            reason_key = _pend_reason_key(record.PEND_REASON)
            state_changed = state != self._last_lsf_state
            reason_changed = (
                state in LSF_STATES_WITH_PEND_REASON
                and reason_key != self._last_pend_reason_key
            )
            if not state_changed and not reason_changed:
                return

            msg = self._format_lsf_state_change(record, state_class)
            if state_class is LsfStateClass.FAILED:
                level = "ERROR"
            elif state not in LSF_STATE_CLASS or state in (
                "PSUSP",
                "SSUSP",
                "USUSP",
                "UNKWN",
            ):
                level = "WARNING"
            else:
                level = "INFO"

            if self.event_queue is not None:
                await self.event_queue.put(
                    BuildEvent(
                        run_metadata=self.entityrun_metadata,
                        type=BuildEventType.MESSAGE_EVENT,
                        payload=BuildEventMessagePayload(level=level, msg=msg),
                    )
                )
            logger.info("[LSFBsubMonitor %s] LSF state change: %s", self.launch_id, msg)
            self._last_lsf_state = state
            self._last_pend_reason_key = reason_key
            await self._publish_gb_status(state, msg)
        except Exception as e:  # messaging must never block the verdict
            logger.warning(
                "[LSFBsubMonitor %s] failed to publish LSF state change: %s",
                self.launch_id,
                e,
                exc_info=True,
            )

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
                "-a",  # also return a just-finished job's record
                "-o",
                '"jobid stat exit_code exit_reason pend_reason"',
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

                state_class = self._classify_state(record.STAT)

                # Surface the native LSF state -- and, while the job is queued or
                # suspended, WHY -- on every change.
                await self._publish_lsf_state_change(record, state_class)

                if state_class is LsfStateClass.ACTIVE:
                    # PEND / RUN / PSUSP / SSUSP / USUSP / UNKWN / WAIT / PROV /
                    # FWD_PEND, plus any STAT this gbserver does not recognize.
                    continue

                self._terminal_record = record
                returncode = self._terminal_returncode(record, state_class)
                # If stop was requested externally (e.g. retry_workload signalled us
                # to stop before bkilling the job), treat this as a clean exit rather
                # than a real failure — we may have just detected the bkill's exit code.
                if self.stop_event.is_set():
                    returncode = 0
                break
            except Exception as e:
                logger.error(
                    "failed to get the status of the job. error: %s", e, exc_info=True
                )
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
            error_message = (
                f"Job {self.job_id} failed with return code {returncode}"
                f"{self._format_terminal_failure_detail()}"
            )
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
                    self.emitted_error_event = True
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
                self.emitted_error_event = True
            return
