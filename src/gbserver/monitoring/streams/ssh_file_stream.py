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
Stream from a remote file using ssh.
"""

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, List, Optional, Self

from gbserver.monitoring.streams.log_stream_base import LogStreamSource, RetryMixin
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# SSH timeout settings to prevent hanging connections.
# ConnectTimeout: fail fast if the SSH connection cannot be established (seconds).
# ServerAliveInterval: send keepalive every N seconds to detect dead connections.
# ServerAliveCountMax: disconnect after N missed keepalives (total dead time = interval * count).
SSH_CONNECT_TIMEOUT = 10
SSH_SERVER_ALIVE_INTERVAL = 5
SSH_SERVER_ALIVE_COUNT_MAX = 3


class SSHFileStream(LogStreamSource, RetryMixin):
    """Stream lines from a remote file via SSH."""

    def __init__(
        self: Self,
        host: str,
        user: Optional[str],
        path: str | Path,
        ssh_opts: Optional[List[str]] = None,
        **retry_kwargs: Any,
    ) -> None:
        RetryMixin.__init__(self, **retry_kwargs)
        self.host = host
        self.user = user
        self.path = str(path)
        self.ssh_opts = ssh_opts or []

    def _build_ssh_cmd(self: Self, target: str, remote_cmd: str) -> List[str]:
        """Build an SSH command with connection timeout and keepalive settings.

        These options ensure SSH fails fast on connection problems rather than
        hanging indefinitely, which was the root cause of intermittent build
        failures where the log drain phase would time out after 120+ seconds.
        """
        return [
            "ssh",
            "-o",
            f"ConnectTimeout={SSH_CONNECT_TIMEOUT}",
            "-o",
            f"ServerAliveInterval={SSH_SERVER_ALIVE_INTERVAL}",
            "-o",
            f"ServerAliveCountMax={SSH_SERVER_ALIVE_COUNT_MAX}",
            *self.ssh_opts,
            target,
            remote_cmd,
        ]

    async def _read_phase_one(
        self: Self,
        target: str,
        path_quoted: str,
        stop_event: asyncio.Event,
    ) -> AsyncIterator[str]:
        """Phase 1: Stream lines in real-time while job is running."""
        cmd = self._build_ssh_cmd(
            target,
            f"while [ ! -f {path_quoted} ]; do sleep 1; done; "
            # + f"cat {path_quoted}; tail -n0 -F {path_quoted}",
            + f"tail -n +1 -F {path_quoted}",
        )

        logger.info("[SSHFileStream] Phase 1 Running command: %s", " ".join(cmd))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def log_stderr():
            if proc.stderr:
                async for line in proc.stderr:
                    logger.warning("[SSHFileStream] stderr: %s", line.decode(errors="ignore"))

        stderr_task = asyncio.create_task(log_stderr())

        try:
            assert proc.stdout
            while not stop_event.is_set():
                try:
                    # TODO: this should be refactored to be a function to be used here and in the 2nd phase
                    raw = await asyncio.wait_for(proc.stdout.readline(), timeout=5.0)
                    if not raw:  # EOF is permanent for a process stream (process ended), so we exit
                        if proc.returncode is not None:
                            logger.info(
                                "[SSHFileStream] The tail process exited with returncode %d",
                                proc.returncode,
                            )
                            break
                        await asyncio.sleep(
                            0.1
                        )  # prevent busy loop that would starve the main event loop
                        continue
                    yield raw.decode(errors="ignore").rstrip()
                except asyncio.TimeoutError:
                    # No data within timeout, loop will check stop_event
                    continue
        finally:
            stderr_task.cancel()
            try:
                await stderr_task
            except asyncio.CancelledError:
                pass

            if proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning(
                        "[SSHFileStream] Tail process didn't terminate gracefully, killing it"
                    )
                    try:
                        proc.kill()
                        await proc.wait()
                    except ProcessLookupError:
                        pass  # Process already exited between terminate and kill

    async def _read_phase_two(
        self: Self,
        target: str,
        path_quoted: str,
        lines_read: int,
        abort_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[str]:
        """Phase 2: stop_event is set, job is complete, drain remaining logs.

        Monitors both stdout and stderr concurrently. If stderr produces output
        (e.g. SSH connection error, remote command failure), we raise immediately
        so the build reports the real error instead of timing out silently.

        If abort_event is set before or during the drain, it is skipped so that
        artifact events from a retried job are not emitted.
        """

        logger.info(
            "[SSHFileStream] Entering Phase 2 draining after reading %d lines from %s",
            lines_read,
            self.path,
        )

        # Read the complete file one final time to capture any remaining content
        # Use tail -n +N to skip lines we already read (N+1 means start from line N+1)
        if lines_read > 0:
            final_cmd_str = f"tail -n +{lines_read + 1} {path_quoted}"
        else:
            # If we haven't read any lines yet, read the whole file
            final_cmd_str = f"cat {path_quoted}"

        final_cmd = self._build_ssh_cmd(target, final_cmd_str)

        logger.info(
            "[SSHFileStream] Reading final content with command: %s",
            " ".join(final_cmd),
        )

        final_proc = await asyncio.create_subprocess_exec(
            *final_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            assert final_proc.stdout
            assert final_proc.stderr
            remaining_lines = 0

            # Monitor both stdout and stderr concurrently.
            # stderr output from SSH indicates a real problem (connection refused,
            # auth failure, host unreachable, etc.) — surface it immediately.
            out_task = asyncio.create_task(final_proc.stdout.readline())
            err_task = asyncio.create_task(final_proc.stderr.readline())
            stderr_lines: list[str] = []

            while True:
                if abort_event and abort_event.is_set():
                    out_task.cancel()
                    err_task.cancel()
                    break

                # Wait for either stdout or stderr to produce data, with a timeout
                pending_tasks = set()
                if not out_task.done():
                    pending_tasks.add(out_task)
                if not err_task.done():
                    pending_tasks.add(err_task)
                if not pending_tasks:
                    break

                done, _ = await asyncio.wait(
                    pending_tasks,
                    timeout=5.0,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if not done:
                    # Timeout — neither stream produced data.
                    # Check if the process has exited.
                    if final_proc.returncode is not None:
                        logger.warning(
                            "[SSHFileStream] Drain process for %s exited with code %d "
                            "but streams not closed. Breaking.",
                            self.path,
                            final_proc.returncode,
                        )
                        out_task.cancel()
                        err_task.cancel()
                        break
                    logger.warning(
                        "[SSHFileStream] Timeout waiting for drain output from %s "
                        "(pid=%d, %d lines drained so far)",
                        self.path,
                        final_proc.pid,
                        remaining_lines,
                    )
                    continue

                # Check stderr — collect lines for diagnostics.
                # SSH writes benign warnings to stderr (e.g. "Warning: Permanently
                # added ... to the list of known hosts.") so we don't raise immediately.
                # Instead, we collect stderr and raise only if the process exits non-zero.
                if err_task in done:
                    err_line = err_task.result()
                    if err_line:
                        stderr_msg = err_line.decode(errors="ignore").strip()
                        stderr_lines.append(stderr_msg)
                        logger.warning(
                            "[SSHFileStream] stderr from drain process for %s: %s",
                            self.path,
                            stderr_msg,
                        )
                        # Schedule next stderr read
                        err_task = asyncio.create_task(final_proc.stderr.readline())
                    # else: stderr EOF — normal, stop monitoring it

                # Check stdout
                if out_task in done:
                    out_line = out_task.result()
                    if not out_line:
                        # stdout EOF — done draining
                        err_task.cancel()
                        break
                    yield out_line.decode(errors="ignore").rstrip()
                    lines_read += 1
                    remaining_lines += 1
                    # Schedule next stdout read
                    out_task = asyncio.create_task(final_proc.stdout.readline())

            try:
                await asyncio.wait_for(final_proc.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.error(
                    "[SSHFileStream] Drain process did not exit within timeout for %s",
                    self.path,
                )

            # Now check: if the process exited with an error AND stderr had output,
            # that's a real SSH/remote failure — raise with the actual error message.
            if final_proc.returncode and final_proc.returncode != 0 and stderr_lines:
                stderr_output = "; ".join(stderr_lines)
                raise ConnectionError(
                    f"SSH drain failed for {self.path} (exit code {final_proc.returncode}): "
                    f"{stderr_output}"
                )

            logger.info(
                "[SSHFileStream] Finished draining %s: total_lines=%d, "
                "remaining_lines=%d, exit_code=%s",
                self.path,
                lines_read,
                remaining_lines,
                final_proc.returncode,
            )
        except ConnectionError:
            raise
        finally:
            if final_proc.returncode is None:
                final_proc.kill()
                try:
                    await asyncio.wait_for(final_proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass

    async def _stream_once(
        self: Self,
        stop_event: Optional[asyncio.Event],
        abort_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[str]:
        if stop_event is None:
            stop_event = asyncio.Event()

        target = f"{self.user + '@' if self.user else ''}{self.host}"
        path_quoted = '"' + self.path + '"'
        lines_read = 0

        async for line in self._read_phase_one(target, path_quoted, stop_event):
            lines_read += 1
            yield line

        logger.info("Preparing for phase 2 by sleeping 5")
        await asyncio.sleep(5)

        async for line in self._read_phase_two(target, path_quoted, lines_read, abort_event):
            yield line

    async def stream_lines(
        self: Self,
        stop_event: Optional[asyncio.Event] = None,
        abort_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[str]:
        async for line in self._stream_once(stop_event, abort_event):
            yield line

    def __str__(self: Self) -> str:
        user_str = f"{self.user}@" if self.user else ""
        ssh_opts_str = " ".join(self.ssh_opts) if self.ssh_opts else "-"
        return (
            f"SSHFileStream(target='{user_str}{self.host}', "
            + f"path='{self.path}', ssh_opts='{ssh_opts_str}')"
        )
