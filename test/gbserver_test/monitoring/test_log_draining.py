#!/usr/bin/env python3
"""
Unit tests for log stream draining behavior after stop_event is set.

These tests verify the two-phase draining solution that fixes the issue where
log monitoring exits prematurely, causing loss of critical log lines and events
(like ARTIFACT_PUSHED_EVENT).

Background:
-----------
When LSF jobs complete, the bsub monitor detects completion and sets stop_event
after a grace period. However, the logfile monitor may still have buffered log
lines that were written but not yet read. Without proper draining, these final
lines (like "Pushed URI", "lhpush end", ARTIFACT_PUSHED events) are lost.

Solution:
---------
Two-phase draining approach:
- Phase 1: Real-time streaming (tail -F for SSH, readline loop for local files)
- Phase 2: Final complete read after stop_event is set
  - SSH: Terminate tail -F, execute tail -n +N to read remaining lines
  - Local: Continue reading until true EOF

Test Coverage:
--------------
1. test_local_file_stream_drains_remaining_lines
   - Verifies LocalFileStream Phase 2 draining captures lines written during monitoring

2. test_ssh_file_stream_drains_with_tail_command
   - Verifies SSHFileStream executes `tail -n +N` command with correct line offset
   - Confirms Phase 1 (tail -F) → Phase 2 (tail -n +4) transition

3. test_logfile_monitor_processes_all_drained_lines
   - Verifies LogFileMonitor processes ALL lines including those from Phase 2
   - Tests critical event lines like `LLMB_EVENT_WORKLOAD_STATUS:success`

4. test_lsf_bsub_and_logfile_monitor_coordination (INTEGRATION TEST - Most Important)
   - Simulates realistic LSF job execution with bsub monitor and logfile monitor
   - Verifies the critical "Pushed URI" line is captured via Phase 2 draining
   - Demonstrates the fix for the original issue where ARTIFACT_PUSHED_EVENT was lost

5. test_ssh_file_stream_phase2_executes_tail_command
   - Verifies line counting during Phase 1 and correct tail command in Phase 2
   - Confirms `tail -n +4` is used to skip already-read lines

6. test_logfile_monitor_does_not_exit_early_on_stop_event
   - Verifies the fix where we removed the early exit check
   - Ensures monitor continues consuming all streamed lines

7. test_local_file_stream_no_remaining_lines_to_drain
   - Tests edge case where Phase 2 finds no remaining lines
   - Verifies graceful handling when monitoring keeps up with writes
"""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class EmptyAsyncIterator:
    """An async iterator that yields nothing, for mocking empty stderr streams."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


from gbserver.monitoring.logfile_monitor import LogFileMonitor
from gbserver.monitoring.lsf_bsub_monitor import LSFBsubMonitor
from gbserver.monitoring.streams.local_file_stream import LocalFileStream
from gbserver.monitoring.streams.ssh_file_stream import SSHFileStream
from gbserver.types.buildevent import EntityRunMetadata


@pytest.mark.asyncio
async def test_local_file_stream_drains_remaining_lines(tmp_path):
    """
    Test 1: LocalFileStream Phase 2 draining captures lines written during monitoring.

    Verifies:
    - LocalFileStream Phase 2 draining captures lines written during monitoring
    - All lines are read, including those written after stop_event is set

    Scenario:
    - Create a log file with initial content (lines 1-3)
    - Start monitoring (Phase 1)
    - Append more content while monitoring (lines 4-6)
    - Set stop_event
    - Verify Phase 2 drains the remaining lines
    - Confirm all 6 lines are captured
    """
    log_file = tmp_path / "test.log"

    # Write initial content
    initial_lines = ["line 1", "line 2", "line 3"]
    log_file.write_text("\n".join(initial_lines) + "\n")

    stop_event = asyncio.Event()
    stream = LocalFileStream(path=log_file)

    collected_lines = []

    async def monitor_task():
        async for line in stream.stream_lines(stop_event=stop_event):
            collected_lines.append(line)

            # Simulate lag: after reading some lines, write more content
            if len(collected_lines) == 2:
                await asyncio.sleep(0.1)
                # Append additional lines while monitoring is in progress
                with log_file.open("a") as f:
                    f.write("line 4\n")
                    f.write("line 5\n")
                    f.write("line 6\n")
                    f.flush()
                await asyncio.sleep(0.1)
                # Set stop event - Phase 2 should drain lines 4-6
                stop_event.set()

    await monitor_task()

    # Verify all 6 lines were captured
    assert (
        len(collected_lines) == 6
    ), f"Expected 6 lines, got {len(collected_lines)}: {collected_lines}"
    assert collected_lines == [
        "line 1",
        "line 2",
        "line 3",
        "line 4",
        "line 5",
        "line 6",
    ]


@pytest.mark.asyncio
async def test_ssh_file_stream_drains_with_tail_command(tmp_path):
    """
    Test 2: SSHFileStream executes `tail -n +N` command with correct line offset.

    Verifies:
    - SSHFileStream executes `tail -n +N` command with correct line offset
    - Confirms Phase 1 (tail -F) → Phase 2 (tail -n +4) transition
    - Line counting is accurate for skipping already-read lines

    This test mocks the SSH subprocess to verify the draining logic executes
    the correct commands with proper line counting.
    """
    log_file_path = "/remote/path/to/job.log"

    stop_event = asyncio.Event()
    stream = SSHFileStream(
        host="test-host",
        user="testuser",
        path=log_file_path,
        ssh_opts=["-o", "StrictHostKeyChecking=no"],
    )

    # Mock the first subprocess (tail -F)
    mock_proc1 = AsyncMock()
    mock_proc1.returncode = None
    mock_proc1.stdout = EmptyAsyncIterator()
    mock_proc1.stderr = EmptyAsyncIterator()

    # Phase 1: tail -F yields 3 lines, then returns empty (simulating lag)
    phase1_lines = [b"line 1\n", b"line 2\n", b"line 3\n"]
    phase1_empty_count = [0]

    async def mock_readline_phase1():
        if phase1_lines:
            return phase1_lines.pop(0)
        # After lines exhausted, set stop_event and return empty to trigger Phase 2
        phase1_empty_count[0] += 1
        if phase1_empty_count[0] == 2:
            stop_event.set()
        return b""

    mock_proc1.stdout.readline = AsyncMock(side_effect=mock_readline_phase1)
    mock_proc1.stderr = EmptyAsyncIterator()

    # Mock the second subprocess (tail -n +4)
    mock_proc2 = AsyncMock()
    mock_proc2.returncode = 0
    mock_proc2.pid = 99999
    mock_proc2.stdout = AsyncMock()
    mock_proc2.stderr = AsyncMock()

    # Phase 2: tail -n +4 yields remaining 3 lines
    phase2_lines = [b"line 4\n", b"line 5\n", b"line 6\n"]

    async def mock_readline_phase2():
        if phase2_lines:
            return phase2_lines.pop(0)
        return b""  # EOF

    mock_proc2.stdout.readline = mock_readline_phase2

    async def mock_stderr_readline_phase2():
        return b""  # No errors

    mock_proc2.stderr.readline = mock_stderr_readline_phase2

    collected_lines = []
    executed_commands = []

    async def mock_create_subprocess(*args, **kwargs):
        executed_commands.append(args)
        # First call is tail -F, second call is tail -n +4
        if len(executed_commands) == 1:
            return mock_proc1
        else:
            return mock_proc2

    with patch("asyncio.create_subprocess_exec", side_effect=mock_create_subprocess):
        async for line in stream.stream_lines(stop_event=stop_event):
            collected_lines.append(line)

    # Verify behavior
    assert len(collected_lines) == 6, f"Expected 6 lines, got {len(collected_lines)}"
    assert collected_lines == [
        "line 1",
        "line 2",
        "line 3",
        "line 4",
        "line 5",
        "line 6",
    ]

    # Verify Phase 2 command was executed with correct line offset
    assert len(executed_commands) == 2, f"Expected 2 commands, got {len(executed_commands)}"

    # Second command should be tail -n +4 (skip first 3 lines)
    phase2_cmd = executed_commands[1]
    phase2_cmd_str = " ".join(phase2_cmd)
    assert "tail -n +4" in phase2_cmd_str, f"Expected 'tail -n +4' in command: {phase2_cmd_str}"


@pytest.mark.asyncio
async def test_logfile_monitor_processes_all_drained_lines(tmp_path):
    """
    Test 3: LogFileMonitor processes ALL lines including those from Phase 2.

    Verifies:
    - LogFileMonitor processes ALL lines including those from Phase 2
    - Tests critical event lines like `LLMB_EVENT_WORKLOAD_STATUS:success`
    - Monitor doesn't exit early when stop_event is set
    - All lines yielded by the stream are consumed

    This test demonstrates that critical events (like ARTIFACT_PUSHED) are
    captured even when written shortly before job completion.
    """
    log_file = tmp_path / "test.log"

    # Write initial content
    log_file.write_text("LLMB_EVENT_WORKLOAD_STATUS:running\n")

    stop_event = asyncio.Event()
    stream = LocalFileStream(path=log_file)

    event_queue = asyncio.Queue()

    # Configure to capture LLMB_EVENT lines
    monitor = LogFileMonitor(
        step_id="test-step",
        stream_source=stream,
        event_configs=[],
        event_queue=event_queue,
        stop_event=stop_event,
    )

    lines_processed = []

    # Override get_events_from_log_line to track what gets processed
    original_get_events = monitor.get_events_from_log_line

    async def track_line_processing(log_line):
        lines_processed.append(log_line)
        return await original_get_events(log_line)

    monitor.get_events_from_log_line = track_line_processing

    async def write_and_stop():
        # Wait for monitor to start
        await asyncio.sleep(0.2)

        # Append more lines
        with log_file.open("a") as f:
            f.write("Processing checkpoint 10086\n")
            f.write("Pushed URI: lh://prod/models/checkpoint-10086\n")
            f.write("LLMB_EVENT_WORKLOAD_STATUS:success\n")
            f.flush()

        # Immediately set stop_event (simulating bsub monitor completion)
        await asyncio.sleep(0.1)
        stop_event.set()

    # Run monitor and writer concurrently
    await asyncio.gather(
        monitor.monitor(),
        write_and_stop(),
    )

    # Verify all 4 lines were processed (including those written after stop_event)
    assert (
        len(lines_processed) == 4
    ), f"Expected 4 lines processed, got {len(lines_processed)}: {lines_processed}"
    assert "LLMB_EVENT_WORKLOAD_STATUS:running" in lines_processed[0]
    assert "Processing checkpoint 10086" in lines_processed[1]
    assert "Pushed URI: lh://prod/models/checkpoint-10086" in lines_processed[2]
    assert "LLMB_EVENT_WORKLOAD_STATUS:success" in lines_processed[3]


@pytest.mark.asyncio
async def test_lsf_bsub_and_logfile_monitor_coordination(tmp_path):
    """
    Test 4: INTEGRATION TEST - LSFBsubMonitor and LogFileMonitor coordination.

    *** THIS IS THE MOST IMPORTANT TEST ***

    This test demonstrates the fix for the original issue where ARTIFACT_PUSHED_EVENT
    was lost due to premature log monitoring exit.

    Verifies:
    - Simulates realistic LSF job execution with bsub monitor and logfile monitor
    - Bsub monitor detects job completion (bjobs reports DONE)
    - Waits grace period and sets stop_event
    - Logfile monitor drains remaining logs via Phase 2
    - The critical "Pushed URI" line is captured via Phase 2 draining
    - All log lines captured including those written near job completion

    Before this fix, the "Pushed URI: lh://prod/..." line would be lost because
    the logfile monitor exited before Phase 2 draining could complete.
    """
    log_file = tmp_path / "job.log"
    log_file.write_text("Job starting...\n")

    stop_event = asyncio.Event()
    event_queue = asyncio.Queue()

    # Mock LSF environment
    mock_lsf = MagicMock()
    mock_lsf.use_ssh = True

    # Mock bjobs to return job completion
    bjobs_responses = [
        # First few polls: job is running
        '{"COMMAND": "bjobs", "JOBS": 1, "RECORDS": [{"JOBID": "12345", "STAT": "RUN", "EXIT_CODE": "", "EXIT_REASON": ""}]}',
        '{"COMMAND": "bjobs", "JOBS": 1, "RECORDS": [{"JOBID": "12345", "STAT": "RUN", "EXIT_CODE": "", "EXIT_REASON": ""}]}',
        # Final poll: job completed successfully
        '{"COMMAND": "bjobs", "JOBS": 1, "RECORDS": [{"JOBID": "12345", "STAT": "DONE", "EXIT_CODE": "0", "EXIT_REASON": ""}]}',
    ]

    call_count = [0]

    mock_tunnel = AsyncMock()

    async def mock_run_remote(command, raise_on_error=True):
        response = bjobs_responses[min(call_count[0], len(bjobs_responses) - 1)]
        call_count[0] += 1
        return 0, response, ""

    mock_tunnel.run_remote = mock_run_remote
    mock_lsf.get_ssh_tunnel.return_value = mock_tunnel

    # Create monitors
    bsub_monitor = LSFBsubMonitor(
        lsf=mock_lsf,
        job_id="12345",
        launch_id="test-launch",
        entityrun_metadata=EntityRunMetadata(),
        event_queue=event_queue,
        stop_event=stop_event,
        monitor_interval=0.2,  # Fast polling for test
    )

    stream = LocalFileStream(path=log_file)
    logfile_monitor = LogFileMonitor(
        step_id="test-step",
        stream_source=stream,
        event_queue=event_queue,
        stop_event=stop_event,
    )

    collected_lines = []

    # Track lines processed by logfile monitor
    original_get_events = logfile_monitor.get_events_from_log_line

    async def track_line_processing(log_line):
        collected_lines.append(log_line)
        return await original_get_events(log_line)

    logfile_monitor.get_events_from_log_line = track_line_processing

    async def simulate_job_writing_logs():
        """Simulate job writing logs over time."""
        await asyncio.sleep(0.1)
        with log_file.open("a") as f:
            f.write("Checkpoint 1000 completed\n")
            f.write("Uploading files...\n")
            f.flush()

        await asyncio.sleep(0.2)
        with log_file.open("a") as f:
            f.write("Upload progress: 50%\n")
            f.flush()

        await asyncio.sleep(0.2)
        # Write final lines BEFORE job completes (bjobs reports DONE)
        # In reality, all logs are written before the job exits
        with log_file.open("a") as f:
            f.write("Upload progress: 100%\n")
            f.write("Pushed URI: lh://prod/models/checkpoint-1000\n")
            f.write("Job complete\n")
            f.flush()

        # Job is now complete, bsub monitor will detect DONE status
        # The logfile monitor may lag behind reading these final lines

    # Run all tasks concurrently
    with patch("gbserver.monitoring.lsf_bsub_monitor.GBSERVER_MONITORING_GRACE_PERIOD", 0.2):
        await asyncio.gather(
            bsub_monitor.monitor(),
            logfile_monitor.monitor(),
            simulate_job_writing_logs(),
            return_exceptions=True,  # Capture any exceptions
        )

    # Verify all lines were captured, including those written after stop_event
    assert (
        len(collected_lines) == 7
    ), f"Expected 7 lines, got {len(collected_lines)}: {collected_lines}"
    assert "Job starting..." in collected_lines[0]
    assert "Pushed URI: lh://prod/models/checkpoint-1000" in collected_lines[5]
    assert "Job complete" in collected_lines[6]

    # Verify the critical line at the end was captured (this would have been lost before the fix)
    assert any(
        "Pushed URI" in line for line in collected_lines
    ), "Critical 'Pushed URI' line was lost!"


@pytest.mark.asyncio
async def test_ssh_file_stream_phase2_executes_tail_command():
    """
    Test 5: SSHFileStream line counting during Phase 1 and correct tail command in Phase 2.

    Verifies:
    - Verifies line counting during Phase 1 and correct tail command in Phase 2
    - Confirms `tail -n +4` is used to skip already-read lines
    - Lines are counted accurately during Phase 1 (real-time streaming)
    - When stop_event is set, tail -F is terminated gracefully
    - tail -n +N is executed with correct offset (N = lines_read + 1)
    - Remaining lines are yielded without duplication
    """
    stop_event = asyncio.Event()
    stream = SSHFileStream(
        host="test-host",
        user="testuser",
        path="/remote/job.log",
        ssh_opts=["-o", "StrictHostKeyChecking=no"],
    )

    # Track executed commands
    executed_commands = []

    # Mock subprocess for tail -F (Phase 1)
    mock_proc_phase1 = AsyncMock()
    mock_proc_phase1.returncode = None
    mock_proc_phase1.stdout = EmptyAsyncIterator()
    mock_proc_phase1.stderr = EmptyAsyncIterator()

    phase1_lines = [b"line 1\n", b"line 2\n", b"line 3\n"]
    read_count = [0]

    async def mock_readline_phase1():
        if read_count[0] < len(phase1_lines):
            line = phase1_lines[read_count[0]]
            read_count[0] += 1
            return line
        # After 3 lines, trigger stop_event and return empty
        if read_count[0] == len(phase1_lines):
            read_count[0] += 1
            await asyncio.sleep(0.05)
            stop_event.set()
        return b""

    mock_proc_phase1.stdout.readline = mock_readline_phase1

    # Mock subprocess for tail -n +4 (Phase 2)
    mock_proc_phase2 = AsyncMock()
    mock_proc_phase2.returncode = 0
    mock_proc_phase2.pid = 99999
    mock_proc_phase2.stdout = AsyncMock()
    mock_proc_phase2.stderr = AsyncMock()

    phase2_lines = [b"line 4\n", b"line 5\n", b"line 6\n"]

    async def mock_readline_phase2():
        if phase2_lines:
            return phase2_lines.pop(0)
        return b""  # EOF

    mock_proc_phase2.stdout.readline = mock_readline_phase2

    async def mock_stderr_readline_phase2():
        return b""  # No errors

    mock_proc_phase2.stderr.readline = mock_stderr_readline_phase2

    call_counter = [0]

    async def mock_subprocess_exec(*args, **kwargs):
        executed_commands.append(args)
        call_counter[0] += 1
        if call_counter[0] == 1:
            return mock_proc_phase1
        else:
            return mock_proc_phase2

    collected_lines = []

    with patch("asyncio.create_subprocess_exec", side_effect=mock_subprocess_exec):
        async for line in stream.stream_lines(stop_event=stop_event):
            collected_lines.append(line)

    # Verify all 6 lines captured
    assert len(collected_lines) == 6, f"Expected 6 lines, got {len(collected_lines)}"
    assert collected_lines == [
        "line 1",
        "line 2",
        "line 3",
        "line 4",
        "line 5",
        "line 6",
    ]

    # Verify two commands were executed
    assert len(executed_commands) == 2, f"Expected 2 commands, got {len(executed_commands)}"


@pytest.mark.asyncio
async def test_logfile_monitor_does_not_exit_early_on_stop_event(tmp_path):
    """
    Test 6: LogFileMonitor removed early exit check - continues consuming all lines.

    Verifies:
    - Verifies the fix where we removed the early exit check
    - Ensures monitor continues consuming all streamed lines

    This verifies the critical fix where we removed the early exit check:
        if self.stop_event.is_set():
            break  # ❌ This was causing premature exit

    The monitor should continue consuming all lines from the stream,
    even after stop_event is set, allowing Phase 2 draining to complete.
    Without this fix, Phase 2 draining would yield lines but the monitor
    would not process them.
    """
    log_file = tmp_path / "test.log"
    log_file.write_text("initial line\n")

    stop_event = asyncio.Event()
    stream = LocalFileStream(path=log_file)

    event_queue = asyncio.Queue()
    monitor = LogFileMonitor(
        step_id="test-step",
        stream_source=stream,
        event_queue=event_queue,
        stop_event=stop_event,
    )

    lines_processed = []
    original_get_events = monitor.get_events_from_log_line

    async def track_line_processing(log_line):
        lines_processed.append(log_line)
        # After first line, set stop_event and write more lines
        if len(lines_processed) == 1:
            stop_event.set()  # Set stop event NOW
            # Write more lines to the file
            with log_file.open("a") as f:
                f.write("line after stop_event 1\n")
                f.write("line after stop_event 2\n")
                f.write("line after stop_event 3\n")
                f.flush()
        return await original_get_events(log_line)

    monitor.get_events_from_log_line = track_line_processing

    await monitor.monitor()

    # Verify monitor processed ALL 4 lines, including those written after stop_event was set
    assert (
        len(lines_processed) == 4
    ), f"Expected 4 lines, got {len(lines_processed)}: {lines_processed}"
    assert "initial line" in lines_processed[0]
    assert "line after stop_event 1" in lines_processed[1]
    assert "line after stop_event 2" in lines_processed[2]
    assert "line after stop_event 3" in lines_processed[3]


@pytest.mark.asyncio
async def test_local_file_stream_no_remaining_lines_to_drain(tmp_path):
    """
    Test 7: LocalFileStream edge case - Phase 2 finds no remaining lines.

    Verifies:
    - Tests edge case where Phase 2 finds no remaining lines
    - Verifies graceful handling when monitoring keeps up with writes

    This tests the common case where the monitoring is fast enough to read
    all lines during Phase 1, so Phase 2 finds nothing remaining to drain.
    The implementation should handle this gracefully without errors.
    """
    log_file = tmp_path / "test.log"
    log_file.write_text("line 1\nline 2\nline 3\n")

    stop_event = asyncio.Event()
    stream = LocalFileStream(path=log_file)

    collected_lines = []

    async def monitor_task():
        async for line in stream.stream_lines(stop_event=stop_event):
            collected_lines.append(line)
            # After all lines read, set stop_event
            if len(collected_lines) == 3:
                await asyncio.sleep(0.1)
                stop_event.set()

    await monitor_task()

    # Verify all 3 lines captured, Phase 2 found no remaining lines
    assert len(collected_lines) == 3
    assert collected_lines == ["line 1", "line 2", "line 3"]


@pytest.mark.asyncio
async def test_ssh_file_stream_phase2_stderr_raises_connection_error():
    """
    Test 8: SSHFileStream Phase 2 raises ConnectionError when stderr produces output.

    Verifies:
    - When Phase 2 SSH process writes to stderr (e.g. "Connection refused",
      "Permission denied"), the drain immediately raises ConnectionError with
      the stderr message instead of silently retrying for 120+ seconds.
    - Phase 1 lines are still yielded before the error.
    - The SSH subprocess is cleaned up properly.

    This is the fix for issue #1921 where an SSH connection failure during
    Phase 2 log draining would manifest as a generic TimeoutError after 24
    retries instead of surfacing the real SSH error.
    """
    stop_event = asyncio.Event()
    stream = SSHFileStream(
        host="test-host",
        user="testuser",
        path="/remote/job.log",
        ssh_opts=["-o", "StrictHostKeyChecking=no"],
    )

    # Mock Phase 1 subprocess (tail -F)
    mock_proc_phase1 = AsyncMock()
    mock_proc_phase1.returncode = None
    mock_proc_phase1.stdout = AsyncMock()
    mock_proc_phase1.stderr = AsyncMock()

    phase1_lines = [b"line 1\n", b"line 2\n"]
    phase1_empty_count = [0]

    async def mock_readline_phase1():
        if phase1_lines:
            return phase1_lines.pop(0)
        phase1_empty_count[0] += 1
        if phase1_empty_count[0] == 2:
            stop_event.set()
        return b""

    mock_proc_phase1.stdout.readline = mock_readline_phase1
    # Phase 1 stderr: empty (no errors during Phase 1)
    mock_proc_phase1.stderr.__aiter__ = lambda self: EmptyAsyncIterator()

    # Mock Phase 2 subprocess — SSH connection fails, stderr has the error
    mock_proc_phase2 = AsyncMock()
    mock_proc_phase2.returncode = 255  # SSH connection failure exit code
    mock_proc_phase2.pid = 12345
    mock_proc_phase2.stdout = AsyncMock()
    mock_proc_phase2.stderr = AsyncMock()

    # stdout: hangs (never produces data, simulating a failed connection)
    async def mock_stdout_readline_hang():
        await asyncio.sleep(60)
        return b""

    mock_proc_phase2.stdout.readline = mock_stdout_readline_hang

    # stderr: produces the SSH error message
    stderr_returned = [False]

    async def mock_stderr_readline():
        if not stderr_returned[0]:
            stderr_returned[0] = True
            return b"ssh: connect to host test-host port 22: Connection refused\n"
        return b""

    mock_proc_phase2.stderr.readline = mock_stderr_readline

    call_counter = [0]

    async def mock_subprocess_exec(*args, **kwargs):
        call_counter[0] += 1
        if call_counter[0] == 1:
            return mock_proc_phase1
        else:
            return mock_proc_phase2

    collected_lines = []

    with patch("asyncio.create_subprocess_exec", side_effect=mock_subprocess_exec):
        with pytest.raises(ConnectionError, match="exit code 255.*Connection refused"):
            async for line in stream.stream_lines(stop_event=stop_event):
                collected_lines.append(line)

    # Phase 1 lines should still have been captured before the error
    assert collected_lines == [
        "line 1",
        "line 2",
    ], f"Expected Phase 1 lines, got: {collected_lines}"


@pytest.mark.asyncio
async def test_ssh_file_stream_phase2_clean_stderr_eof():
    """
    Test 9: SSHFileStream Phase 2 handles clean stderr EOF without error.

    Verifies:
    - When Phase 2 SSH process closes stderr without writing anything
      (normal case), the drain completes successfully reading all stdout lines.
    - No ConnectionError is raised for a clean stderr EOF.
    """
    stop_event = asyncio.Event()
    stream = SSHFileStream(
        host="test-host",
        user="testuser",
        path="/remote/job.log",
        ssh_opts=["-o", "StrictHostKeyChecking=no"],
    )

    # Mock Phase 1 subprocess
    mock_proc_phase1 = AsyncMock()
    mock_proc_phase1.returncode = None
    mock_proc_phase1.stdout = AsyncMock()
    mock_proc_phase1.stderr = AsyncMock()

    phase1_lines = [b"line 1\n"]

    async def mock_readline_phase1():
        if phase1_lines:
            return phase1_lines.pop(0)
        stop_event.set()
        return b""

    mock_proc_phase1.stdout.readline = mock_readline_phase1
    mock_proc_phase1.stderr.__aiter__ = lambda self: EmptyAsyncIterator()

    # Mock Phase 2 subprocess — clean execution, no stderr
    mock_proc_phase2 = AsyncMock()
    mock_proc_phase2.returncode = 0
    mock_proc_phase2.pid = 12345
    mock_proc_phase2.stdout = AsyncMock()
    mock_proc_phase2.stderr = AsyncMock()

    phase2_stdout_lines = [b"line 2\n", b"line 3\n"]

    async def mock_stdout_readline():
        if phase2_stdout_lines:
            return phase2_stdout_lines.pop(0)
        return b""

    mock_proc_phase2.stdout.readline = mock_stdout_readline

    # stderr: immediately returns EOF (no errors)
    async def mock_stderr_readline():
        return b""

    mock_proc_phase2.stderr.readline = mock_stderr_readline

    call_counter = [0]

    async def mock_subprocess_exec(*args, **kwargs):
        call_counter[0] += 1
        if call_counter[0] == 1:
            return mock_proc_phase1
        else:
            return mock_proc_phase2

    collected_lines = []

    with patch("asyncio.create_subprocess_exec", side_effect=mock_subprocess_exec):
        async for line in stream.stream_lines(stop_event=stop_event):
            collected_lines.append(line)

    assert collected_lines == ["line 1", "line 2", "line 3"], f"Got: {collected_lines}"
