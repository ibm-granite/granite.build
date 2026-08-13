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

"""State-machine tests for LSFBsubMonitor.

These drive a real ``LSFBsubMonitor`` against a scripted sequence of ``bjobs``
JSON responses (the same harness idiom as
``test/unit/monitoring/test_log_draining.py``), which lets us replay LSF states
that are awkward or impossible to produce on demand against a real cluster.

Every test goes through :func:`_drive`, which bounds the monitor with
``asyncio.wait_for``. That is deliberate: the monitor's poll loop wraps its whole
body in ``except Exception: continue``, so a classification bug becomes an
*infinite loop* rather than an error. The bounded wait converts that into a
deterministic failure -- and for the non-terminal states, "it keeps polling" is
itself the assertion.
"""

import asyncio
import contextlib
import json
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.monitoring.lsf_bsub_monitor import (
    LSF_ACTIVE_STATE_TO_GB_STATUS,
    LSF_STATE_CLASS,
    BJobRecord,
    LSFBsubMonitor,
    LsfStateClass,
)
from gbserver.resilience.retry_handler import RetryHandler
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventMessagePayload,
    BuildEventType,
    EntityRunMetadata,
)
from gbserver.types.status import Status

# Short intervals keep the suite fast; the grace period is patched per test.
MONITOR_INTERVAL = 0.02
GRACE = 0.02
# Generous enough to absorb scheduler jitter, small enough to fail fast.
TERMINATE_BUDGET = 3.0
NON_TERMINAL_BUDGET = 0.4


def _bjobs_json(
    stat: str,
    exit_code: str = "",
    exit_reason: str = "",
    pend_reason: str = "",
    jobid: str = "12345",
) -> str:
    """One ``bjobs -o ... -json`` response carrying a single record."""
    return json.dumps(
        {
            "COMMAND": "bjobs",
            "JOBS": 1,
            "RECORDS": [
                {
                    "JOBID": jobid,
                    "STAT": stat,
                    "EXIT_CODE": exit_code,
                    "EXIT_REASON": exit_reason,
                    "PEND_REASON": pend_reason,
                }
            ],
        }
    )


def _make_monitor(
    responses: List[str],
    *,
    stop_event: Optional[asyncio.Event] = None,
    job_id: str = "12345",
):
    """Build a monitor over a scripted bjobs sequence.

    The last response repeats forever (same clamping idiom as
    ``test_log_draining.py``), so a monitor that never terminates keeps seeing a
    consistent state rather than an IndexError.
    """
    mock_lsf = MagicMock()
    mock_lsf.use_ssh = True
    # Must be a real str: on the failure path the monitor does
    # Path(log_path).parent, and Path(MagicMock()) raises TypeError *outside*
    # the try block, which would escape monitor() instead of being swallowed.
    mock_lsf.get_log_path.return_value = "/tmp/gbtest-lsf/job.log"

    commands: List[str] = []

    async def run_remote(command, raise_on_error=True):  # noqa: ANN001
        commands.append(command)
        return 0, responses[min(len(commands) - 1, len(responses) - 1)], ""

    tunnel = AsyncMock()
    tunnel.run_remote = run_remote
    mock_lsf.get_ssh_tunnel.return_value = tunnel

    queue: asyncio.Queue = asyncio.Queue()
    monitor = LSFBsubMonitor(
        lsf=mock_lsf,
        job_id=job_id,
        launch_id="test-launch",
        event_queue=queue,
        stop_event=stop_event,
        monitor_interval=MONITOR_INTERVAL,
    )
    # Never let a test reach out to a cluster for the transient-error probe.
    monitor._check_for_transient_lsf_error = AsyncMock(return_value=None)  # type: ignore[method-assign]
    return monitor, queue, commands


async def _drive(monitor: LSFBsubMonitor, *, expect_terminates: bool) -> None:
    """Run the monitor under a bounded wait.

    ``expect_terminates=False`` asserts the monitor is still polling when the
    budget expires -- i.e. it correctly treated the state as non-terminal.
    """
    task = asyncio.create_task(monitor.monitor())
    if expect_terminates:
        try:
            await asyncio.wait_for(task, timeout=TERMINATE_BUDGET)
        except asyncio.TimeoutError:  # pragma: no cover - failure path
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            pytest.fail(
                "monitor did not terminate: a terminal LSF state was treated as "
                "non-terminal (or the poll loop is stuck swallowing an exception)"
            )
    else:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), timeout=NON_TERMINAL_BUDGET)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def _msgs(queue: asyncio.Queue) -> List[str]:
    """Drain the queue to the MESSAGE_EVENT narrative lines.

    Scoped to MESSAGE_EVENT on purpose: the monitor also emits STATUS_EVENTs
    carrying the same text to drive the step status, and counting both would
    double every assertion about how many narrative lines a poll sequence
    produces.
    """
    out = []
    while not queue.empty():
        event = queue.get_nowait()
        if event.type is BuildEventType.MESSAGE_EVENT:
            out.append(getattr(event.payload, "msg", "") or "")
    return out


def _as_message_event(msg: str) -> BuildEvent:
    """Wrap a message string in a real MESSAGE_EVENT."""
    return BuildEvent(
        run_metadata=EntityRunMetadata(),
        type=BuildEventType.MESSAGE_EVENT,
        payload=BuildEventMessagePayload(level="ERROR", msg=msg),
    )


def _is_build_failing(event: BuildEvent) -> bool:
    """Ask the real RetryHandler whether this event fails the build.

    Called unbound with a dummy ``self``: the method only touches ``self`` for a
    log line. Reusing the real predicate rather than reimplementing its regex
    keeps these tests honest if that logic changes.
    """
    return bool(RetryHandler._is_terminal_failure_event(MagicMock(), event))


def _has_terminal_failure(msgs: List[str]) -> bool:
    """True if any message would make RetryHandler fail the build."""
    return any(_is_build_failing(_as_message_event(msg)) for msg in msgs)


@pytest.fixture(autouse=True)
def _fast_grace():
    with patch(
        "gbserver.monitoring.lsf_bsub_monitor.GBSERVER_MONITORING_GRACE_PERIOD", GRACE
    ):
        yield


# ---------------------------------------------------------------------------
# Case L -- the hang trap. Keep this first.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_bjobs_payload_without_pend_reason_still_parses():
    """A bjobs record with no PEND_REASON key must remain valid.

    These are the exact byte strings used by
    ``test/unit/monitoring/test_log_draining.py`` (see its ``bjobs_responses``).
    If PEND_REASON were a *required* field, validation would raise, the poll
    loop's ``except Exception: continue`` would swallow it, and the monitor would
    spin forever -- hanging that test (and CI) rather than failing it. Keep
    every BJobRecord field defaulted.
    """
    legacy = [
        '{"COMMAND": "bjobs", "JOBS": 1, "RECORDS": [{"JOBID": "12345", "STAT": "RUN", "EXIT_CODE": "", "EXIT_REASON": ""}]}',
        '{"COMMAND": "bjobs", "JOBS": 1, "RECORDS": [{"JOBID": "12345", "STAT": "DONE", "EXIT_CODE": "0", "EXIT_REASON": ""}]}',
    ]
    monitor, queue, _ = _make_monitor(legacy)
    await _drive(monitor, expect_terminates=True)
    assert not _has_terminal_failure(_msgs(queue))


# ---------------------------------------------------------------------------
# Case C / D / E -- EXIT must fail, whatever EXIT_CODE says
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exit_with_empty_exit_code_fails_the_build():
    """EXIT with a blank EXIT_CODE must FAIL, not succeed.

    Regression guard for the measured bug: on BlueVela, 103,256 of 315,504 EXIT
    jobs report EXIT_CODE="" (102,208 of them TERM_OWNER, i.e. killed). The old
    ``int(record.EXIT_CODE or "0")`` scored every one of those as returncode 0,
    so a third of genuinely failed LSF jobs were reported as successful builds.
    """
    monitor, queue, _ = _make_monitor(
        [_bjobs_json("RUN"), _bjobs_json("EXIT", exit_code="")]
    )
    await _drive(monitor, expect_terminates=True)
    assert _has_terminal_failure(_msgs(queue)), "EXIT with empty EXIT_CODE must fail"


@pytest.mark.asyncio
async def test_exit_with_dash_exit_code_fails_the_build():
    """The ``-noheader`` table form renders unset as "-"; it must not crash int()."""
    monitor, queue, _ = _make_monitor(
        [_bjobs_json("RUN"), _bjobs_json("EXIT", exit_code="-")]
    )
    await _drive(monitor, expect_terminates=True)
    assert _has_terminal_failure(_msgs(queue))


@pytest.mark.asyncio
async def test_exit_reason_is_surfaced_in_the_failure_message():
    """EXIT_REASON has always been fetched and never read -- surface it.

    It is the only field distinguishing TERM_MEMLIMIT / TERM_RUNLIMIT /
    TERM_OWNER / TERM_ADMIN, i.e. "ask for more memory" from "ask for more time"
    from "someone killed it".
    """
    monitor, queue, _ = _make_monitor(
        [
            _bjobs_json("RUN"),
            _bjobs_json(
                "EXIT",
                exit_code="137",
                exit_reason="TERM_MEMLIMIT: job killed after reaching LSF memory usage limit",
            ),
        ]
    )
    await _drive(monitor, expect_terminates=True)
    msgs = _msgs(queue)
    assert _has_terminal_failure(msgs)
    joined = "\n".join(msgs)
    assert "137" in joined
    assert "TERM_MEMLIMIT" in joined


# ---------------------------------------------------------------------------
# Case F / G -- suspended and unknown states are NOT terminal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stat", ["PSUSP", "SSUSP", "USUSP", "UNKWN", "WAIT", "PROV", "FWD_PEND"]
)
async def test_non_terminal_states_keep_polling(stat):
    """A suspended/queued/unknown job has not finished -- keep monitoring.

    Regression guard for the measured bug: these states all carry EXIT_CODE="",
    so treating them as terminal made ``int("" or "0")`` == 0 and reported a
    live job as a SUCCESSFUL build. Verified against a real cluster: a job
    bstop'd 8 seconds into a 180-second workload produced a green build.
    """
    monitor, queue, _ = _make_monitor([_bjobs_json("RUN"), _bjobs_json(stat)])
    await _drive(monitor, expect_terminates=False)
    msgs = _msgs(queue)
    assert not _has_terminal_failure(msgs), f"{stat} must not fail the build"
    assert any(stat in m for m in msgs), f"{stat} should be surfaced to the user"
    monitor._check_for_transient_lsf_error.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_suspend_and_resume_round_trip_succeeds():
    """RUN -> USUSP -> RUN -> DONE is a success, with one event per transition."""
    monitor, queue, _ = _make_monitor(
        [
            _bjobs_json("RUN"),
            _bjobs_json("USUSP"),
            _bjobs_json("USUSP"),
            _bjobs_json("RUN"),
            _bjobs_json("DONE"),
        ]
    )
    await _drive(monitor, expect_terminates=True)
    msgs = _msgs(queue)
    assert not _has_terminal_failure(msgs)
    # 4 transitions, not 5 polls: the repeated USUSP must be deduped.
    assert len(msgs) == 4, msgs


# ---------------------------------------------------------------------------
# Case H / I / J -- PEND reason surfacing, on change only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pend_reason_is_emitted_once_per_distinct_reason():
    """The user must be able to find out WHY a job is pending -- without spam."""
    reason_a = "Job's requirements for resource reservation not satisfied (Resource: ngpus_physical): 340 hosts;"
    reason_b = "Not enough job slot(s): ;"
    monitor, queue, _ = _make_monitor(
        [
            _bjobs_json("PEND", pend_reason=reason_a),
            _bjobs_json("PEND", pend_reason=reason_a),
            _bjobs_json("PEND", pend_reason=reason_a),
            _bjobs_json("PEND", pend_reason=reason_b),
            _bjobs_json("RUN"),
            _bjobs_json("DONE"),
        ]
    )
    await _drive(monitor, expect_terminates=True)
    msgs = _msgs(queue)
    assert any("ngpus_physical" in m for m in msgs), msgs
    assert any("Not enough job slot" in m for m in msgs), msgs
    # PEND(a), PEND(b), RUN, DONE -- reason A emitted once despite three polls.
    assert len(msgs) == 4, msgs


@pytest.mark.asyncio
async def test_pend_reason_host_counter_churn_does_not_spam():
    """LSF re-renders live counters ("340 hosts" -> "339 hosts") every cycle.

    Comparing raw reason text would emit an event on every poll for a long PEND,
    flooding the events table, the CLI history and the PR comment. Dedup must
    compare a digit-normalized key while still displaying the raw reason.
    """
    tmpl = "Job's requirements for resource reservation not satisfied (Resource: ngpus_physical): {} hosts;"
    monitor, queue, _ = _make_monitor(
        [
            _bjobs_json("PEND", pend_reason=tmpl.format(340)),
            _bjobs_json("PEND", pend_reason=tmpl.format(339)),
            _bjobs_json("PEND", pend_reason=tmpl.format(12)),
            _bjobs_json("RUN"),
            _bjobs_json("DONE"),
        ]
    )
    await _drive(monitor, expect_terminates=True)
    msgs = _msgs(queue)
    # PEND (once), RUN, DONE -- the counter churn must not add events.
    assert len(msgs) == 3, msgs


@pytest.mark.asyncio
async def test_reason_only_change_does_not_render_a_self_transition():
    """A reason-only change must not read as "PEND -> PEND".

    Observed on a real cluster: LSF reports "New job is waiting for scheduling"
    and then replaces it with the real blocker, e.g. "Job has a specified start
    time". The state never moved, so an arrow would look like a bug.
    """
    monitor, queue, _ = _make_monitor(
        [
            _bjobs_json("PEND", pend_reason="New job is waiting for scheduling;"),
            _bjobs_json("PEND", pend_reason="Job has a specified start time;"),
            _bjobs_json("RUN"),
            _bjobs_json("DONE"),
        ]
    )
    await _drive(monitor, expect_terminates=True)
    msgs = _msgs(queue)
    pend_msgs = [m for m in msgs if "start time" in m]
    assert pend_msgs, msgs
    assert "PEND -> PEND" not in pend_msgs[0], pend_msgs[0]
    assert "is PEND" in pend_msgs[0], pend_msgs[0]
    # A genuine transition still gets an arrow.
    assert any("PEND -> RUN" in m for m in msgs), msgs


@pytest.mark.asyncio
async def test_pend_reason_appearing_later_is_emitted():
    """LSF often reports PEND with no reason first, then fills it in."""
    reason = "Job dependency condition not satisfied;"
    monitor, queue, _ = _make_monitor(
        [
            _bjobs_json("PEND"),
            _bjobs_json("PEND"),
            _bjobs_json("PEND", pend_reason=reason),
            _bjobs_json("RUN"),
            _bjobs_json("DONE"),
        ]
    )
    await _drive(monitor, expect_terminates=True)
    msgs = _msgs(queue)
    assert any("dependency condition" in m for m in msgs), msgs
    assert len(msgs) == 4, msgs


# ---------------------------------------------------------------------------
# Case A / K -- the terminal states
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pend_run_done_succeeds_with_one_event_per_transition():
    monitor, queue, _ = _make_monitor(
        [
            _bjobs_json("PEND"),
            _bjobs_json("RUN"),
            _bjobs_json("RUN"),
            _bjobs_json("DONE"),
        ]
    )
    await _drive(monitor, expect_terminates=True)
    msgs = _msgs(queue)
    assert not _has_terminal_failure(msgs)
    assert len(msgs) == 3, msgs  # PEND, RUN, DONE -- repeated RUN deduped


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stat,should_fail",
    [("ZOMBI", True), ("POST_ERR", True), ("POST_DONE", False), ("DONE", False)],
)
async def test_terminal_states_map_to_the_right_verdict(stat, should_fail):
    monitor, queue, _ = _make_monitor([_bjobs_json("RUN"), _bjobs_json(stat)])
    await _drive(monitor, expect_terminates=True)
    assert _has_terminal_failure(_msgs(queue)) is should_fail


# ---------------------------------------------------------------------------
# Case M -- unrecognized states fail safe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unrecognized_state_is_non_terminal_and_warns(caplog):
    """An unmodelled STAT must NOT be assumed terminal.

    Defaulting unknown states to "terminal" is exactly how a suspended job
    became a green build. Failing safe means the worst case is a visible hang
    (loud, diagnosable) instead of a silent wrong SUCCESS (trusted).
    """
    monitor, queue, _ = _make_monitor([_bjobs_json("RUN"), _bjobs_json("FROBNICATE")])
    with caplog.at_level("WARNING"):
        await _drive(monitor, expect_terminates=False)
    msgs = _msgs(queue)
    assert not _has_terminal_failure(msgs)
    assert any("FROBNICATE" in m for m in msgs), msgs
    assert "unrecognized" in caplog.text.lower()


# ---------------------------------------------------------------------------
# Case N / O -- the bkill/retry guard must not regress
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_event_set_before_start_returns_cleanly():
    """retry_workload sets the launch-stopped event *before* bkilling.

    The monitor must then return without manufacturing a failure, or every
    transient-error retry would turn into a hard build failure.
    """
    stop = asyncio.Event()
    stop.set()
    monitor, queue, _ = _make_monitor(
        [_bjobs_json("RUN"), _bjobs_json("EXIT", exit_code="1")], stop_event=stop
    )
    await _drive(monitor, expect_terminates=True)
    assert not _has_terminal_failure(_msgs(queue))
    monitor._check_for_transient_lsf_error.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_stop_event_set_during_terminal_poll_is_not_a_failure():
    """The other retry branch: the bkill's own EXIT is observed after stop_event."""
    stop = asyncio.Event()
    monitor, queue, _ = _make_monitor(
        [_bjobs_json("RUN"), _bjobs_json("EXIT", exit_code="1")], stop_event=stop
    )

    async def set_stop_soon():
        await asyncio.sleep(MONITOR_INTERVAL * 1.5)
        stop.set()

    asyncio.create_task(set_stop_soon())
    await _drive(monitor, expect_terminates=True)
    assert not _has_terminal_failure(_msgs(queue))


# ---------------------------------------------------------------------------
# Case P / Q / R -- invariants
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_change_events_can_never_fail_the_build():
    """Informational state events must not trip the resilience layer.

    ``RetryHandler._is_terminal_failure_event`` scrapes ```json blocks and fails
    the build on ``state == "Failed"``. Locking this invariant means a future
    reword of these messages cannot silently start failing or retrying builds.
    """
    from gbserver.resilience.strategies.lsf_transient_error import (
        LsfTransientErrorRetryStrategy,
    )

    strategy = LsfTransientErrorRetryStrategy()
    for stat in list(LSF_STATE_CLASS) + ["FROBNICATE"]:
        state_class = LSF_STATE_CLASS.get(stat, LsfStateClass.ACTIVE)
        if state_class is not LsfStateClass.ACTIVE:
            continue  # terminal states legitimately do fail the build
        monitor, queue, _ = _make_monitor([_bjobs_json(stat)])
        record = BJobRecord(JOBID="1", STAT=stat, PEND_REASON="some reason;")
        msg = monitor._format_lsf_state_change(record, state_class)
        assert "```json" not in msg, f"{stat}: must not emit a fenced json block"
        event = _as_message_event(msg)
        assert _is_build_failing(event) is False, stat
        assert bool(strategy.should_retry(event)) is False, stat


def _status_events(queue: asyncio.Queue) -> List[Status]:
    """Drain the queue to the sequence of STATUS_EVENT statuses."""
    out = []
    while not queue.empty():
        event = queue.get_nowait()
        if event.type is BuildEventType.STATUS_EVENT:
            out.append(event.payload.status)
    return out


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stat", ["PEND", "FWD_PEND", "WAIT", "PROV", "PSUSP", "SSUSP", "USUSP", "UNKWN"]
)
async def test_non_running_lsf_states_report_gb_pending(stat):
    """A job queued or suspended in LSF must NOT report the step as RUNNING.

    This is the point of the mapping: gb should never claim progress the
    scheduler isn't making. Only LSF RUN means RUNNING.
    """
    monitor, queue, _ = _make_monitor([_bjobs_json(stat)])
    await _drive(monitor, expect_terminates=False)
    statuses = _status_events(queue)
    assert Status.PENDING in statuses, f"{stat} should report PENDING, got {statuses}"
    assert Status.RUNNING not in statuses, f"{stat} must not report RUNNING"


@pytest.mark.asyncio
async def test_lsf_run_reports_gb_running():
    monitor, queue, _ = _make_monitor([_bjobs_json("RUN")])
    await _drive(monitor, expect_terminates=False)
    assert _status_events(queue) == [Status.RUNNING]


@pytest.mark.asyncio
async def test_gb_status_is_emitted_only_when_the_mapped_status_changes():
    """PEND -> PSUSP is the same gb status, so it must not re-emit.

    The LSF-level narrative still gets an event per state change; the gb status
    only moves on PENDING <-> RUNNING.
    """
    monitor, queue, _ = _make_monitor(
        [
            _bjobs_json("PEND"),
            _bjobs_json("PSUSP"),
            _bjobs_json("USUSP"),
            _bjobs_json("RUN"),
            _bjobs_json("DONE"),
        ]
    )
    await _drive(monitor, expect_terminates=True)
    # PENDING once across PEND/PSUSP/USUSP, then RUNNING on dispatch. DONE is
    # terminal and owned by the returncode path, so it emits no STATUS_EVENT.
    assert _status_events(queue) == [Status.PENDING, Status.RUNNING]


@pytest.mark.asyncio
async def test_unrecognized_state_does_not_assert_a_gb_status():
    """If we don't know the LSF state, don't claim to know the step status."""
    monitor, queue, _ = _make_monitor([_bjobs_json("FROBNICATE")])
    await _drive(monitor, expect_terminates=False)
    assert _status_events(queue) == []


def test_every_active_state_has_a_gb_status_mapping():
    """Keep the two tables in step: every ACTIVE LSF state maps to a gb status."""
    active = {s for s, c in LSF_STATE_CLASS.items() if c is LsfStateClass.ACTIVE}
    assert set(LSF_ACTIVE_STATE_TO_GB_STATUS) == active
    # Only RUN is RUNNING; everything else that isn't finished is PENDING.
    assert LSF_ACTIVE_STATE_TO_GB_STATUS["RUN"] is Status.RUNNING
    assert {
        s for s, v in LSF_ACTIVE_STATE_TO_GB_STATUS.items() if v is Status.PENDING
    } == (active - {"RUN"})


def test_lsf_state_class_table_is_locked():
    """Pin the agreed partition so any future edit is deliberate."""
    active = {
        "PEND",
        "FWD_PEND",
        "WAIT",
        "PROV",
        "RUN",
        "PSUSP",
        "SSUSP",
        "USUSP",
        "UNKWN",
    }
    succeeded = {"DONE", "POST_DONE"}
    failed = {"EXIT", "ZOMBI", "POST_ERR"}
    assert {
        s for s, c in LSF_STATE_CLASS.items() if c is LsfStateClass.ACTIVE
    } == active
    assert {
        s for s, c in LSF_STATE_CLASS.items() if c is LsfStateClass.SUCCEEDED
    } == succeeded
    assert {
        s for s, c in LSF_STATE_CLASS.items() if c is LsfStateClass.FAILED
    } == failed


@pytest.mark.parametrize(
    "exit_code,state_class,expected",
    [
        ("", LsfStateClass.FAILED, 1),
        ("0", LsfStateClass.FAILED, 1),
        ("-", LsfStateClass.FAILED, 1),
        ("garbage", LsfStateClass.FAILED, 1),
        ("137", LsfStateClass.FAILED, 137),
        ("", LsfStateClass.SUCCEEDED, 0),
        ("0", LsfStateClass.SUCCEEDED, 0),
        ("137", LsfStateClass.SUCCEEDED, 0),
    ],
)
def test_terminal_returncode(exit_code, state_class, expected):
    """STAT decides pass/fail; EXIT_CODE only refines a failure's code."""
    record = BJobRecord(JOBID="1", STAT="EXIT", EXIT_CODE=exit_code)
    assert LSFBsubMonitor._terminal_returncode(record, state_class) == expected


def test_bjobs_record_fields_are_all_optional():
    """Every field defaulted -- see the module docstring on the hang trap."""
    record = BJobRecord()
    assert record.STAT == ""
    assert record.PEND_REASON == ""
