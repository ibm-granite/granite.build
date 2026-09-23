"""Tests for post-launch host SSH bounding/retry (_skypilot_ssh).

A slow-banner host used to hang to the 600s blanket timeout, since ConnectTimeout
bounds only the TCP leg. Now a bounded login probe fails fast, the connect phase
(only) retries, and the payload runs on a native async subprocess so cancellation
reaches the ssh child.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from gbserver.environment._skypilot_ssh import (
    _await_host_reachable,
    _host_ssh_base_cmd,
    _is_fatal_ssh_error,
    execute_on_host_via_ssh,
)


def test_base_cmd_bounds_connection_and_disables_prompts():
    cmd = _host_ssh_base_cmd("/k/id", "1.2.3.4", 30)
    assert cmd[0] == "ssh"
    assert "ConnectTimeout=30" in cmd
    # BatchMode stops ssh blocking on an interactive password prompt.
    assert "BatchMode=yes" in cmd
    assert "ubuntu@1.2.3.4" in cmd


@pytest.mark.asyncio
async def test_reachable_returns_on_first_success():
    proc = _proc(rc=0)
    with patch("asyncio.create_subprocess_exec", side_effect=_spawns(proc)) as spawn:
        await _await_host_reachable("1.2.3.4", "/k/id", 5)
    assert spawn.call_count == 1
    # Probe is an echo, never the caller's payload.
    assert "echo" in spawn.call_args[0]


@pytest.mark.asyncio
async def test_login_hang_is_bounded_killed_and_raises():
    """Late banner: bounded by wait_for and the child killed, not left hanging."""
    proc = MagicMock()

    async def _hang(*_a, **_kw):
        await asyncio.sleep(3600)

    proc.communicate = _hang
    proc.wait = _async_return(0)

    with (
        patch("asyncio.create_subprocess_exec", side_effect=_spawns(proc)),
        patch("gbserver.types.constants.GBSERVER_SKYPILOT_HOST_SSH_ATTEMPTS", 1),
    ):
        with pytest.raises(RuntimeError, match="did not accept an SSH login"):
            await asyncio.wait_for(
                _await_host_reachable("1.2.3.4", "/k/id", 1), timeout=30
            )
    proc.kill.assert_called()


@pytest.mark.asyncio
async def test_connect_phase_retries_then_succeeds():
    """A briefly-wedged login node is ridden out rather than failing the sidecar."""
    procs = [_proc(rc=255, stderr=b"timed out"), _proc(rc=0)]

    async def _spawn(*_a, **_kw):
        return procs.pop(0)

    with (
        patch("asyncio.create_subprocess_exec", side_effect=_spawn),
        patch("gbserver.types.constants.GBSERVER_SKYPILOT_HOST_SSH_ATTEMPTS", 3),
        patch("asyncio.sleep", _async_return(None)),
    ):
        await _await_host_reachable("1.2.3.4", "/k/id", 5)
    assert procs == []


@pytest.mark.asyncio
async def test_auth_rejection_fails_fast_without_retrying():
    """A rejected key stays rejected: fail on attempt 1, don't burn the budget."""
    proc = _proc(rc=255, stderr=b"ubuntu@h: Permission denied (publickey).")
    slept = []

    async def _sleep(d):
        slept.append(d)

    with (
        patch("asyncio.create_subprocess_exec", side_effect=_spawns(proc)) as spawn,
        patch("gbserver.types.constants.GBSERVER_SKYPILOT_HOST_SSH_ATTEMPTS", 3),
        patch("asyncio.sleep", _sleep),
    ):
        with pytest.raises(RuntimeError, match="did not accept an SSH login"):
            await _await_host_reachable("1.2.3.4", "/k/id", 5)
    assert spawn.call_count == 1
    assert slept == []


@pytest.mark.asyncio
async def test_spawn_failure_still_backs_off_between_attempts():
    """A spawn failure must not skip the backoff and blow the budget instantly."""
    slept = []

    async def _sleep(d):
        slept.append(d)

    async def _boom(*_a, **_kw):
        raise OSError("cannot fork")

    with (
        patch("asyncio.create_subprocess_exec", side_effect=_boom),
        patch("gbserver.types.constants.GBSERVER_SKYPILOT_HOST_SSH_ATTEMPTS", 3),
        patch("asyncio.sleep", _sleep),
    ):
        with pytest.raises(RuntimeError, match="could not spawn ssh"):
            await _await_host_reachable("1.2.3.4", "/k/id", 5)
    # Backoff ran between attempts (2 sleeps for 3 attempts).
    assert slept == [1, 2]


@pytest.mark.asyncio
async def test_io_error_is_not_reported_as_a_timeout():
    """An IO failure must not be mislabelled 'timed out' (wrong debugging path)."""
    proc = MagicMock()

    async def _raise(*_a, **_kw):
        raise OSError("pipe read failed")

    proc.communicate = _raise
    proc.wait = _async_return(0)

    with (
        patch(
            "gbserver.environment._skypilot_ssh._await_host_reachable",
            _async_return(None),
        ),
        patch("asyncio.create_subprocess_exec", side_effect=_spawns(proc)),
    ):
        with pytest.raises(RuntimeError, match="failed while reading output"):
            await execute_on_host_via_ssh(
                host_ip="1.2.3.4", ssh_key="/k/id", commands="c"
            )


@pytest.mark.asyncio
async def test_unreachable_host_fails_before_running_payload():
    """An unreachable host must not get the payload at all."""
    with (
        patch(
            "gbserver.environment._skypilot_ssh._await_host_reachable",
            side_effect=RuntimeError("Host 1.2.3.4 did not accept an SSH login"),
        ),
        patch("asyncio.create_subprocess_exec") as spawn,
    ):
        with pytest.raises(RuntimeError, match="did not accept an SSH login"):
            await execute_on_host_via_ssh(
                host_ip="1.2.3.4", ssh_key="/k/id", commands="start-sidecar"
            )
    spawn.assert_not_called()


@pytest.mark.asyncio
async def test_payload_runs_once_and_is_not_retried():
    """The payload may not be idempotent: a failure must not re-run it."""
    proc = _proc(rc=1, stderr=b"boom")
    with (
        patch(
            "gbserver.environment._skypilot_ssh._await_host_reachable",
            _async_return(None),
        ),
        patch("asyncio.create_subprocess_exec", side_effect=_spawns(proc)) as spawn,
    ):
        with pytest.raises(RuntimeError, match="exit code 1"):
            await execute_on_host_via_ssh(
                host_ip="1.2.3.4", ssh_key="/k/id", commands="start-sidecar"
            )
    assert spawn.call_count == 1


@pytest.mark.asyncio
async def test_payload_timeout_kills_child():
    """A wedged payload is killed and reaped, not orphaned in a thread."""
    proc = MagicMock()

    async def _hang(*_a, **_kw):
        await asyncio.sleep(3600)

    proc.communicate = _hang
    proc.wait = _async_return(0)

    with (
        patch(
            "gbserver.environment._skypilot_ssh._await_host_reachable",
            _async_return(None),
        ),
        patch("asyncio.create_subprocess_exec", side_effect=_spawns(proc)),
    ):
        with pytest.raises(RuntimeError, match="timed out"):
            await execute_on_host_via_ssh(
                host_ip="1.2.3.4", ssh_key="/k/id", commands="c", timeout=1
            )
    proc.kill.assert_called()


@pytest.mark.asyncio
async def test_cancellation_kills_child_and_propagates():
    """Cancelling the build must reach the ssh child (the to_thread leak)."""
    proc = MagicMock()

    async def _hang(*_a, **_kw):
        await asyncio.sleep(3600)

    proc.communicate = _hang
    proc.wait = _async_return(0)

    with (
        patch(
            "gbserver.environment._skypilot_ssh._await_host_reachable",
            _async_return(None),
        ),
        patch("asyncio.create_subprocess_exec", side_effect=_spawns(proc)),
    ):
        task = asyncio.create_task(
            execute_on_host_via_ssh(host_ip="1.2.3.4", ssh_key="/k/id", commands="c")
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    proc.kill.assert_called()


@pytest.mark.asyncio
async def test_env_vars_are_exported_and_quotes_escaped():
    proc = _proc(rc=0)
    captured = {}

    async def _comm(input=None):  # noqa: A002 — mirrors communicate's kwarg
        captured["stdin"] = (input or b"").decode()
        return (b"", b"")

    proc.communicate = _comm
    with (
        patch(
            "gbserver.environment._skypilot_ssh._await_host_reachable",
            _async_return(None),
        ),
        patch("asyncio.create_subprocess_exec", side_effect=_spawns(proc)),
    ):
        await execute_on_host_via_ssh(
            host_ip="1.2.3.4",
            ssh_key="/k/id",
            commands="run.sh",
            env_vars={"TOKEN": "a'b"},
        )
    assert "export TOKEN='a'\\''b'" in captured["stdin"]
    assert captured["stdin"].endswith("run.sh")


# --- helpers ---------------------------------------------------------------


def _proc(rc=0, stdout=b"", stderr=b""):
    proc = MagicMock()
    proc.returncode = rc
    proc.communicate = _async_return((stdout, stderr))
    proc.wait = _async_return(rc)
    return proc


def _async_return(value):
    async def _inner(*_a, **_kw):
        return value

    return _inner


def _spawns(proc):
    async def _inner(*_a, **_kw):
        return proc

    return _inner


@pytest.mark.parametrize(
    "stderr",
    [
        "Permission denied (publickey).",
        "Too many authentication failures",
        "Host key verification failed.",
        "no such identity: /home/gbserver/.ssh/id_rsa",
        "Load key: invalid privatekey",
        "Bad configuration option: FooBar",
    ],
)
def test_auth_and_config_failures_stay_fatal(stderr):
    """A rejected key or bad option stays rejected — retrying only wastes time."""
    assert _is_fatal_ssh_error(stderr) is True


@pytest.mark.parametrize(
    "stderr",
    [
        "Warning: Identity file /shared/keys/id_rsa not accessible: "
        "No such file or directory.",
        "no such file or directory",
    ],
)
def test_missing_file_is_retriable(stderr):
    """An identity file on a momentarily-unavailable shared mount emits exactly
    this. Treating it as fatal aborted every remaining attempt on the first try,
    which is the opposite of what this module is for; retrying a genuinely missing
    key costs a few bounded attempts before the same failure surfaces anyway."""
    assert _is_fatal_ssh_error(stderr) is False


@pytest.mark.parametrize(
    "stderr",
    [
        "Connection timed out during banner exchange",
        "kex_exchange_identification: Connection closed by remote host",
        "Connection reset by peer",
    ],
)
def test_transient_ssh_failures_are_retriable(stderr):
    assert _is_fatal_ssh_error(stderr) is False


@pytest.mark.asyncio
async def test_zero_login_timeout_does_not_brick_the_payload():
    """login_timeout=0 must be floored, not passed to wait_for.

    Regression guard: unlike the pre-launch probe's timeout (where 0 means
    "disabled"), a 0 here reached wait_for(timeout=0), fired immediately, failed
    every attempt, and raised before the payload ever ran. The two vars sit side by
    side in the same table with opposite semantics for 0.
    """
    proc = MagicMock()

    async def _communicate(*_a, **_kw):
        return (b"gbserver probe\n", b"")

    proc.communicate = _communicate
    proc.returncode = 0

    async def _spawn(*args, **_kwargs):
        _spawn.argv = args
        return proc

    with patch("asyncio.create_subprocess_exec", side_effect=_spawn):
        # Must not raise: with the floor, the echo probe succeeds.
        await _await_host_reachable("10.0.0.1", "/k.pem", 0)

    # ConnectTimeout is rendered from the floored value, never 0.
    assert "ConnectTimeout=0" not in " ".join(_spawn.argv)
