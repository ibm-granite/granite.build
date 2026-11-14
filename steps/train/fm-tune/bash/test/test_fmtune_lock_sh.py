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

"""Unit tests for src/fmtune_run.sh: the cache lock (owner, waits, timeouts, release).

A lock is a mkdir dir whose pid file names <host>:<pid>; it is never taken over.
The 5 s poll is stubbed to 0.05 s (_run_waiting). Mode 1 only.
"""

import os
import pathlib
import shutil
import signal
import subprocess
import time

import pytest
from _fmtune_helpers import (
    _SCRIPT,
    _clear_output,
    _clone_lock,
    _git_repo,
    _run,
    _stub,
)


def _dead_owner() -> str:
    dead = subprocess.Popen(["true"])
    dead.wait()
    return f"{os.uname().nodename}:{dead.pid}"


def _held_lock(env, owner) -> pathlib.Path:
    """A lock dir as another build left it: owner None = no pid file at all."""
    lock = _clone_lock(env)
    lock.mkdir(parents=True)
    if owner is not None:
        (lock / "pid").write_text(owner)
    return lock


def _sq(path) -> str:
    return "'" + str(path).replace("'", "'\\''") + "'"


def _run_waiting(env, tmp_path, limit=20):
    """Run with the 5 s poll stubbed to 0.05 s; returns (proc, seconds taken)."""
    _stub(tmp_path / "bin", "sleep", f'exec "{shutil.which("sleep")}" 0.05\n')
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            ["bash", str(_SCRIPT)],
            env=env,
            capture_output=True,
            check=False,
            text=True,
            timeout=limit,
        )
    except subprocess.TimeoutExpired as e:
        pytest.fail(
            f"still waiting on the lock after {limit}s: {e.stdout!r} {e.stderr!r}"
        )
    return proc, time.monotonic() - t0


def test_a_lock_whose_owner_died_on_this_host_dies_at_once_with_the_rm_hint(
    env, tmp_path
):
    """Never taken over: the run stops at once and says how to clear it."""
    _git_repo(env, tmp_path)
    env["FMTUNE_CACHE_DIR"] = str(
        tmp_path / "it's cache"
    )  # the hint shell-quotes the path
    owner = _dead_owner()
    lock = _held_lock(env, owner + "\n")
    env["FMTUNE_LOCK_TIMEOUT"] = "60"
    proc, took = _run_waiting(env, tmp_path)
    assert took < 10, took
    assert proc.returncode == 2
    assert (
        f"fm-tune: lock {lock} is held by {owner}, which is no longer running; "
        f"remove it with: rm -rf {_sq(lock)} if no fm-tune build is running"
    ) in proc.stderr
    assert (lock / "pid").read_text() == owner + "\n"
    assert sorted(p.name for p in lock.iterdir()) == ["pid"]
    assert not pathlib.Path(env["STUB_RECORD"]).exists()


def test_an_owner_that_released_the_lock_and_then_exited_is_not_reported(env, tmp_path):
    """The waiter read the owner, then the owner released the lock and exited before
    kill -0 ran: that lock is gone, not stale. A stub cat releases it right after
    the waiter's first read of the pid file."""
    _git_repo(env, tmp_path)
    owner = _dead_owner()
    lock = _held_lock(env, owner + "\n")
    flipped = tmp_path / "flipped"
    _stub(
        tmp_path / "bin",
        "cat",
        f'"{shutil.which("cat")}" "$@"; rc=$?\n'
        f'if [ "$1" = "{lock}/pid" ] && [ ! -e "{flipped}" ]; then\n'
        f'  touch "{flipped}"; rm -rf "{lock}"\n'
        "fi\nexit $rc\n",
    )
    env["FMTUNE_LOCK_TIMEOUT"] = "60"
    proc, _ = _run_waiting(env, tmp_path)
    assert flipped.exists()
    assert proc.returncode == 0, proc.stderr
    assert pathlib.Path(env["STUB_RECORD"]).exists()
    assert not lock.exists()


def _assert_timed_out(env, proc, took, lock, owner_text, contents):
    assert 1 <= took < 15, took
    assert proc.returncode == 2
    assert f"fm-tune: lock {lock} is still held after " in proc.stderr
    assert (
        f"(owner: {owner_text}); the owner may still be working (a venv build can "
        "take longer): raise FMTUNE_LOCK_TIMEOUT and re-run. Only if no fm-tune build "
        f"is running, remove it with: rm -rf {_sq(lock)}"
    ) in proc.stderr
    assert sorted(p.name for p in lock.iterdir()) == contents
    assert not pathlib.Path(env["STUB_RECORD"]).exists()


@pytest.mark.parametrize("pid", ["own", "1"])
def test_a_lock_held_by_a_live_owner_on_this_host_is_waited_on_then_reported(
    env, tmp_path, pid
):
    """pid 1 is another user's live process: kill -0 fails (EPERM), but it is not dead."""
    if pid == "1":
        try:
            os.kill(1, 0)
        except PermissionError:
            pass
        else:  # root, or a container whose pid 1 is ours: it would test the alive path
            pytest.skip("this user may signal pid 1")
    _git_repo(env, tmp_path)
    owner = f"{os.uname().nodename}:{os.getpid() if pid == 'own' else 1}"
    lock = _held_lock(env, owner + "\n")
    env["FMTUNE_LOCK_TIMEOUT"] = "2"
    proc, took = _run_waiting(env, tmp_path)
    _assert_timed_out(env, proc, took, lock, owner, ["pid"])
    assert (lock / "pid").read_text() == owner + "\n"


@pytest.mark.parametrize("pid_file", [None, ""], ids=["pid-less", "empty-pid"])
def test_a_lock_with_no_owner_recorded_is_waited_on_then_reported(
    env, tmp_path, pid_file
):
    _git_repo(env, tmp_path)
    lock = _held_lock(env, pid_file)
    env["FMTUNE_LOCK_TIMEOUT"] = "2"
    proc, took = _run_waiting(env, tmp_path)
    _assert_timed_out(
        env, proc, took, lock, "none recorded", [] if pid_file is None else ["pid"]
    )
    if pid_file is not None:
        assert (lock / "pid").read_text() == ""


def test_a_lock_released_while_waiting_is_then_taken(env, tmp_path):
    _git_repo(env, tmp_path)
    lock = _held_lock(env, f"{os.uname().nodename}:{os.getpid()}\n")
    waited = tmp_path / "waited"
    _stub(
        tmp_path / "bin",
        "sleep",
        f'touch "{waited}"; exec "{shutil.which("sleep")}" 0.05\n',
    )
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["FMTUNE_LOCK_TIMEOUT"] = "60"
    proc = subprocess.Popen(
        ["bash", str(_SCRIPT)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 15
    while not waited.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert waited.exists(), proc.communicate(timeout=15)
    shutil.rmtree(lock)  # the holder finishes
    out, err = proc.communicate(timeout=15)
    assert proc.returncode == 0, err
    assert f"[fm-tune] waiting for lock {lock}" in out
    assert pathlib.Path(env["STUB_RECORD"]).exists()
    assert not lock.exists()


def test_a_lock_released_between_the_mkdir_and_the_look_is_then_taken(env, tmp_path):
    """mkdir fails on a held lock, which is released before the waiter looks at it:
    no dir there is not an unwritable cache. A stub mkdir releases it after the
    first failed attempt."""
    _git_repo(env, tmp_path)
    lock = _held_lock(env, "otherhost:1\n")
    flipped = tmp_path / "flipped"
    _stub(
        tmp_path / "bin",
        "mkdir",
        f'"{shutil.which("mkdir")}" "$@"; rc=$?\n'
        f'if [ "$1" = "{lock}" ] && [ ! -e "{flipped}" ]; then\n'
        f'  touch "{flipped}"; rm -rf "{lock}"\n'
        "fi\nexit $rc\n",
    )
    env["FMTUNE_LOCK_TIMEOUT"] = "60"
    proc, _ = _run_waiting(env, tmp_path)
    assert flipped.exists()
    assert proc.returncode == 0, proc.stderr
    assert not lock.exists()


def test_concurrent_builds_hold_the_lock_one_at_a_time(env, tmp_path):
    """At most one build may be inside the locked section at a time. A stub git logs
    enter/exit around each call (the clone section runs only git); a stub sleep makes
    the 5 s poll fast."""
    _git_repo(env, tmp_path)
    events = tmp_path / "events"
    real_sleep = shutil.which("sleep")
    _stub(
        tmp_path / "bin",
        "git",
        f'echo "$STUB_ID enter" >> "{events}"; "{real_sleep}" 0.1\n'
        f'"{shutil.which("git")}" "$@"; rc=$?\n'
        f'echo "$STUB_ID exit" >> "{events}"; exit $rc\n',
    )
    _stub(tmp_path / "bin", "sleep", f'exec "{real_sleep}" 0.05\n')
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["FMTUNE_LOCK_TIMEOUT"] = "60"
    procs = []
    for i in range(5):
        e = dict(
            env,
            STUB_ID=str(i),
            FMTUNE_OUTPUT_DIR=str(tmp_path / f"out{i}"),
            STUB_RECORD=str(tmp_path / f"record{i}.json"),
        )
        procs.append(
            subprocess.Popen(
                ["bash", str(_SCRIPT)],
                env=e,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )
    outs = [p.communicate(timeout=60) for p in procs]
    ids = [ln.split()[0] for ln in events.read_text().splitlines()]
    # One build's git calls form one unbroken run; another build's call inside it overlaps.
    runs = [x for k, x in enumerate(ids) if k == 0 or ids[k - 1] != x]
    assert len(runs) == len(set(runs)) == 5, f"locked sections overlapped: {ids}"
    assert [p.returncode for p in procs] == [0] * 5, [e for _, e in outs]
    assert (
        sorted(
            p.name
            for p in pathlib.Path(env["FMTUNE_CACHE_DIR"]).iterdir()
            if ".lock" in p.name
        )
        == []
    )


def test_the_lock_records_host_and_pid(env, tmp_path):
    """While held, the lock's pid file names this host, so other nodes never judge it."""
    _git_repo(env, tmp_path)
    env["STUB_LOCKS"] = env["FMTUNE_CACHE_DIR"]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    real_git = shutil.which("git")
    (bin_dir / "git").write_text(
        '#!/bin/sh\ncat "$STUB_LOCKS"/*.lock/pid >> "$STUB_LOCKS/../owners" 2>/dev/null\n'
        f'exec "{real_git}" "$@"\n'
    )
    (bin_dir / "git").chmod(0o755)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    assert _run(env).returncode == 0
    owners = set((tmp_path / "owners").read_text().split())
    assert len(owners) == 1
    host, pid = owners.pop().rsplit(":", 1)
    assert host == os.uname().nodename and pid.isdigit()


def test_a_lock_held_by_another_host_is_never_taken_over(env, tmp_path):
    _git_repo(env, tmp_path)
    lock = _held_lock(
        env, "otherhost:1\n"
    )  # pid 1 is alive here, dead there: unknowable
    env["FMTUNE_LOCK_TIMEOUT"] = "2"
    proc, took = _run_waiting(env, tmp_path)
    _assert_timed_out(env, proc, took, lock, "otherhost:1", ["pid"])
    assert (lock / "pid").read_text() == "otherhost:1\n"


def _gone_rm_stub(tmp_path, lock, body: str) -> None:
    """A stub rm running ``body`` when unlock removes the renamed-away lock dir."""
    _stub(
        tmp_path / "bin",
        "rm",
        f'case "$*" in *{lock.name}.gone.*) {body} ;; esac\n'
        f'exec "{shutil.which("rm")}" "$@"\n',
    )


def test_a_failed_unlock_never_removes_a_lock_taken_since(env, tmp_path):
    """rm -rf of the renamed-away dir can fail (NFS .nfsXXXX, EIO): set -e exited
    there, and the EXIT trap's second unlock moved and removed the lock another
    build had taken at the freed path meanwhile."""
    _git_repo(env, tmp_path)
    lock = _clone_lock(env)
    _gone_rm_stub(
        tmp_path,
        lock,
        f'[ -e "{tmp_path}/flipped" ] || {{ touch "{tmp_path}/flipped"; '
        f'mkdir "{lock}"; echo otherhost:1 > "{lock}/pid"; exit 1; }}',
    )
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    proc = _run(env)
    assert (tmp_path / "flipped").exists()
    assert (lock / "pid").read_text() == "otherhost:1\n", proc.stderr
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP])
def test_a_build_killed_while_unlocking_never_removes_a_lock_taken_since(
    env, tmp_path, sig
):
    """Killed while rm -rf runs: the EXIT trap ran unlock again on the freed path.
    The signal goes to the process group, as Ctrl-C or a scheduler's kill does."""
    _git_repo(env, tmp_path)
    lock = _clone_lock(env)
    marker = tmp_path / "removing"
    _gone_rm_stub(tmp_path, lock, f'touch "{marker}"; "{shutil.which("sleep")}" 2')
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    proc = subprocess.Popen(
        ["bash", str(_SCRIPT)],
        env=env,
        text=True,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 15
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert marker.exists(), proc.communicate(timeout=15)
    lock.mkdir()  # another build takes the freed lock
    (lock / "pid").write_text("otherhost:1\n")
    os.killpg(proc.pid, sig)
    proc.communicate(timeout=15)
    assert (lock / "pid").read_text() == "otherhost:1\n"


def test_a_failed_owner_write_fails_fast_every_time(env, tmp_path):
    """A claim that cannot record its owner must not leave its lock dir: every later
    run would wait on it for the whole timeout."""
    _git_repo(env, tmp_path)
    _stub(
        tmp_path / "bin",
        "mv",
        f'case "$*" in *.lock/pid.new*) exit 1 ;; esac\nexec "{shutil.which("mv")}" "$@"\n',
    )
    env["FMTUNE_LOCK_TIMEOUT"] = "60"
    for _ in range(2):
        proc, took = _run_waiting(env, tmp_path, limit=15)
        assert took < 10, took
        assert proc.returncode == 2
        assert (
            f"fm-tune: cannot write the owner of lock {_clone_lock(env)}" in proc.stderr
        )
    assert not _clone_lock(env).exists()


def test_a_lock_dir_that_cannot_be_renamed_is_still_released(env, tmp_path):
    """Some FUSE mounts cannot rename a dir, and a leftover .gone dir of the same name
    blocks the rename too: the lock leaked, and with no takeover every later run died.
    """
    _git_repo(env, tmp_path)
    _stub(
        tmp_path / "bin",
        "mv",
        f'case "$*" in *.lock.gone.*) exit 1 ;; esac\nexec "{shutil.which("mv")}" "$@"\n',
    )
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["FMTUNE_LOCK_TIMEOUT"] = "2"
    cache = pathlib.Path(env["FMTUNE_CACHE_DIR"])
    first = _run(env)
    assert first.returncode == 0, first.stderr
    assert not list(cache.glob("*.lock*"))
    _clear_output(env)
    second = _run(env)
    assert second.returncode == 0, second.stderr


def test_unlock_leaves_a_lock_that_is_no_longer_ours(env, tmp_path):
    """Someone removed our lock mid-section and another build took it: our unlock
    must not remove theirs. A stub git swaps the pid file for another owner's."""
    _git_repo(env, tmp_path)
    lock = _clone_lock(env)
    _stub(
        tmp_path / "bin",
        "git",
        f'[ -d "{lock}" ] && echo otherhost:1 > "{lock}/pid"\n'
        f'exec "{shutil.which("git")}" "$@"\n',
    )
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    assert (lock / "pid").read_text() == "otherhost:1\n"


@pytest.mark.parametrize("timeout", ["9223372036854775807", "1234567890"])
def test_a_lock_timeout_too_big_for_shell_arithmetic_is_refused(env, timeout):
    """A value this long would wrap in $(( ))."""
    env["FMTUNE_LOCK_TIMEOUT"] = timeout
    proc = _run(env)
    assert proc.returncode == 2
    assert (
        "fm-tune: FMTUNE_LOCK_TIMEOUT must be a whole number of seconds (at most 9 digits)"
        in proc.stderr
    )
    assert not pathlib.Path(env["STUB_RECORD"]).exists()


@pytest.mark.parametrize("timeout", ["02", "08"])
def test_a_lock_timeout_with_a_leading_zero_is_read_as_decimal(env, tmp_path, timeout):
    """$(( 08 )) is an octal error in bash, so the wait on a held lock never ended."""
    _git_repo(env, tmp_path)
    lock = _held_lock(env, "otherhost:1\n")
    env["FMTUNE_LOCK_TIMEOUT"] = timeout
    proc, took = _run_waiting(env, tmp_path, limit=30)
    assert (
        int(timeout) - 1 <= took < int(timeout) + 10
    ), took  # $SECONDS is whole seconds
    assert proc.returncode == 2
    assert f"fm-tune: lock {lock} is still held after {int(timeout)}s " in proc.stderr
    assert (lock / "pid").read_text() == "otherhost:1\n"
