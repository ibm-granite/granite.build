#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""The db-space snapshot cache stays correct with many threads and processes:
copies are never lost or replaced under a reader, damaged or dead ones are
never served, and nothing a live process holds is pruned."""

import fcntl
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

import gbserver.spaces.db_space as db_space
from gbserver.spaces import snapshot_cache
from gbserver.spaces.snapshot_cache import key_dir, materialize, parse_cached_path

SRC = str(Path(snapshot_cache.__file__).resolve().parents[2])


def _writer(files, delay: float = 0.0):
    def extract(dest: Path) -> None:
        if delay:
            threading.Event().wait(delay)
        for name, text in files.items():
            (dest / name).parent.mkdir(parents=True, exist_ok=True)
            (dest / name).write_text(text)

    return extract


@pytest.fixture
def cache(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setenv("GBSERVER_DB_SPACE_CACHE_DIR", str(root))
    yield root
    snapshot_cache.release_holds()


def _copies(space="x", slot="space"):
    return sorted(p for p in key_dir(space, slot).iterdir() if p.is_dir())


def _get(content, files=None, space="x", slot="space", delay=0.0):
    return materialize(
        space, slot, content, _writer(files or {"a.txt": content}, delay), "test"
    )


def _child(root: Path, content: str, hold: bool, delay: float = 0.0):
    """A separate process that materializes ``content`` and prints the copy."""
    script = textwrap.dedent(f"""
        import sys, threading
        from pathlib import Path
        from gbserver.spaces.snapshot_cache import materialize

        def extract(dest):
            threading.Event().wait({delay})
            (dest / "a.txt").write_text({content!r})

        print(materialize("x", "space", {content!r}, extract, "child"), flush=True)
        if {hold}:
            sys.stdin.readline()  # keep holding until told to exit
        """)
    env = dict(os.environ, PYTHONPATH=SRC, GBSERVER_DB_SPACE_CACHE_DIR=str(root))
    return subprocess.Popen(
        [sys.executable, "-c", script],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )


def test_reused_while_unchanged(cache):
    first = _get("v1")
    assert _get("v1") == first
    assert _copies() == [first]
    assert (first / "a.txt").read_text() == "v1"


def test_refresh_prunes_superseded_copy_nobody_holds(cache):
    old = _get("v1")
    snapshot_cache.release_holds()  # as if every process using it had exited
    new = _get("v2")
    assert _copies() == [new] and not old.exists()


def test_copy_held_by_another_process_is_kept(cache):
    holder = _child(cache, "v1", hold=True)
    old = Path(holder.stdout.readline().strip())
    try:
        new = _get("v2")
        assert old.exists() and (old / "a.txt").read_text() == "v1"
        assert set(_copies()) == {old, new}
    finally:
        holder.communicate("\n", timeout=30)
    _get("v2")  # the holder exited: the OS dropped its lock
    assert not old.exists()


def test_concurrent_processes_get_complete_copies(cache):
    children = [_child(cache, "v1", hold=False, delay=0.3) for _ in range(6)]
    paths = [Path(c.communicate(timeout=60)[0].strip()) for c in children]
    assert all(c.returncode == 0 for c in children)
    for path in paths:
        # Each returned copy was complete when returned; a spare from a lost
        # race may since have been pruned, but the winner is always there.
        assert path.parent == key_dir("x", "space")
    final = _get("v1")
    assert _copies() == [final] and (final / "a.txt").read_text() == "v1"


def test_concurrent_threads_share_one_copy(cache):
    results = []

    def run():
        results.append(_get("v1", delay=0.2))

    threads = [threading.Thread(target=run) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(results)) == 1 and _copies() == [results[0]]
    assert not any(p.name.startswith(".tmp-") for p in key_dir("x", "space").iterdir())


def test_damaged_copy_is_never_served(cache):
    old = _get("v1", files={"a.txt": "v1", "b.txt": "b"})
    (old / "b.txt").unlink()  # e.g. a cleaner deleted files, marker left behind
    new = _get("v1", files={"a.txt": "v1", "b.txt": "b"})
    assert new != old and (new / "b.txt").is_file()
    assert old.exists()  # still held by this process: not pulled from under it
    snapshot_cache.release_holds()
    _get("v1", files={"a.txt": "v1", "b.txt": "b"})
    assert not old.exists()


def test_dead_extraction_is_removed_live_one_kept(cache):
    keydir = key_dir("x", "space")
    dead = keydir / ".tmp-dead"
    dead.mkdir()
    (dead / ".gb-writing").touch()
    live = keydir / ".tmp-live"
    live.mkdir()
    fd = os.open(live / ".gb-writing", os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)  # its writer is still running
    try:
        _get("v1")
        assert not dead.exists() and live.exists()
    finally:
        os.close(fd)


def test_identical_archives_in_two_spaces_stay_apart(cache):
    a = _get("same", space="a")
    b = _get("same", space="b")
    assert a != b
    assert db_space.portable_uri((a / "env").as_uri()) == (
        "db://spaces/a#subdirectory=env"
    )
    assert db_space.portable_uri(b.as_uri()) == "db://spaces/b"


def test_cached_path_names_space_and_slot(cache):
    copy = _get("v1", space="a/b", slot="base_uri_3")
    assert parse_cached_path(copy / "steps" / "x") == ("a/b", "base_uri_3", "steps/x")
    assert db_space.portable_uri((copy / "steps").as_uri()) == (
        "db://spaces/a%2Fb/base_uri_3#subdirectory=steps"
    )
    assert parse_cached_path(cache.parent) is None


def test_failed_extraction_leaves_nothing(cache):
    def broken(dest: Path) -> None:
        (dest / "half").write_text("x")
        raise ValueError("bad zip")

    with pytest.raises(ValueError, match="bad zip"):
        materialize("x", "space", "v1", broken, "test")
    assert _copies() == []


def test_cache_root_must_be_private(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    monkeypatch.setenv("GBSERVER_DB_SPACE_CACHE_DIR", str(shared))
    with pytest.raises(ValueError, match="refusing snapshot cache"):
        _get("v1")
