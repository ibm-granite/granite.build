#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""The db-space snapshot cache stays correct with many threads and processes:
copies are never lost or replaced under a reader, damaged or dead ones are
never served, and nothing a live lease holds is pruned."""

import fcntl
import gc
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

import gbserver.spaces.db_space as db_space
from gbserver.spaces import snapshot_cache
from gbserver.spaces.snapshot_cache import (
    get_cache_root,
    key_dir,
    materialize,
    parse_cached_path,
)

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
        from gbserver.spaces.snapshot_cache import materialize

        def extract(dest):
            threading.Event().wait({delay})
            (dest / "a.txt").write_text({content!r})

        lease = materialize("x", "space", {content!r}, extract, "child")
        print(lease.path, flush=True)
        if {hold}:
            sys.stdin.readline()  # keep the lease until told to exit
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
    second = _get("v1")
    assert second.path == first.path and _copies() == [first.path]
    assert (first.path / "a.txt").read_text() == "v1"


def test_released_copy_is_pruned_after_a_refresh(cache):
    old = _get("v1")
    old_path = old.path
    old.release()  # nothing uses it any more
    new = _get("v2")
    assert _copies() == [new.path] and not old_path.exists()


def test_garbage_collected_lease_releases(cache):
    old_path = _get("v1").path  # the lease is dropped at once
    gc.collect()
    new = _get("v2")
    assert _copies() == [new.path] and not old_path.exists()


def test_leased_copy_is_kept(cache):
    old = _get("v1")
    new = _get("v2")
    assert set(_copies()) == {old.path, new.path}
    old.release()
    _get("v3")
    assert not old.path.exists()


def test_copy_held_by_another_process_is_kept(cache):
    holder = _child(cache, "v1", hold=True)
    old = Path(holder.stdout.readline().strip())
    try:
        new = _get("v2")
        assert old.exists() and (old / "a.txt").read_text() == "v1"
        assert set(_copies()) == {old, new.path}
    finally:
        holder.communicate("\n", timeout=30)
    _get("v3")  # the holder exited: the OS dropped its lock
    assert not old.exists()


def test_concurrent_processes_get_complete_copies(cache):
    children = [_child(cache, "v1", hold=False, delay=0.3) for _ in range(6)]
    outputs = [c.communicate(timeout=60)[0].strip() for c in children]
    assert all(c.returncode == 0 for c in children)
    assert all(Path(o).parent == key_dir("x", "space") for o in outputs)
    final = _get("v1")
    assert (final.path / "a.txt").read_text() == "v1"
    _get("v2").release()  # publishing prunes the unheld spares
    assert len([p for p in _copies() if p.name.startswith(final.path.name[:32])]) == 1


def test_concurrent_threads_share_one_copy(cache):
    results = []

    def run():
        results.append(_get("v1", delay=0.2))

    threads = [threading.Thread(target=run) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({r.path for r in results}) == 1 and _copies() == [results[0].path]
    assert not any(p.name.startswith(".tmp-") for p in key_dir("x", "space").iterdir())


def test_damaged_copy_is_never_served(cache):
    files = {"a.txt": "v1", "b.txt": "b"}
    old = _get("v1", files=files)
    (old.path / "b.txt").unlink()  # e.g. a cleaner deleted files, marker left
    new = _get("v1", files=files)
    assert new.path != old.path and (new.path / "b.txt").is_file()
    assert old.path.exists()  # still leased: not pulled from under its user
    old.release()
    _get("v2")
    assert not old.path.exists()


def test_files_added_to_a_copy_do_not_damage_it(cache):
    first = _get("v1")
    (first.path / "__pycache__").mkdir()
    (first.path / "__pycache__" / "m.pyc").write_text("compiled")
    assert _get("v1").path == first.path


def test_dead_extraction_is_removed_live_one_kept(cache):
    keydir = key_dir("x", "space")
    dead = keydir / ".tmp-dead"
    dead.mkdir()  # its writer crashed: no lock held
    (keydir / ".tmp-dead.lock").touch()
    live_lock = keydir / ".tmp-live.lock"  # taken before its dir exists
    fd = os.open(live_lock, os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        _get("v1")
        assert not dead.exists() and not (keydir / ".tmp-dead.lock").exists()
        assert live_lock.exists()
    finally:
        os.close(fd)


def test_dot_names_stay_inside_the_root(cache):
    for name in (".", "..", "a/../b"):
        path = _get("v1", space=name).path
        assert get_cache_root().resolve() in path.resolve().parents
        assert parse_cached_path(path)[0] == name
    assert len({_get("v1", space=n).path.parent.parent for n in (".", "..")}) == 2


def test_identical_archives_in_two_spaces_stay_apart(cache):
    a = _get("same", space="a").path
    b = _get("same", space="b").path
    assert a != b
    assert db_space.portable_uri((a / "env").as_uri()) == (
        "db://spaces/a#subdirectory=env"
    )
    assert db_space.portable_uri(b.as_uri()) == "db://spaces/b"


def test_cached_path_names_space_and_slot(cache):
    copy = _get("v1", space="a/b", slot="base_uri_3").path
    assert parse_cached_path(copy / "steps" / "x") == ("a/b", "base_uri_3", "steps/x")
    assert db_space.portable_uri((copy / "steps").as_uri()) == (
        "db://spaces/a%2Fb/base_uri_3#subdirectory=steps"
    )
    assert parse_cached_path(cache.parent) is None


def test_stray_path_under_the_root_is_left_alone(cache):
    stray = cache / "s-x" / "not-a-slot" / "copy"
    stray.mkdir(parents=True)
    assert db_space.portable_uri(stray.as_uri()) == stray.as_uri()


def test_failed_extraction_leaves_nothing(cache):
    def broken(dest: Path) -> None:
        (dest / "half").write_text("x")
        raise ValueError("bad zip")

    with pytest.raises(ValueError, match="bad zip"):
        materialize("x", "space", "v1", broken, "test")
    assert list(key_dir("x", "space").iterdir()) == [key_dir("x", "space") / ".lock"]


def test_cache_root_must_be_private(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    monkeypatch.setenv("GBSERVER_DB_SPACE_CACHE_DIR", str(shared))
    with pytest.raises(ValueError, match="refusing snapshot cache"):
        _get("v1")
