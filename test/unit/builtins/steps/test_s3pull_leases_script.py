# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/s3/s3pull-leases.sh (list / release s3pull reader leases)."""

import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path("scripts/s3/s3pull-leases.sh")
ENTRY = "a" * 64


def _lease(cache: Path, run_id: str, run_dir: str, version: str = "v1-1-1") -> Path:
    """Write a lease for ENTRY in the s3pull format."""
    leases = cache / f".{ENTRY}.leases"
    leases.mkdir(parents=True, exist_ok=True)
    lease = leases / run_id
    lease.write_text(f"{run_dir}\n{version}\n{int(time.time())}\n")
    return lease


def _script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), *args], capture_output=True, text=True, check=False
    )


def test_list_reports_live_and_dead(tmp_path):
    """A lease whose run dir exists is live; one whose run dir is gone is dead."""
    run = tmp_path / "runs" / "live"
    run.mkdir(parents=True)
    _lease(tmp_path / "cache", "trLive", str(run))
    _lease(tmp_path / "cache", "trDead", str(tmp_path / "gone"))
    result = _script("list", "--cache", str(tmp_path / "cache"))
    assert result.returncode == 0, result.stderr
    rows = {line.split()[2]: line.split()[3] for line in result.stdout.splitlines()[1:]}
    assert rows == {"trLive": "live", "trDead": "dead"}


def test_release_removes_only_that_runs_lease(tmp_path):
    cache = tmp_path / "cache"
    keep = _lease(cache, "trKeep", "-")
    gone = _lease(cache, "trGone", "-")
    result = _script("release", "--cache", str(cache), "trGone")
    assert result.returncode == 0, result.stderr
    assert not gone.exists() and keep.exists()
    assert "1 lease(s) released for trGone" in result.stdout
    assert not (cache / f".{ENTRY}.lock").exists()


def test_release_waits_for_and_respects_the_entry_lock(tmp_path):
    """A held entry lock makes release fail rather than race the holder."""
    cache = tmp_path / "cache"
    lease = _lease(cache, "trX", "-")
    lock = cache / f".{ENTRY}.lock"
    lock.mkdir()
    result = subprocess.run(
        ["bash", str(SCRIPT), "release", "--cache", str(cache), "trX"],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "GB_S3PULL_LOCK_WAIT_SECONDS": "1"},
    )
    assert lease.exists()
    assert "is held" in result.stderr


@pytest.mark.parametrize("bad", ["", "../x", "a/b", ".."])
def test_release_rejects_bad_ids(tmp_path, bad):
    (tmp_path / "cache").mkdir()
    result = _script("release", "--cache", str(tmp_path / "cache"), bad)
    assert result.returncode == 2
