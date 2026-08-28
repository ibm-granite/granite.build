#!/usr/bin/env python3

# Copyright Granite.Build Authors
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

"""Download serialization and self-healing of ``HfURI.pull`` (issue #320).

huggingface_hub already protects each file's ``.incomplete`` write with its own
per-file lock inside the destination, so ``pull`` adds only a *best-effort*
cross-process ``filelock.FileLock`` keyed to the destination: concurrent pulls
of the same repo/revision run one at a time, but if the lock can't be acquired
(timeout, or a mount that doesn't honor advisory locks) the pull proceeds and
relies on HF's per-file locks rather than failing the build.

Separately, when the HF download cache is corrupt (a ``.incomplete`` file whose
parent dir was removed, or a size-mismatch "Consistency check failed"), ``pull``
self-heals: it retries with ``force_download=True`` and, failing that, drops
HF's scratch download dir -- replacing the manual ``rm -rf`` recovery.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from filelock import FileLock, Timeout

from gbcommon.types.testing import ENV_VAR_GBTEST_MOCKED_HF_OPS
from gbcommon.uri.hf import (
    DEFAULT_HFPULL_LOCK_TIMEOUT_S,
    HFPULL_LOCK_TIMEOUT_ENV,
    HfType,
    HfURI,
    _hfpull_lock_path,
    _hfpull_lock_timeout,
)


@pytest.fixture(autouse=True)
def _disable_hf_op_mocking(monkeypatch):
    """Run the real pull() path against a mocked HfApi / snapshot_download."""
    monkeypatch.delenv(ENV_VAR_GBTEST_MOCKED_HF_OPS, raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GB_HFPULL_FORCE", raising=False)


def _expected_lock(dest: Path) -> Path:
    """The lock file pull() is expected to use for *dest*."""
    return _hfpull_lock_path(dest)


def _is_locked(path: Path) -> bool:
    """True if *path* is currently held by some other FileLock holder."""
    probe = FileLock(str(path))
    try:
        probe.acquire(timeout=0)
    except Timeout:
        return True
    probe.release()
    return False


def _incomplete_error(dest: Path) -> FileNotFoundError:
    """The #320 failure: a vanished ``.incomplete`` under a held per-file lock."""
    incomplete = dest / ".cache/huggingface/download/IO4x.etag123.incomplete"
    return FileNotFoundError(2, "No such file or directory", str(incomplete))


def test_lock_path_is_outside_the_revision_namespace(tmp_path):
    """The lock lives in a dedicated subdir, not beside revision dirs."""
    dest = tmp_path / "ibm-granite" / "granite-4.2-8b" / "abc123"
    lock_path = _hfpull_lock_path(dest)
    # Not a sibling of the revision dir (would pollute the owner/repo glob).
    assert lock_path.parent != dest.parent
    assert lock_path.parent.name == ".gb-hfpull-locks"
    assert lock_path.name == "abc123.lock"


def test_repo_pull_holds_lock_during_download_and_releases_after(tmp_path):
    """snapshot_download runs while the per-dest lock is held; released after."""
    dest = tmp_path / "ibm-granite" / "granite-4.2-8b" / "abc123"
    lock_path = _expected_lock(dest)
    observed = {}

    def fake_download(*_args, **_kwargs):
        observed["locked_during"] = _is_locked(lock_path)

    uri = HfURI.from_parts(
        owner="ibm-granite", repo="granite-4.2-8b", hf_type=HfType.MODEL
    )
    with patch("gbcommon.uri.hf.snapshot_download", side_effect=fake_download):
        result = uri.pull(dest)

    assert result is True
    assert observed.get("locked_during") is True, "lock not held during download"
    # After a successful pull the lock must be released.
    assert _is_locked(lock_path) is False


def test_bucket_pull_holds_lock_during_sync(tmp_path):
    """sync_bucket also runs under the lock (the bucket branch of pull)."""
    dest = tmp_path / "org" / "my-bucket" / "def456"
    lock_path = _expected_lock(dest)
    observed = {}

    def fake_sync(*_args, **_kwargs):
        observed["locked_during"] = _is_locked(lock_path)

    uri = HfURI.from_parts(owner="org", repo="my-bucket", hf_type=HfType.BUCKET)
    with patch("gbcommon.uri.hf.HfApi") as MockApi:
        MockApi.return_value.sync_bucket.side_effect = fake_sync
        result = uri.pull(dest)

    assert result is True
    assert observed.get("locked_during") is True, "lock not held during sync_bucket"
    assert _is_locked(lock_path) is False


def test_pull_proceeds_when_lock_not_acquired(tmp_path, monkeypatch):
    """A peer holding the lock past the timeout does NOT fail the pull.

    The lock is best-effort: on timeout pull() falls through to the download
    and relies on huggingface_hub's own per-file locks.
    """
    dest = tmp_path / "org" / "repo" / "hash"
    lock_path = _expected_lock(dest)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("GB_HFPULL_LOCK_TIMEOUT", "0.5")

    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)
    held = FileLock(str(lock_path))
    held.acquire()
    try:
        with patch("gbcommon.uri.hf.snapshot_download") as mock_dl:
            result = uri.pull(dest)
    finally:
        held.release()

    assert result is True
    mock_dl.assert_called_once()


@pytest.mark.parametrize(
    "exc",
    [
        NotImplementedError("FileSystem does not appear to support flock"),
        OSError("[Errno 30] Read-only file system"),
    ],
)
def test_pull_proceeds_when_lock_acquire_raises(tmp_path, exc):
    """A lock-infra failure at acquire time must not fail the pull (best-effort).

    Covers the unsupported-mount (NotImplementedError) and flaky-mount (OSError)
    branches: the download proceeds and relies on HF's per-file locks.
    """
    dest = tmp_path / "org" / "repo" / "hash"
    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)
    with patch("filelock.FileLock.acquire", side_effect=exc):
        with patch("gbcommon.uri.hf.snapshot_download") as mock_dl:
            result = uri.pull(dest)

    assert result is True
    mock_dl.assert_called_once()


def test_different_dests_do_not_contend(tmp_path, monkeypatch):
    """A lock held for one dest must not block a pull into a different dest."""
    other = tmp_path / "org" / "repo-a" / "h1"
    other_lock = _expected_lock(other)
    other_lock.parent.mkdir(parents=True, exist_ok=True)
    held = FileLock(str(other_lock))
    held.acquire()
    monkeypatch.setenv("GB_HFPULL_LOCK_TIMEOUT", "0.5")

    dest = tmp_path / "org" / "repo-b" / "h2"
    uri = HfURI.from_parts(owner="org", repo="repo-b", hf_type=HfType.MODEL)
    try:
        with patch("gbcommon.uri.hf.snapshot_download") as mock_dl:
            result = uri.pull(dest)
    finally:
        held.release()

    assert result is True
    mock_dl.assert_called_once()


def test_repo_pull_recovers_from_removed_incomplete_dir(tmp_path):
    """A vanished ``.incomplete`` triggers a single force_download retry."""
    dest = tmp_path / "ibm-granite" / "granite-4.2-8b" / "abc123"
    forces = []

    def fake_download(*_args, **kwargs):
        forces.append(kwargs.get("force_download"))
        if len(forces) == 1:
            raise _incomplete_error(dest)
        # second (forced) call succeeds

    uri = HfURI.from_parts(
        owner="ibm-granite", repo="granite-4.2-8b", hf_type=HfType.MODEL
    )
    with patch("gbcommon.uri.hf.snapshot_download", side_effect=fake_download):
        result = uri.pull(dest)

    assert result is True
    assert forces == [False, True], "expected one normal then one forced download"


def test_repo_pull_clears_scratch_when_force_retry_still_fails(tmp_path):
    """If the force retry also fails, the scratch download dir is dropped."""
    dest = tmp_path / "org" / "repo" / "h"
    scratch = dest / ".cache" / "huggingface" / "download"
    scratch.mkdir(parents=True)
    (scratch / "leftover.incomplete").write_text("partial")
    forces = []
    seen = {}

    def fake_download(*_args, **kwargs):
        forces.append(kwargs.get("force_download"))
        if len(forces) == 1:
            raise _incomplete_error(dest)
        if len(forces) == 2:
            raise OSError(
                "Consistency check failed: file should be of size 10 but has "
                "size 5 (model-00001-of-00004.safetensors)."
            )
        # third call: scratch must have been cleared before this retry
        seen["scratch_exists"] = scratch.exists()

    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)
    with patch("gbcommon.uri.hf.snapshot_download", side_effect=fake_download):
        result = uri.pull(dest)

    assert result is True
    assert forces == [False, True, True]
    assert seen.get("scratch_exists") is False, "scratch dir not cleared"


def test_repo_pull_does_not_retry_non_recoverable_error(tmp_path):
    """An unrelated error is not treated as corruption; no retry, pull fails."""
    dest = tmp_path / "org" / "repo" / "h"
    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)
    with patch(
        "gbcommon.uri.hf.snapshot_download", side_effect=ValueError("boom")
    ) as mock_dl:
        result = uri.pull(dest)

    assert result is False
    mock_dl.assert_called_once()


def test_hfpull_step_force_env_forces_pull(tmp_path, monkeypatch):
    """GB_HFPULL_FORCE makes hfpull_step call pull(force=True)."""
    monkeypatch.setenv("GB_HFPULL_FORCE", "1")
    with patch.object(HfURI, "pull", return_value=True) as mock_pull:
        rc = HfURI.hfpull_step("hf:///org/repo", str(tmp_path / "dest"))

    assert rc == 0
    assert mock_pull.call_args.kwargs.get("force") is True


def test_lock_timeout_reads_env_with_default(monkeypatch):
    """_hfpull_lock_timeout parses GB_HFPULL_LOCK_TIMEOUT, falling back to default."""
    monkeypatch.delenv(HFPULL_LOCK_TIMEOUT_ENV, raising=False)
    assert _hfpull_lock_timeout() == DEFAULT_HFPULL_LOCK_TIMEOUT_S

    monkeypatch.setenv(HFPULL_LOCK_TIMEOUT_ENV, "12.5")
    assert _hfpull_lock_timeout() == 12.5

    monkeypatch.setenv(HFPULL_LOCK_TIMEOUT_ENV, "not-a-number")
    assert _hfpull_lock_timeout() == DEFAULT_HFPULL_LOCK_TIMEOUT_S

    # A negative value would make FileLock.acquire block forever; reject it.
    monkeypatch.setenv(HFPULL_LOCK_TIMEOUT_ENV, "-1")
    assert _hfpull_lock_timeout() == DEFAULT_HFPULL_LOCK_TIMEOUT_S

    # Zero is allowed (try-once, immediate fall-through).
    monkeypatch.setenv(HFPULL_LOCK_TIMEOUT_ENV, "0")
    assert _hfpull_lock_timeout() == 0.0
