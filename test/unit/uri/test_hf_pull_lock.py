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

"""Cross-process file-lock behavior of ``HfURI.pull`` (issue #315/#320).

``HfApi.sync_bucket`` / ``snapshot_download`` are not multi-process safe: when
several hfpull containers pull the same hf:// URI into the same shared cache
directory they race on HuggingFace's ``.cache/huggingface/download/*.incomplete``
files. ``HfURI.pull`` therefore serializes the download behind a
``filelock.FileLock`` keyed to the destination, so concurrent pulls of the same
repo/revision run one at a time while different destinations proceed in parallel.
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
    _hfpull_lock_timeout,
)


@pytest.fixture(autouse=True)
def _disable_hf_op_mocking(monkeypatch):
    """Run the real pull() path against a mocked HfApi / snapshot_download."""
    monkeypatch.delenv(ENV_VAR_GBTEST_MOCKED_HF_OPS, raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)


def _expected_lock(dest: Path) -> Path:
    """The sibling lock file pull() is expected to use for *dest*."""
    return dest.parent / f"{dest.name}.lock"


def _is_locked(path: Path) -> bool:
    """True if *path* is currently held by some other FileLock holder."""
    probe = FileLock(str(path))
    try:
        probe.acquire(timeout=0)
    except Timeout:
        return True
    probe.release()
    return False


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


def test_pull_returns_false_when_lock_cannot_be_acquired(tmp_path, monkeypatch):
    """A held lock past the timeout fails the pull without downloading."""
    dest = tmp_path / "org" / "repo" / "hash"
    dest.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _expected_lock(dest)
    monkeypatch.setenv("GB_HFPULL_LOCK_TIMEOUT", "0.5")

    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)
    held = FileLock(str(lock_path))
    held.acquire()
    try:
        with patch("gbcommon.uri.hf.snapshot_download") as mock_dl:
            result = uri.pull(dest)
    finally:
        held.release()

    assert result is False
    mock_dl.assert_not_called()


def test_different_dests_do_not_contend(tmp_path, monkeypatch):
    """A lock held for one dest must not block a pull into a different dest."""
    other = tmp_path / "org" / "repo-a" / "h1"
    other.parent.mkdir(parents=True, exist_ok=True)
    held = FileLock(str(_expected_lock(other)))
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


def test_lock_timeout_reads_env_with_default(monkeypatch):
    """_hfpull_lock_timeout parses GB_HFPULL_LOCK_TIMEOUT, falling back to default."""
    monkeypatch.delenv(HFPULL_LOCK_TIMEOUT_ENV, raising=False)
    assert _hfpull_lock_timeout() == DEFAULT_HFPULL_LOCK_TIMEOUT_S

    monkeypatch.setenv(HFPULL_LOCK_TIMEOUT_ENV, "12.5")
    assert _hfpull_lock_timeout() == 12.5

    monkeypatch.setenv(HFPULL_LOCK_TIMEOUT_ENV, "not-a-number")
    assert _hfpull_lock_timeout() == DEFAULT_HFPULL_LOCK_TIMEOUT_S
