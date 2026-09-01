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

``pull`` serializes concurrent pulls into the same destination behind a
best-effort cross-process lock implemented with atomic ``os.mkdir`` (coherent
across nodes on the shared GPFS/AFM cache, unlike BSD ``flock``). If the lock is
held by a peer past ``GB_HFPULL_LOCK_TIMEOUT`` or cannot be set up, the pull
proceeds anyway and relies on huggingface_hub's per-file locks rather than
failing the build. The lock directory is removed on release, so it does not
accumulate.

Separately, when the HF download cache is corrupt (a ``.incomplete`` file whose
parent dir was removed, or a size-mismatch "Consistency check failed"), ``pull``
self-heals: it retries with ``force_download=True`` and, failing that, drops
HF's scratch download dir -- replacing the manual ``rm -rf`` recovery.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from gbcommon.types.testing import ENV_VAR_GBTEST_MOCK_HF
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
    """Run the real pull() path against a mocked HfApi / snapshot_download.

    The suite defaults GBTEST_MOCK_HF=true (so CI never touches HF); clear it
    here so ``is_hf_mocked()`` is False and pull() exercises the real
    lock/self-heal path instead of short-circuiting to True.
    """
    monkeypatch.delenv(ENV_VAR_GBTEST_MOCK_HF, raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GB_HFPULL_FORCE", raising=False)


def _lock_held(path: Path) -> bool:
    """True while the mkdir lock directory exists (i.e. someone holds it)."""
    return path.exists()


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


def test_repo_pull_holds_lock_during_download_and_removes_it_after(tmp_path):
    """snapshot_download runs while the lock dir exists; removed after."""
    dest = tmp_path / "ibm-granite" / "granite-4.2-8b" / "abc123"
    lock_path = _hfpull_lock_path(dest)
    observed = {}

    def fake_download(*_args, **_kwargs):
        observed["held_during"] = _lock_held(lock_path)

    uri = HfURI.from_parts(
        owner="ibm-granite", repo="granite-4.2-8b", hf_type=HfType.MODEL
    )
    with patch("gbcommon.uri.hf.snapshot_download", side_effect=fake_download):
        result = uri.pull(dest)

    assert result is True
    assert observed.get("held_during") is True, "lock dir not present during download"
    # Released -> the lock dir must be gone (no accumulation on the shared cache).
    assert not lock_path.exists()


def test_bucket_pull_holds_lock_during_sync(tmp_path):
    """sync_bucket also runs under the lock (the bucket branch of pull)."""
    dest = tmp_path / "org" / "my-bucket" / "def456"
    lock_path = _hfpull_lock_path(dest)
    observed = {}

    def fake_sync(*_args, **_kwargs):
        observed["held_during"] = _lock_held(lock_path)

    uri = HfURI.from_parts(owner="org", repo="my-bucket", hf_type=HfType.BUCKET)
    with patch("gbcommon.uri.hf.HfApi") as mock_api:
        mock_api.return_value.sync_bucket.side_effect = fake_sync
        result = uri.pull(dest)

    assert result is True
    assert (
        observed.get("held_during") is True
    ), "lock dir not present during sync_bucket"
    assert not lock_path.exists()


def test_pull_proceeds_when_peer_holds_lock(tmp_path, monkeypatch):
    """A peer holding the lock past the timeout does NOT fail the pull.

    Best-effort: on timeout pull() falls through to the download. It must also
    leave the peer's lock dir intact (it never acquired it).
    """
    dest = tmp_path / "org" / "repo" / "hash"
    lock_path = _hfpull_lock_path(dest)
    lock_path.mkdir(parents=True)  # simulate a peer already holding the lock
    monkeypatch.setenv("GB_HFPULL_LOCK_TIMEOUT", "0")

    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)
    with patch("gbcommon.uri.hf.snapshot_download") as mock_dl:
        result = uri.pull(dest)

    assert result is True
    mock_dl.assert_called_once()
    # We never acquired the peer's lock, so we must not have removed it.
    assert lock_path.exists()


def test_pull_proceeds_when_lock_setup_fails(tmp_path):
    """A lock-infra failure (mkdir raises OSError) must not fail the pull."""
    dest = tmp_path / "org" / "repo" / "hash"
    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)
    with patch(
        "pathlib.Path.mkdir", side_effect=OSError("[Errno 30] Read-only file system")
    ):
        with patch("gbcommon.uri.hf.snapshot_download") as mock_dl:
            result = uri.pull(dest)

    assert result is True
    mock_dl.assert_called_once()


def test_different_dests_do_not_contend(tmp_path, monkeypatch):
    """A lock held for one dest must not block a pull into a different dest."""
    other = tmp_path / "org" / "repo-a" / "h1"
    _hfpull_lock_path(other).mkdir(parents=True)  # peer holds a different dest's lock
    monkeypatch.setenv("GB_HFPULL_LOCK_TIMEOUT", "0")

    dest = tmp_path / "org" / "repo-b" / "h2"
    uri = HfURI.from_parts(owner="org", repo="repo-b", hf_type=HfType.MODEL)
    with patch("gbcommon.uri.hf.snapshot_download") as mock_dl:
        result = uri.pull(dest)

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


def test_repo_pull_does_not_self_heal_when_lock_not_held(tmp_path, monkeypatch):
    """Under the unlocked fall-through, no self-heal runs -- the error propagates.

    When a peer holds the lock past the timeout, pull() proceeds unlocked, so a
    peer may be writing the shared tree concurrently. The self-heal
    (``force_download`` re-download + scratch ``rm -rf``) mutates that tree and
    could pull files out from under a live writer, re-inducing #320. So on a
    recoverable error the unlocked path propagates immediately: no
    ``force_download`` retry and no scratch clear.
    """
    dest = tmp_path / "org" / "repo" / "h"
    _hfpull_lock_path(dest).mkdir(parents=True)  # a peer holds the lock
    monkeypatch.setenv("GB_HFPULL_LOCK_TIMEOUT", "0")
    scratch = dest / ".cache" / "huggingface" / "download"
    scratch.mkdir(parents=True)
    (scratch / "leftover.incomplete").write_text("partial")
    forces = []

    def fake_download(*_args, **kwargs):
        forces.append(kwargs.get("force_download"))
        raise _incomplete_error(dest)

    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)
    with patch("gbcommon.uri.hf.snapshot_download", side_effect=fake_download):
        result = uri.pull(dest)

    assert result is False, "the recoverable error must propagate, not be swallowed"
    # A single attempt only: no force_download retry, no scratch clear.
    assert forces == [False]
    assert scratch.exists(), "scratch dir must NOT be cleared on the unlocked path"
    assert (scratch / "leftover.incomplete").exists()


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

    # A negative value is meaningless -- the acquire loop's deadline is already
    # past so it falls through immediately (it does not hang) -- so reject it.
    monkeypatch.setenv(HFPULL_LOCK_TIMEOUT_ENV, "-1")
    assert _hfpull_lock_timeout() == DEFAULT_HFPULL_LOCK_TIMEOUT_S

    # Non-finite values slip past a ``< 0`` check but make the acquire loop wait
    # indefinitely (the actual hang risk), so reject inf/-inf/nan too.
    for raw in ("inf", "+inf", "-inf", "nan"):
        monkeypatch.setenv(HFPULL_LOCK_TIMEOUT_ENV, raw)
        assert _hfpull_lock_timeout() == DEFAULT_HFPULL_LOCK_TIMEOUT_S

    # Zero is allowed (try-once, immediate fall-through).
    monkeypatch.setenv(HFPULL_LOCK_TIMEOUT_ENV, "0")
    assert _hfpull_lock_timeout() == 0.0
