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
cross-process lock implemented with atomic ``os.mkdir`` (coherent across nodes
on the shared GPFS/AFM cache, unlike BSD ``flock``). A waiter blocks while the
holder keeps writing under the destination and reclaims the lock only after
``GB_HFPULL_LOCK_TIMEOUT`` seconds with no such writes (a dead holder); it falls
through *unlocked* only when the lock filesystem itself is unusable, relying then
on huggingface_hub's per-file locks rather than failing the build. The lock
directory is removed on release, so it does not accumulate.

Separately, when the HF download cache is corrupt (a ``.incomplete`` file whose
parent dir was removed, or a size-mismatch "Consistency check failed"), ``pull``
self-heals: it retries with ``force_download=True`` and, failing that, drops
HF's scratch download dir -- replacing the manual ``rm -rf`` recovery.
"""

import shutil
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from gbcommon.types.testing import ENV_VAR_GBTEST_MOCK_HF
from gbcommon.uri.hf import (
    DEFAULT_HFPULL_LOCK_TTL_S,
    HFPULL_LOCK_TIMEOUT_ENV,
    HFPULL_LOCK_TTL_MAX_S,
    HfType,
    HfURI,
    _hfpull_lock_path,
    _hfpull_lock_ttl,
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


def test_lock_path_is_in_a_dot_prefixed_sibling_container(tmp_path):
    """The lock dir sits one level inside a dot-prefixed sibling container.

    ``.gb-hfpull-locks`` is itself a sibling of the ``<revision>`` dirs (it lives
    at ``<owner>/<repo>/``), but the individual ``<revision>.lock`` dir lives one
    level down inside it, so a ``.lock`` never sits directly beside a revision
    dir, and the dot prefix distinguishes the container from any real revision.
    """
    dest = tmp_path / "ibm-granite" / "granite-4.2-8b" / "abc123"
    lock_path = _hfpull_lock_path(dest)
    assert lock_path.parent != dest.parent  # the .lock is not a revision sibling
    assert lock_path.parent.name == ".gb-hfpull-locks"
    assert lock_path.parent.parent == dest.parent  # the container is, though
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


def test_pull_reclaims_a_dead_holders_stale_lock(tmp_path, monkeypatch):
    """A crashed holder's stale lock (past ttl, no writes) is reclaimed.

    Finding-2 behavior: rather than every future puller stalling behind an
    orphaned lock, a waiter reclaims it once it is past the ttl with no write
    progress under the destination, then proceeds under the lock it now holds.
    """
    dest = tmp_path / "org" / "repo" / "hash"
    lock_path = _hfpull_lock_path(dest)
    lock_path.mkdir(parents=True)  # a peer took it...
    # ...long ago, and never wrote anything under dest (dest does not exist), so
    # it looks dead. ttl=1 with an hour-old start makes it reclaimable at once.
    (lock_path / "lock.info").write_text(f"host:dead|pid:1\n{time.time() - 3600}\n")
    monkeypatch.setenv("GB_HFPULL_LOCK_TIMEOUT", "1")

    held_during = {}

    def fake_download(*_args, **_kwargs):
        # We reclaimed and now hold the lock (our identity recorded in it).
        held_during["ours"] = lock_path.is_dir() and lock_path.joinpath(
            "lock.info"
        ).read_text().startswith("host:")

    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)
    with patch("gbcommon.uri.hf.snapshot_download", side_effect=fake_download):
        result = uri.pull(dest)

    assert result is True
    assert (
        held_during.get("ours") is True
    ), "pull should download under the reclaimed lock"
    # We acquired then released, so the lock dir is gone (not left behind).
    assert not lock_path.exists()


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


def test_repo_pull_does_not_self_heal_on_unlocked_infra_fallthrough(tmp_path):
    """When the lock can't be set up, pull proceeds unlocked and does NOT self-heal.

    The only way ``pull`` runs unlocked now is a lock-infra failure (e.g. a
    read-only mount makes ``mkdir`` fail). On that fall-through a peer may be
    writing the shared tree concurrently, so the self-heal (``force_download``
    re-download + scratch ``rm -rf``) -- which mutates that tree and could pull
    files out from under a live writer, re-inducing #320 -- must not run: a
    recoverable error propagates immediately, with no retry and no scratch clear.
    """
    dest = tmp_path / "org" / "repo" / "h"
    scratch = dest / ".cache" / "huggingface" / "download"
    scratch.mkdir(parents=True)  # created before we break mkdir below
    (scratch / "leftover.incomplete").write_text("partial")
    forces = []

    def fake_download(*_args, **kwargs):
        forces.append(kwargs.get("force_download"))
        raise _incomplete_error(dest)

    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)
    # Break the lock filesystem so acquire() fails -> unlocked fall-through.
    with patch(
        "pathlib.Path.mkdir", side_effect=OSError("[Errno 30] Read-only file system")
    ):
        with patch("gbcommon.uri.hf.snapshot_download", side_effect=fake_download):
            result = uri.pull(dest)

    assert result is False, "the recoverable error must propagate, not be swallowed"
    # A single attempt only: no force_download retry, no scratch clear.
    assert forces == [False]
    assert scratch.exists(), "scratch dir must NOT be cleared on the unlocked path"
    assert (scratch / "leftover.incomplete").exists()


def test_pull_rewaits_when_evicted_after_download(tmp_path):
    """If the lock is reclaimed while we download, we discard and re-wait.

    A long stall can make a live holder look dead, so a peer reclaims the lock.
    The end-of-download ownership fence catches that: the (partly) unlocked
    result is discarded and the pull re-acquires and retries -- it does not treat
    a tree the new owner is writing as authoritative. Here the second attempt
    (after re-acquiring the now-free lock) succeeds.
    """
    dest = tmp_path / "org" / "repo" / "h"
    lock_path = _hfpull_lock_path(dest)
    calls = []

    def fake_download(*_args, **_kwargs):
        calls.append(_kwargs.get("force_download"))
        if len(calls) == 1:
            # Simulate a peer reclaiming (and releasing) our lock mid-download.
            shutil.rmtree(lock_path, ignore_errors=True)
        # both attempts otherwise "succeed"

    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)
    with patch("gbcommon.uri.hf.snapshot_download", side_effect=fake_download):
        result = uri.pull(dest)

    assert result is True
    assert len(calls) == 2, "the evicted first attempt must be discarded and retried"
    assert not lock_path.exists(), "the re-acquired lock is released after success"


def test_pull_rewaits_instead_of_self_healing_under_a_peer(tmp_path):
    """Evicted before a self-heal: don't force/rm under the new owner -- re-wait.

    If we hit a recoverable error but discover we've been reclaimed, running the
    self-heal (force_download / scratch rm -rf) would mutate a tree the new owner
    is writing and re-induce #320. Instead we abandon to the waiter loop; the
    retry after re-acquiring downloads cleanly (no force_download under a peer).
    """
    dest = tmp_path / "org" / "repo" / "h"
    lock_path = _hfpull_lock_path(dest)
    forces = []

    def fake_download(*_args, **_kwargs):
        forces.append(_kwargs.get("force_download"))
        if len(forces) == 1:
            # Recoverable error AND a peer has reclaimed (and released) our lock.
            shutil.rmtree(lock_path, ignore_errors=True)
            raise _incomplete_error(dest)
        # second attempt (after re-acquiring) succeeds

    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)
    with patch("gbcommon.uri.hf.snapshot_download", side_effect=fake_download):
        result = uri.pull(dest)

    assert result is True
    # No force_download anywhere: the evicted attempt refused to self-heal under
    # the peer, and the post-re-wait retry was a fresh normal download.
    assert forces == [False, False], "must re-wait, not force-download under a peer"


def test_pull_rewaits_before_final_scratch_clear_when_evicted(tmp_path):
    """Eviction right before the final scratch rm -rf abandons, not clobbers.

    The last self-heal step (scratch clear + re-download) re-fences immediately
    before it (finding C). If a peer reclaims the lock after the force retry
    fails but before the scratch clear, we must abandon to the re-wait loop --
    NOT rm -rf the scratch dir the new owner may be writing.
    """
    dest = tmp_path / "org" / "repo" / "h"
    lock_path = _hfpull_lock_path(dest)
    scratch = dest / ".cache" / "huggingface" / "download"
    scratch.mkdir(parents=True)
    (scratch / "leftover.incomplete").write_text("partial")
    calls = []

    def fake_download(*_args, **_kwargs):
        calls.append(_kwargs.get("force_download"))
        n = len(calls)
        if n == 1:
            raise _incomplete_error(dest)  # first attempt: recoverable
        if n == 2:
            # Force retry still fails AND a peer reclaims us right here, so the
            # fence before the scratch clear must trip.
            shutil.rmtree(lock_path, ignore_errors=True)
            raise OSError(
                "Consistency check failed: file should be of size 10 but has size 5"
            )
        # n >= 3: after re-acquiring, the retry succeeds.

    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)
    with patch("gbcommon.uri.hf.snapshot_download", side_effect=fake_download):
        result = uri.pull(dest)

    assert result is True
    # normal, force-retry, then (after re-wait) a fresh normal download.
    assert calls == [False, True, False]
    # The evicted attempt must NOT have cleared the scratch dir under the peer.
    assert (scratch / "leftover.incomplete").exists(), "scratch cleared under a peer"


def test_pull_rewaits_when_evicted_during_final_download(tmp_path):
    """Eviction *during* the final self-heal download re-waits, not hard-fails.

    The last self-heal step (scratch clear + re-download) has no following fenced
    step, so an eviction that lands mid-download surfaces as a raw download error.
    Unlike the earlier attempts -- which fall through to a fence that catches the
    eviction -- this one must re-check ownership on failure and abandon to the
    re-wait (like every other eviction point) rather than fail the build. Here the
    retry after re-acquiring the freed lock succeeds.
    """
    dest = tmp_path / "org" / "repo" / "h"
    lock_path = _hfpull_lock_path(dest)
    scratch = dest / ".cache" / "huggingface" / "download"
    scratch.mkdir(parents=True)
    calls = []

    def fake_download(*_args, **_kwargs):
        calls.append(_kwargs.get("force_download"))
        n = len(calls)
        if n == 1:
            raise _incomplete_error(dest)  # recoverable -> force retry
        if n == 2:
            raise OSError(  # recoverable -> scratch clear + final retry
                "Consistency check failed: file should be of size 10 but has size 5"
            )
        if n == 3:
            # Evicted right as the final download runs: a peer reclaims (and
            # releases) our lock, and the download then errors.
            shutil.rmtree(lock_path, ignore_errors=True)
            raise _incomplete_error(dest)
        # n >= 4: after re-acquiring the freed lock, a fresh download succeeds.

    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)
    with patch("gbcommon.uri.hf.snapshot_download", side_effect=fake_download):
        result = uri.pull(dest)

    assert result is True, "an eviction in the final download must re-wait, not fail"
    # normal, force-retry, evicted final, then (after re-wait) a fresh normal pull.
    assert calls == [False, True, True, False]
    assert not lock_path.exists(), "the re-acquired lock is released after success"


def test_pull_propagates_final_download_failure_when_still_owned(tmp_path):
    """A genuine (non-eviction) final-download failure still fails the pull.

    The final-attempt eviction re-check must not swallow real errors: if we still
    hold the lock when the final download raises, that is a genuine failure and
    propagates (pull returns False) with no re-wait.
    """
    dest = tmp_path / "org" / "repo" / "h"
    calls = []

    def fake_download(*_args, **_kwargs):
        calls.append(_kwargs.get("force_download"))
        n = len(calls)
        if n == 1:
            raise _incomplete_error(dest)  # recoverable
        if n == 2:
            raise OSError("Consistency check failed: size 10 vs 5")  # recoverable
        # n == 3: final download fails for an unrelated reason, lock still held.
        raise ValueError("boom")

    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)
    with patch("gbcommon.uri.hf.snapshot_download", side_effect=fake_download):
        result = uri.pull(dest)

    assert result is False, "a real final-download failure must propagate, not re-wait"
    assert calls == [False, True, True], "no re-wait when we still own the lock"


def test_pull_gives_up_after_repeated_eviction(tmp_path, monkeypatch):
    """Repeated eviction is capped: past HFPULL_MAX_LOCK_REWAITS, fail cleanly.

    A reclaim storm (or a node that keeps stalling past the reclaim window) must
    not loop forever -- after the cap the pull returns False instead of retrying.
    """
    import gbcommon.uri.hf as hf

    monkeypatch.setattr(hf, "HFPULL_MAX_LOCK_REWAITS", 3)
    dest = tmp_path / "org" / "repo" / "h"
    lock_path = _hfpull_lock_path(dest)
    calls = []

    def fake_download(*_args, **_kwargs):
        calls.append(1)
        # Always get evicted right after "downloading": drop the lock dir so the
        # end-of-download fence trips and we re-wait, every time.
        shutil.rmtree(lock_path, ignore_errors=True)

    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)
    with patch("gbcommon.uri.hf.snapshot_download", side_effect=fake_download):
        result = uri.pull(dest)

    assert result is False, "must give up (not loop) once the re-wait cap is exceeded"
    # Initial attempt + 3 permitted re-waits = 4 attempts, then it gives up.
    assert len(calls) == 4, f"expected 1 + cap(3) attempts, got {len(calls)}"


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


def test_lock_ttl_reads_env_matching_the_shell_parse(monkeypatch):
    """_hfpull_lock_ttl parses GB_HFPULL_LOCK_TIMEOUT exactly like the shell copies.

    Only plain decimals are accepted (whole-second granularity), so an operator
    gets the same reclaim window on k8s/CLI and LSF/skypilot (issue #322): the
    same grammar, the same rounding, the same fallbacks on both paths.
    """
    monkeypatch.delenv(HFPULL_LOCK_TIMEOUT_ENV, raising=False)
    assert _hfpull_lock_ttl() == DEFAULT_HFPULL_LOCK_TTL_S

    # Whole-second granularity: a decimal is floored to its integer part.
    monkeypatch.setenv(HFPULL_LOCK_TIMEOUT_ENV, "12.5")
    assert _hfpull_lock_ttl() == 12.0

    monkeypatch.setenv(HFPULL_LOCK_TIMEOUT_ENV, "300")
    assert _hfpull_lock_ttl() == 300.0

    # Leading zero is base-10, not octal (matches the shell's 10#).
    monkeypatch.setenv(HFPULL_LOCK_TIMEOUT_ENV, "08")
    assert _hfpull_lock_ttl() == 8.0

    monkeypatch.setenv(HFPULL_LOCK_TIMEOUT_ENV, "not-a-number")
    assert _hfpull_lock_ttl() == DEFAULT_HFPULL_LOCK_TTL_S

    # Scientific notation is rejected here just as the shell rejects it, so the
    # two paths cannot diverge (the finding-3 regression).
    monkeypatch.setenv(HFPULL_LOCK_TIMEOUT_ENV, "1e2")
    assert _hfpull_lock_ttl() == DEFAULT_HFPULL_LOCK_TTL_S

    # Signs and non-finite values are rejected (not plain decimals).
    for raw in ("-1", "+5", "inf", "nan"):
        monkeypatch.setenv(HFPULL_LOCK_TIMEOUT_ENV, raw)
        assert _hfpull_lock_ttl() == DEFAULT_HFPULL_LOCK_TTL_S

    # A positive sub-second value rounds up to the 1s poll granularity, not down
    # to 0 (matches the shell), so it never collapses to "reclaim immediately".
    monkeypatch.setenv(HFPULL_LOCK_TIMEOUT_ENV, "0.5")
    assert _hfpull_lock_ttl() == 1.0

    # Explicit zero stays zero (reclaim a stalled peer at once).
    monkeypatch.setenv(HFPULL_LOCK_TIMEOUT_ENV, "0")
    assert _hfpull_lock_ttl() == 0.0
    monkeypatch.setenv(HFPULL_LOCK_TIMEOUT_ENV, "0.0")
    assert _hfpull_lock_ttl() == 0.0

    # Clamp to the max (matches the shell clamp, and keeps bash 64-bit arithmetic
    # from overflowing on an absurd value and silently re-diverging).
    monkeypatch.setenv(HFPULL_LOCK_TIMEOUT_ENV, str(int(HFPULL_LOCK_TTL_MAX_S)))
    assert _hfpull_lock_ttl() == HFPULL_LOCK_TTL_MAX_S  # exactly max: unchanged
    monkeypatch.setenv(HFPULL_LOCK_TIMEOUT_ENV, "99999999")
    assert _hfpull_lock_ttl() == HFPULL_LOCK_TTL_MAX_S  # above max: clamped
    monkeypatch.setenv(HFPULL_LOCK_TIMEOUT_ENV, "9" * 24)
    assert _hfpull_lock_ttl() == HFPULL_LOCK_TTL_MAX_S  # would overflow bash: clamped
    # Leading zeros don't inflate the magnitude past the clamp.
    monkeypatch.setenv(HFPULL_LOCK_TIMEOUT_ENV, "0000000005")
    assert _hfpull_lock_ttl() == 5.0
