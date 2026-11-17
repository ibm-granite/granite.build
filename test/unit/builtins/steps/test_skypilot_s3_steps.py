# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for the Skypilot s3pull/s3push builtin step scripts.

Each test renders the step's ``run`` block through gbserver's template engine
(as the launcher does) and executes it with bash against a stub ``aws`` on PATH
that records its arguments and fakes downloads. This exercises the shell logic
(object vs prefix, file vs directory, mocking, credential and empty-input
checks, and s3pull's shared versioned cache and its lock) without network access
or a cluster.
"""

import os
import stat
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import pytest
import yaml

from gbserver.utils.template import fill_template

STEPS = Path("src/gbserver/builtins/steps/skypilot")

# Records each invocation and fakes downloads into a local destination:
# - head-object succeeds only when STUB_HEAD_RC=0, printing ETag STUB_ETAG;
#   otherwise it fails with STUB_HEAD_ERR (default: a 404 Not Found);
# - a download sync writes the objects named in STUB_FILES (default "f") with
#   content STUB_CONTENT, each via a temp file + rename like the real CLI (so a
#   hard-linked previous version is never modified); --delete removes files not
#   in STUB_FILES; STUB_EMPTY makes the prefix empty; STUB_SLEEP pauses inside
#   the sync, which is bracketed by sync-start/sync-end log lines.
STUB_AWS = """#!/usr/bin/env bash
echo "$*" >> "${STUB_LOG}"
last="${!#}"
case " $* " in
  *" s3api head-object "*)
    if [[ "${STUB_HEAD_RC:-1}" != 0 ]]; then
      echo "${STUB_HEAD_ERR:-An error occurred (404) when calling the HeadObject operation: Not Found}" >&2
      exit "${STUB_HEAD_RC:-254}"
    fi
    echo "{\\"ETag\\": \\"${STUB_ETAG:-e1}\\", \\"ContentLength\\": 5, \\"LastModified\\": \\"2026-01-01T00:00:00+00:00\\"}"
    exit 0 ;;
  *" s3 cp "*) [[ "${last}" == s3://* ]] || { mkdir -p "$(dirname "${last}")"; echo d > "${last}"; } ;;
  *" s3 sync "*)
    [[ "${last}" == s3://* ]] && exit 0
    echo "sync-start" >> "${STUB_LOG}"
    files="${STUB_FILES:-f}"
    [[ -n "${STUB_EMPTY:-}" ]] && files=""
    dir="${last%/}"
    mkdir -p "${dir}"
    if [[ " $* " == *" --delete "* ]]; then
      for existing in "${dir}"/*; do
        [ -e "${existing}" ] || continue
        [[ " ${files} " == *" $(basename "${existing}") "* ]] || rm -f "${existing}"
      done
    fi
    for name in ${files}; do
      echo "${STUB_CONTENT:-d}" > "${dir}/${name}.part" && mv "${dir}/${name}.part" "${dir}/${name}"
    done
    sleep "${STUB_SLEEP:-0}"
    echo "sync-end" >> "${STUB_LOG}"
    ;;
esac
exit 0
"""


def _launcher(kind: str) -> Dict:
    """Return the Skypilot launcher config of the ``kind`` step yaml."""
    data = yaml.safe_load((STEPS / kind / "step.yaml").read_text())
    return data["environment_configs"]["Skypilot"]["launchers"][kind]["config"]


def _prepare(
    tmp_path: Path,
    kind: str,
    s3_uri: str,
    local_path: Path,
    extra_env: Optional[Dict[str, str]] = None,
    creds: bool = True,
    step_config: Optional[Dict] = None,
) -> Tuple[str, Dict[str, str]]:
    """Render a step's run script and build its env, with the stub aws on PATH.

    :param tmp_path: pytest tmp dir (holds the stub and its log).
    :param kind: ``s3pull`` or ``s3push``.
    :param s3_uri: the configured s3_uri.
    :param local_path: the configured local_path.
    :param extra_env: additional env vars for the script.
    :param creds: whether to set the (dummy) AWS credential env vars.
    :param step_config: extra ``<kind>_config`` keys (e.g. cache_max_age_days).
    :returns: ``(script, env)``.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    stub = bindir / "aws"
    stub.write_text(STUB_AWS)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    config = {
        f"{kind}_config": {
            "uri": "s3://local/artifact",
            "binding_id": "bind-1",
            "s3_uri": s3_uri,
            "local_path": str(local_path),
            "endpoint_url": "http://gb-s3:9000",
            "region": "us-east-1",
            **(step_config or {}),
        }
    }
    script = fill_template(_launcher(kind)["run"], {"config": config}, strict=True)
    env = {"PATH": f"{bindir}:{os.environ['PATH']}", "STUB_LOG": str(tmp_path / "log")}
    if creds:
        env.update(AWS_ACCESS_KEY_ID="dummy-id", AWS_SECRET_ACCESS_KEY="dummy-key")
    env.update(extra_env or {})
    return script, env


def _run(
    tmp_path: Path,
    kind: str,
    s3_uri: str,
    local_path: Path,
    extra_env: Optional[Dict[str, str]] = None,
    creds: bool = True,
    step_config: Optional[Dict] = None,
) -> subprocess.CompletedProcess:
    """Render and execute a step's run script against the stub aws.

    Arguments as for :func:`_prepare`.

    :returns: the completed process (stdout/stderr captured).
    """
    script, env = _prepare(
        tmp_path, kind, s3_uri, local_path, extra_env, creds, step_config
    )
    return subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, check=False
    )


def _versions(dest: Path) -> list:
    """The version directories behind an s3pull cache entry ``dest``."""
    root = dest.parent / f".{dest.name}.versions"
    return sorted(v.name for v in root.iterdir() if v.is_dir()) if root.exists() else []


def _log(tmp_path: Path) -> str:
    """Return the stub aws invocation log ('' if aws was never called)."""
    log = tmp_path / "log"
    return log.read_text() if log.exists() else ""


class TestStepYamls:
    @pytest.mark.parametrize("kind", ["s3pull", "s3push"])
    def test_no_image_and_cli_installed_in_setup(self, kind):
        """The AWS CLI is installed in setup (no container image dependency)."""
        launcher = _launcher(kind)
        assert "image_id" not in launcher
        assert "pip install --no-cache-dir awscli" in launcher["setup"]

    @pytest.mark.parametrize("kind", ["s3pull", "s3push"])
    def test_hostile_values_stay_quoted(self, tmp_path, kind):
        """A URI with quotes/command substitution is passed through literally."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "a").write_text("x")
        result = _run(tmp_path, kind, "s3://b/k'$(touch pwned)", src)
        assert not (Path.cwd() / "pwned").exists()
        assert "s3://b/k'$(touch pwned)" in result.stdout + _log(tmp_path)


class TestS3PullScript:
    def test_object_is_copied_under_basename(self, tmp_path):
        """An existing object is copied into local_path under its basename."""
        dest = tmp_path / "dest"
        result = _run(
            tmp_path, "s3pull", "s3://b/dir/data.jsonl", dest, {"STUB_HEAD_RC": "0"}
        )
        assert result.returncode == 0, result.stderr
        assert (dest / "data.jsonl").exists()
        assert "--endpoint-url http://gb-s3:9000 s3 cp" in _log(tmp_path)

    def test_prefix_is_synced(self, tmp_path):
        """A non-object URI is synced as a prefix."""
        dest = tmp_path / "dest"
        result = _run(tmp_path, "s3pull", "s3://b/dir", dest)
        assert result.returncode == 0, result.stderr
        assert "s3 sync --delete --only-show-errors s3://b/dir/" in _log(tmp_path)

    def test_empty_prefix_fails(self, tmp_path):
        """Pulling nothing is an error, not an empty input."""
        result = _run(
            tmp_path, "s3pull", "s3://b/none", tmp_path / "d", {"STUB_EMPTY": "1"}
        )
        assert result.returncode != 0
        assert "no objects found" in result.stderr

    @pytest.mark.parametrize(
        "error",
        [
            "An error occurred (403) when calling the HeadObject operation: Forbidden",
            'Could not connect to the endpoint URL: "http://gb-s3:9000/b/dir/data.jsonl"',
        ],
    )
    def test_head_object_errors_fail_loudly(self, tmp_path, error):
        """A 403 or network error is reported, not hidden behind a prefix sync."""
        dest = tmp_path / "cache" / "h"
        result = _run(
            tmp_path,
            "s3pull",
            "s3://b/dir/data.jsonl",
            dest,
            {"STUB_HEAD_RC": "254", "STUB_HEAD_ERR": error},
        )
        assert result.returncode != 0
        assert "cannot read s3://b/dir/data.jsonl" in result.stderr
        assert error in result.stderr
        assert "no objects found" not in result.stderr
        assert " s3 sync " not in _log(tmp_path)
        assert not dest.exists()
        assert not (dest.parent / ".h.lock").exists()

    def test_not_found_key_is_pulled_as_a_prefix(self, tmp_path):
        """Only a genuine 404 makes the URI a prefix."""
        dest = tmp_path / "cache" / "h"
        result = _run(tmp_path, "s3pull", "s3://b/dir", dest)
        assert result.returncode == 0, result.stderr
        assert "s3 sync --delete --only-show-errors s3://b/dir/" in _log(tmp_path)

    def test_missing_credentials_fail(self, tmp_path):
        """Without credentials the step fails before calling aws."""
        result = _run(tmp_path, "s3pull", "s3://b/k", tmp_path / "d", creds=False)
        assert result.returncode != 0
        assert "AWS_ACCESS_KEY_ID" in result.stderr
        assert _log(tmp_path) == ""

    def test_mocked_writes_marker_without_aws(self, tmp_path):
        """GBTEST_MOCK_COS skips the network and leaves a marker file."""
        dest = tmp_path / "d"
        result = _run(
            tmp_path,
            "s3pull",
            "s3://b/k",
            dest,
            {"GBTEST_MOCK_COS": "true"},
            creds=False,
        )
        assert result.returncode == 0, result.stderr
        assert (dest / ".gbtest_mock_s3pull").exists()
        assert _log(tmp_path) == ""


class TestS3PullSharedCache:
    """local_path is a symlink to a complete version; pulls are serialized by a
    per-URI lock; a re-pull never modifies files a reader may be using."""

    def test_dest_is_a_symlink_to_a_version_and_the_lock_is_released(self, tmp_path):
        dest = tmp_path / "cache" / "h"
        result = _run(tmp_path, "s3pull", "s3://b/dir", dest)
        assert result.returncode == 0, result.stderr
        assert dest.is_symlink() and (dest / "f").exists()
        assert len(_versions(dest)) == 1
        assert not (dest.parent / ".h.lock").exists()

    def test_repull_drops_objects_deleted_upstream(self, tmp_path):
        """Later pulls don't see deleted objects; old versions are pruned."""
        dest = tmp_path / "cache" / "h"
        for files in ("a b", "a", "a"):
            result = _run(tmp_path, "s3pull", "s3://b/dir", dest, {"STUB_FILES": files})
            assert result.returncode == 0, result.stderr
        assert sorted(f.name for f in dest.iterdir()) == ["a"]
        assert len(_versions(dest)) == 2  # the live one and the one it replaced

    def test_repull_never_modifies_the_previous_version(self, tmp_path):
        """A reader of the old version keeps its content after a re-pull."""
        dest = tmp_path / "cache" / "h"
        _run(tmp_path, "s3pull", "s3://b/dir", dest, {"STUB_CONTENT": "one"})
        old_file = dest.resolve() / "f"
        result = _run(tmp_path, "s3pull", "s3://b/dir", dest, {"STUB_CONTENT": "two"})
        assert result.returncode == 0, result.stderr
        assert old_file.read_text().strip() == "one"
        assert (dest / "f").read_text().strip() == "two"

    def test_concurrent_pulls_are_serialized(self, tmp_path):
        """Two simultaneous pulls of one URI never sync at the same time."""
        dest = tmp_path / "cache" / "h"
        procs = []
        for _ in range(2):
            script, env = _prepare(
                tmp_path, "s3pull", "s3://b/dir", dest, {"STUB_SLEEP": "1"}
            )
            procs.append(
                subprocess.Popen(
                    ["bash", "-c", script],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        for proc in procs:
            _, err = proc.communicate(timeout=60)
            assert proc.returncode == 0, err
        marks = [ln for ln in _log(tmp_path).splitlines() if ln.startswith("sync-")]
        assert marks == ["sync-start", "sync-end", "sync-start", "sync-end"]
        assert (dest / "f").exists()

    def test_stale_lock_is_reclaimed(self, tmp_path):
        """A lock its dead holder stopped refreshing is taken over."""
        dest = tmp_path / "cache" / "h"
        lock = dest.parent / ".h.lock"
        lock.mkdir(parents=True)
        (lock / "owner").write_text("host:gone|pid:1")
        os.utime(lock, (0, 0))
        result = _run(tmp_path, "s3pull", "s3://b/dir", dest)
        assert result.returncode == 0, result.stderr
        assert "reclaimed stale lock" in result.stderr

    def test_lock_retaken_during_reclaim_is_put_back(self, tmp_path):
        """The race: after this pull judges the lock stale, another waiter
        reclaims it and takes a fresh one. This pull must not move the fresh
        lock away and take it too; it puts it back and keeps waiting."""
        dest = tmp_path / "cache" / "h"
        lock = dest.parent / ".h.lock"
        lock.mkdir(parents=True)
        (lock / "owner").write_text("host:dead|pid:1")
        os.utime(lock, (0, 0))
        retake = tmp_path / "retake.sh"
        retake.write_text(
            '#!/usr/bin/env bash\nrm -rf "$1"; mkdir "$1"; echo "host:A|pid:2" > "$1/owner"\n'
        )
        retake.chmod(0o755)
        result = _run(
            tmp_path,
            "s3pull",
            "s3://b/dir",
            dest,
            {
                "GBTEST_S3PULL_AFTER_STALE_CHECK": str(retake),
                "GB_S3PULL_LOCK_WAIT_SECONDS": "2",
            },
        )
        assert result.returncode != 0
        assert "re-taken during reclaim; put it back" in result.stderr
        assert "timed out" in result.stderr
        assert (lock / "owner").read_text().strip() == "host:A|pid:2"
        assert not list(dest.parent.glob(".h.lock.stale.*"))
        assert " s3 sync " not in _log(tmp_path)

    def test_live_lock_times_out_and_is_left_alone(self, tmp_path):
        """A held, fresh lock makes the pull wait, then fail; it isn't stolen."""
        dest = tmp_path / "cache" / "h"
        lock = dest.parent / ".h.lock"
        lock.mkdir(parents=True)
        (lock / "owner").write_text("host:other|pid:2")
        result = _run(
            tmp_path, "s3pull", "s3://b/dir", dest, {"GB_S3PULL_LOCK_WAIT_SECONDS": "2"}
        )
        assert result.returncode != 0
        assert "timed out" in result.stderr
        assert (lock / "owner").read_text() == "host:other|pid:2"

    def test_legacy_directory_is_adopted(self, tmp_path):
        """A real directory from the old layout becomes a version, then is synced."""
        dest = tmp_path / "cache" / "h"
        dest.mkdir(parents=True)
        (dest / "stale").write_text("x")
        result = _run(tmp_path, "s3pull", "s3://b/dir", dest, {"STUB_FILES": "new"})
        assert result.returncode == 0, result.stderr
        assert dest.is_symlink()
        assert sorted(f.name for f in dest.iterdir()) == ["new"]

    def test_unchanged_object_is_not_downloaded_again(self, tmp_path):
        """A re-pull of an object with the same signature keeps the live copy."""
        dest = tmp_path / "cache" / "h"
        env = {"STUB_HEAD_RC": "0", "STUB_ETAG": "e1"}
        for _ in range(2):
            result = _run(tmp_path, "s3pull", "s3://b/dir/model.bin", dest, env)
            assert result.returncode == 0, result.stderr
        assert "is unchanged; keeping the cached copy" in result.stdout
        assert _log(tmp_path).count(" s3 cp ") == 1
        assert (dest / "model.bin").exists()
        assert len(_versions(dest)) == 1
        assert not (dest.parent / ".h.lock").exists()

    def test_changed_object_is_downloaded_again(self, tmp_path):
        """A different ETag means a new download and a new live version."""
        dest = tmp_path / "cache" / "h"
        for etag in ("e1", "e2"):
            result = _run(
                tmp_path,
                "s3pull",
                "s3://b/dir/model.bin",
                dest,
                {"STUB_HEAD_RC": "0", "STUB_ETAG": etag},
            )
            assert result.returncode == 0, result.stderr
        assert _log(tmp_path).count(" s3 cp ") == 2
        sig = dest.parent / ".h.versions" / f"{dest.resolve().name}.sig"
        assert sig.read_text().startswith("e2|")

    def test_failed_pull_keeps_the_live_version(self, tmp_path):
        """An empty re-pull fails without touching the live version."""
        dest = tmp_path / "cache" / "h"
        _run(tmp_path, "s3pull", "s3://b/dir", dest, {"STUB_FILES": "a"})
        before = _versions(dest)
        result = _run(tmp_path, "s3pull", "s3://b/dir", dest, {"STUB_EMPTY": "1"})
        assert result.returncode != 0
        assert (dest / "a").exists()
        assert _versions(dest) == before
        assert not (dest.parent / ".h.lock").exists()


DAY = 86400
OLD = int(time.time()) - 40 * DAY  # older than the 30-day default
FRESH = int(time.time()) - 1 * DAY


def _fake_entry(cache: Path, name: str, epoch: int, sig: Optional[str] = None) -> Path:
    """Create a versioned cache entry ``name`` whose live version was built at ``epoch``.

    :param cache: the cache directory.
    :param name: the entry name (a 64-hex URI hash for eviction candidates).
    :param epoch: the epoch encoded in the live version's name.
    :param sig: optional object signature to record for that version.
    :returns: the entry's symlink path.
    """
    version = f"v{epoch}-1-1"
    root = cache / f".{name}.versions"
    (root / version).mkdir(parents=True)
    (root / version / "model.bin").write_text("cached")
    if sig is not None:
        (root / f"{version}.sig").write_text(sig + "\n")
    entry = cache / name
    entry.symlink_to(f".{name}.versions/{version}")
    return entry


class TestS3PullEviction:
    """After a pull, entries downloaded more than cache_max_age_days ago are
    removed under their own lock; only URI-hash names are considered."""

    def test_old_entries_are_evicted_and_fresh_ones_kept(self, tmp_path):
        cache = tmp_path / "cache"
        old = _fake_entry(cache, "a" * 64, OLD)
        fresh = _fake_entry(cache, "b" * 64, FRESH)
        result = _run(tmp_path, "s3pull", "s3://b/dir", cache / "h")
        assert result.returncode == 0, result.stderr
        assert not old.is_symlink()
        assert not (cache / f".{'a' * 64}.versions").exists()
        assert fresh.is_symlink() and (fresh / "model.bin").exists()
        assert f"evicted cache entry {'a' * 64}" in result.stdout
        assert not (cache / f".{'a' * 64}.lock").exists()

    def test_locked_old_entry_is_skipped(self, tmp_path):
        """An entry being pulled (its lock held) is left alone, lock included."""
        cache = tmp_path / "cache"
        old = _fake_entry(cache, "a" * 64, OLD)
        lock = cache / f".{'a' * 64}.lock"
        lock.mkdir()
        (lock / "owner").write_text("host:other|pid:2")
        result = _run(tmp_path, "s3pull", "s3://b/dir", cache / "h")
        assert result.returncode == 0, result.stderr
        assert old.is_symlink()
        assert (lock / "owner").read_text() == "host:other|pid:2"
        assert "it is locked" in result.stdout

    def test_only_uri_hash_names_are_considered(self, tmp_path):
        """Other content of a user-supplied cache_path is never touched."""
        cache = tmp_path / "cache"
        cache.mkdir()
        keep = [cache / "notes", cache / ("a" * 63), cache / "README"]
        keep[0].mkdir()
        keep[1].mkdir()
        keep[2].write_text("x")
        for path in keep:
            os.utime(path, (OLD, OLD))
        result = _run(tmp_path, "s3pull", "s3://b/dir", cache / "h")
        assert result.returncode == 0, result.stderr
        assert all(path.exists() for path in keep)

    def test_old_legacy_dir_and_orphaned_versions_are_evicted(self, tmp_path):
        cache = tmp_path / "cache"
        legacy = cache / ("c" * 64)
        legacy.mkdir(parents=True)
        (legacy / "data").write_text("x")
        os.utime(legacy, (OLD, OLD))
        orphan = cache / f".{'d' * 64}.versions"
        (orphan / "v1-1-1").mkdir(parents=True)
        os.utime(orphan, (OLD, OLD))
        result = _run(tmp_path, "s3pull", "s3://b/dir", cache / "h")
        assert result.returncode == 0, result.stderr
        assert not legacy.exists()
        assert not orphan.exists()

    def test_zero_disables_eviction(self, tmp_path):
        cache = tmp_path / "cache"
        old = _fake_entry(cache, "a" * 64, OLD)
        result = _run(
            tmp_path,
            "s3pull",
            "s3://b/dir",
            cache / "h",
            step_config={"cache_max_age_days": 0},
        )
        assert result.returncode == 0, result.stderr
        assert old.is_symlink()

    def test_configured_age_applies(self, tmp_path):
        """A shorter cache_max_age_days evicts a younger entry."""
        cache = tmp_path / "cache"
        aged = _fake_entry(cache, "a" * 64, int(time.time()) - 3 * DAY)
        result = _run(
            tmp_path,
            "s3pull",
            "s3://b/dir",
            cache / "h",
            step_config={"cache_max_age_days": 2},
        )
        assert result.returncode == 0, result.stderr
        assert not aged.is_symlink()

    @pytest.mark.parametrize("epoch,downloads", [(OLD, 1), (FRESH, 0)])
    def test_unchanged_object_is_reused_only_while_young(
        self, tmp_path, epoch, downloads
    ):
        """An unchanged object older than the max age is downloaded again."""
        cache = tmp_path / "cache"
        sig = "e1|5|2026-01-01T00:00:00+00:00"
        dest = _fake_entry(cache, "h", epoch, sig=sig)
        result = _run(
            tmp_path,
            "s3pull",
            "s3://b/dir/model.bin",
            dest,
            {"STUB_HEAD_RC": "0", "STUB_ETAG": "e1"},
        )
        assert result.returncode == 0, result.stderr
        assert _log(tmp_path).count(" s3 cp ") == downloads
        assert (dest / "model.bin").exists()

    def test_own_entry_is_never_evicted(self, tmp_path):
        """A hash-named destination survives its own pull's sweep."""
        cache = tmp_path / "cache"
        dest = _fake_entry(cache, "e" * 64, OLD)
        result = _run(tmp_path, "s3pull", "s3://b/dir", dest)
        assert result.returncode == 0, result.stderr
        assert dest.is_symlink() and (dest / "f").exists()


def _lease_env(run_dir: Path, run_id: str) -> Dict[str, str]:
    """Env making a pull lease for target run ``run_id`` (its workdir exists)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    return {"GB_BUILD_WORKDIR": str(run_dir), "GB_TARGETRUN_ID": run_id}


def _write_lease(
    cache: Path,
    name: str,
    run_id: str,
    run_dir: str,
    version: str,
    created: Optional[int] = None,
) -> Path:
    """Write a lease file for entry ``name`` as s3pull would."""
    leases = cache / f".{name}.leases"
    leases.mkdir(parents=True, exist_ok=True)
    lease = leases / run_id
    lease.write_text(f"{run_dir}\n{version}\n{created or int(time.time())}\n")
    return lease


class TestS3PullLeases:
    """A pull leases the live version for its target run; pruning and eviction
    never delete a version or entry a live lease (run dir present) holds."""

    def test_pull_leases_the_live_version(self, tmp_path):
        dest = tmp_path / "cache" / "h"
        run = tmp_path / "builds" / "b1" / "runs" / "tr1"
        result = _run(tmp_path, "s3pull", "s3://b/dir", dest, _lease_env(run, "tr1"))
        assert result.returncode == 0, result.stderr
        lines = (dest.parent / ".h.leases" / "tr1").read_text().splitlines()
        assert lines[0] == str(run)
        assert lines[1] == dest.resolve().name

    def test_repeat_pull_by_the_same_run_repins(self, tmp_path):
        dest = tmp_path / "cache" / "h"
        env = _lease_env(tmp_path / "runs" / "tr1", "tr1")
        for _ in range(2):
            assert _run(tmp_path, "s3pull", "s3://b/dir", dest, env).returncode == 0
        lease = dest.parent / ".h.leases" / "tr1"
        assert lease.read_text().splitlines()[1] == dest.resolve().name
        assert len(list(lease.parent.iterdir())) == 1

    def test_prune_keeps_leased_versions_until_the_run_ends(self, tmp_path):
        """Run A's version survives later pulls by run B while A's dir exists."""
        dest = tmp_path / "cache" / "h"
        run_a = tmp_path / "runs" / "trA"
        _run(tmp_path, "s3pull", "s3://b/dir", dest, _lease_env(run_a, "trA"))
        a_version = dest.resolve().name
        env_b = _lease_env(tmp_path / "runs" / "trB", "trB")
        for _ in range(2):
            result = _run(tmp_path, "s3pull", "s3://b/dir", dest, env_b)
            assert result.returncode == 0, result.stderr
        assert a_version in _versions(dest)
        assert "it is leased" in result.stdout
        # Run A's build ended: its dir is gone, so its lease and version go too.
        run_a.rmdir()
        _run(tmp_path, "s3pull", "s3://b/dir", dest, env_b)
        assert a_version not in _versions(dest)
        assert not (dest.parent / ".h.leases" / "trA").exists()

    def test_eviction_skips_an_entry_with_a_live_lease(self, tmp_path):
        cache = tmp_path / "cache"
        old = _fake_entry(cache, "a" * 64, OLD)
        run = tmp_path / "runs" / "trX"
        run.mkdir(parents=True)
        lease = _write_lease(cache, "a" * 64, "trX", str(run), old.resolve().name)
        result = _run(tmp_path, "s3pull", "s3://b/dir", cache / "h")
        assert result.returncode == 0, result.stderr
        assert old.is_symlink()
        assert lease.exists()
        assert "leased by" in result.stdout and "trX" in result.stdout

    def test_a_dead_lease_does_not_protect_and_is_removed(self, tmp_path):
        cache = tmp_path / "cache"
        old = _fake_entry(cache, "a" * 64, OLD)
        _write_lease(cache, "a" * 64, "trGone", str(tmp_path / "gone"), "x")
        result = _run(tmp_path, "s3pull", "s3://b/dir", cache / "h")
        assert result.returncode == 0, result.stderr
        assert not old.is_symlink()
        assert not (cache / f".{'a' * 64}.leases").exists()

    @pytest.mark.parametrize("max_age,evicted", [(7, True), (0, False)])
    def test_lease_max_age_days(self, tmp_path, max_age, evicted):
        """An old lease expires only when lease_max_age_days is set."""
        cache = tmp_path / "cache"
        old = _fake_entry(cache, "a" * 64, OLD)
        run = tmp_path / "runs" / "trX"
        run.mkdir(parents=True)
        _write_lease(cache, "a" * 64, "trX", str(run), "x", created=OLD)
        result = _run(
            tmp_path,
            "s3pull",
            "s3://b/dir",
            cache / "h",
            step_config={"lease_max_age_days": max_age},
        )
        assert result.returncode == 0, result.stderr
        assert old.is_symlink() is not evicted

    def test_unknown_run_dir_lease_counts_as_live(self, tmp_path):
        cache = tmp_path / "cache"
        old = _fake_entry(cache, "a" * 64, OLD)
        _write_lease(cache, "a" * 64, "trX", "-", "x")
        assert _run(tmp_path, "s3pull", "s3://b/dir", cache / "h").returncode == 0
        assert old.is_symlink()

    def test_no_lease_without_a_run_dir_or_when_mocked(self, tmp_path):
        dest = tmp_path / "cache" / "h"
        result = _run(tmp_path, "s3pull", "s3://b/dir", dest)
        assert "no reader lease taken" in result.stdout
        assert not (dest.parent / ".h.leases").exists()
        mocked = tmp_path / "cache" / "m"
        env = {**_lease_env(tmp_path / "runs" / "tr1", "tr1"), "GBTEST_MOCK_COS": "1"}
        assert (
            _run(tmp_path, "s3pull", "s3://b/k", mocked, env, creds=False).returncode
            == 0
        )
        assert not (mocked.parent / ".m.leases").exists()


def _run_link(run_dir: Path) -> Path:
    """The per-run binding Skypilot.pullasset_cosstore would choose."""
    return run_dir / ".gb_inputs" / "h"


class TestS3PullRunLink:
    """Consumers are bound to a per-run link pinned to the version their run
    leased, so another build's re-pull can't switch them to new data mid-run."""

    def _pull(self, tmp_path, dest, run_id, extra=None, uri="s3://b/dir"):
        run_dir = tmp_path / "runs" / run_id
        env = {**_lease_env(run_dir, run_id), **(extra or {})}
        result = _run(
            tmp_path,
            "s3pull",
            uri,
            dest,
            env,
            step_config={"run_link": str(_run_link(run_dir))},
        )
        assert result.returncode == 0, result.stderr
        return _run_link(run_dir), result

    def test_link_points_at_the_leased_version(self, tmp_path):
        dest = tmp_path / "cache" / "h"
        link, result = self._pull(tmp_path, dest, "trA")
        assert link.is_symlink()
        assert link.resolve() == dest.resolve()
        lease = (dest.parent / ".h.leases" / "trA").read_text().splitlines()
        assert lease[1] == link.resolve().name
        assert f"to path {link}" in result.stdout

    def test_link_keeps_its_version_when_another_run_repulls(self, tmp_path):
        """The reported bug: upstream changes and run B re-pulls while run A runs."""
        dest = tmp_path / "cache" / "h"
        link_a, _ = self._pull(tmp_path, dest, "trA", {"STUB_CONTENT": "one"})
        self._pull(tmp_path, dest, "trB", {"STUB_CONTENT": "two"})
        assert (dest / "f").read_text().strip() == "two"
        assert (link_a / "f").read_text().strip() == "one"

    def test_repull_by_the_same_run_repoints_its_link(self, tmp_path):
        dest = tmp_path / "cache" / "h"
        self._pull(tmp_path, dest, "trA", {"STUB_CONTENT": "one"})
        link, _ = self._pull(tmp_path, dest, "trA", {"STUB_CONTENT": "two"})
        assert (link / "f").read_text().strip() == "two"

    def test_unchanged_object_path_creates_the_link(self, tmp_path):
        dest = tmp_path / "cache" / "h"
        env = {"STUB_HEAD_RC": "0", "STUB_ETAG": "e1"}
        self._pull(tmp_path, dest, "trA", env, uri="s3://b/dir/model.bin")
        link, result = self._pull(
            tmp_path, dest, "trB", env, uri="s3://b/dir/model.bin"
        )
        assert "is unchanged" in result.stdout
        assert (link / "model.bin").exists()

    def test_mock_writes_the_marker_where_consumers_look(self, tmp_path):
        dest = tmp_path / "cache" / "h"
        link, _ = self._pull(
            tmp_path, dest, "trA", {"GBTEST_MOCK_COS": "true"}, uri="s3://b/k"
        )
        assert (link / ".gbtest_mock_s3pull").exists()


class TestS3PushScript:
    def test_directory_is_synced(self, tmp_path):
        """A directory source is synced under the URI as a prefix."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "a").write_text("x")
        result = _run(tmp_path, "s3push", "s3://b/out", src)
        assert result.returncode == 0, result.stderr
        assert f"s3 sync --only-show-errors {src}/ s3://b/out/" in _log(tmp_path)

    def test_file_is_copied_to_exact_key(self, tmp_path):
        """A file source is copied to the exact object key."""
        src = tmp_path / "model.bin"
        src.write_text("x")
        result = _run(tmp_path, "s3push", "s3://b/out/model.bin", src)
        assert result.returncode == 0, result.stderr
        assert f"s3 cp --only-show-errors {src} s3://b/out/model.bin" in _log(tmp_path)

    def test_missing_source_fails(self, tmp_path):
        """A missing source path is an error, even when mocked."""
        result = _run(
            tmp_path, "s3push", "s3://b/o", tmp_path / "nope", {"GBTEST_MOCK_COS": "1"}
        )
        assert result.returncode != 0
        assert "does not exist" in result.stderr

    def test_mocked_skips_aws(self, tmp_path):
        """GBTEST_MOCK_COS skips the upload (and needs no credentials)."""
        src = tmp_path / "src"
        src.mkdir()
        result = _run(
            tmp_path, "s3push", "s3://b/o", src, {"GBTEST_MOCK_COS": "on"}, creds=False
        )
        assert result.returncode == 0, result.stderr
        assert _log(tmp_path) == ""


class TestS3PushArtifactEvent:
    """The push reports success the way hfpush does: a ``Pushed S3 URI:`` line
    the step's monitor turns into ARTIFACT_PUSHED_EVENT (artifact uri + binding)."""

    def _event_config(self) -> Dict:
        data = yaml.safe_load((STEPS / "s3push" / "step.yaml").read_text())
        monitor = data["environment_configs"]["Skypilot"]["monitors"][
            "skypilot_monitor"
        ]
        (event,) = monitor["config"]["event_configs"]
        return event

    @pytest.mark.parametrize("mock", [False, True])
    def test_pushed_line_parses_to_artifact_event(self, tmp_path, mock):
        """Real and mocked pushes both print a line the monitor regexes match."""
        import re

        src = tmp_path / "f.txt"
        src.write_text("x")
        extra = {"GBTEST_MOCK_COS": "true"} if mock else {}
        result = _run(tmp_path, "s3push", "s3://gb-checkpoints/a", src, extra)
        assert result.returncode == 0, result.stderr
        event = self._event_config()
        assert event["event_type"] == "ARTIFACT_PUSHED_EVENT"
        (line,) = [
            ln
            for ln in result.stdout.splitlines()
            if re.search(event["line_regex"], ln)
        ]
        fields = {
            f["field_name"]: re.search(f["field_regex"], line).group(0)
            for f in event["event_fields"]
        }
        assert fields == {"uri": "s3://local/artifact", "binding_id": "bind-1"}
