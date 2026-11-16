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
from pathlib import Path
from typing import Dict, Optional, Tuple

import pytest
import yaml

from gbserver.utils.template import fill_template

STEPS = Path("src/gbserver/builtins/steps/skypilot")

# Records each invocation and fakes downloads into a local destination:
# - head-object succeeds only when STUB_HEAD_RC=0;
# - a download sync writes the objects named in STUB_FILES (default "f") with
#   content STUB_CONTENT, each via a temp file + rename like the real CLI (so a
#   hard-linked previous version is never modified); --delete removes files not
#   in STUB_FILES; STUB_EMPTY makes the prefix empty; STUB_SLEEP pauses inside
#   the sync, which is bracketed by sync-start/sync-end log lines.
STUB_AWS = """#!/usr/bin/env bash
echo "$*" >> "${STUB_LOG}"
last="${!#}"
case " $* " in
  *" s3api head-object "*) exit "${STUB_HEAD_RC:-1}" ;;
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
) -> Tuple[str, Dict[str, str]]:
    """Render a step's run script and build its env, with the stub aws on PATH.

    :param tmp_path: pytest tmp dir (holds the stub and its log).
    :param kind: ``s3pull`` or ``s3push``.
    :param s3_uri: the configured s3_uri.
    :param local_path: the configured local_path.
    :param extra_env: additional env vars for the script.
    :param creds: whether to set the (dummy) AWS credential env vars.
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
) -> subprocess.CompletedProcess:
    """Render and execute a step's run script against the stub aws.

    Arguments as for :func:`_prepare`.

    :returns: the completed process (stdout/stderr captured).
    """
    script, env = _prepare(tmp_path, kind, s3_uri, local_path, extra_env, creds)
    return subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, check=False
    )


def _versions(dest: Path) -> list:
    """The version directories behind an s3pull cache entry ``dest``."""
    root = dest.parent / f".{dest.name}.versions"
    return sorted(v.name for v in root.iterdir()) if root.exists() else []


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
        assert "reclaiming stale lock" in result.stderr

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
