"""Checks on the k8s lhpull chart's shell command.

Asserted against the template source (and, for the lock guard, by running the
extracted bash) so the checks need no helm binary.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

LHPULL_HELPERS = (
    Path(__file__).resolve().parents[3]
    / "src/gbserver/builtins/steps/k8s/lhpull/helm-charts/lhpull/templates/_helpers.tpl"
)


def _template() -> str:
    return LHPULL_HELPERS.read_text(encoding="utf-8")


def _shell_lines() -> list[str]:
    """Template lines that are shell, not comments or Go-template directives."""
    return [
        ln
        for ln in _template().splitlines()
        if not ln.lstrip().startswith("#") and not ln.lstrip().startswith("{{")
    ]


def _lock_guard() -> str:
    """The `if ! exec 9>... fi` block that opens the lock file."""
    lines = _template().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("if ! exec 9>"))
    end = next(n for n in range(start, len(lines)) if lines[n] == "fi")
    return "\n".join(lines[start : end + 1])


def test_no_double_dollar_pid():
    """Kubernetes rewrites `$$` to `$` in a container command, so `$$` logs `pid=$`."""
    offenders = [ln for ln in _shell_lines() if "$$" in ln]
    assert not offenders, offenders


def test_no_hostname_binary():
    """ubi10-minimal images lack `hostname`; use bash's HOSTNAME instead."""
    offenders = [ln for ln in _shell_lines() if re.search(r"\bhostname\b", ln)]
    assert not offenders, offenders


def test_bashpid_read_outside_pipeline():
    """BASHPID inside `echo ... | tee` would name the pipeline subshell, not lhpull."""
    offenders = [ln for ln in _shell_lines() if "BASHPID" in ln and "|" in ln]
    assert not offenders, offenders
    assert any(ln.strip() == 'MY_PID="${BASHPID}"' for ln in _shell_lines())


def _gnu_stat() -> bool:
    probe = subprocess.run(
        ["stat", "-c", "%u", "/"], capture_output=True, text=True, check=False
    )
    return probe.returncode == 0


def _run_guard(tmp_path: Path, unwritable: str = "") -> subprocess.CompletedProcess:
    """Run the guard; *unwritable* names a file ("lock" or "state") to make read-only."""
    lhpull_dir = tmp_path / "cache"
    lhpull_dir.mkdir()
    files = {
        "lock": lhpull_dir / ".llmbpull.lock",
        "state": lhpull_dir / ".llmbpull.state",
    }
    for f in files.values():
        f.touch()
    if unwritable:
        files[unwritable].chmod(0o444)
    script = (
        f'LHPULL_DIR="{lhpull_dir}"\nLOCK_FILE="{files["lock"]}"\n'
        f'STATE_FILE="{files["state"]}"\n{_lock_guard()}\necho GUARD_PASSED'
    )
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=False
    )


def test_guard_passes_when_writable(tmp_path):
    result = _run_guard(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "GUARD_PASSED" in result.stdout


@pytest.mark.parametrize("unwritable", ["lock", "state"])
def test_guard_explains_permission_failure(tmp_path, unwritable):
    """An unwritable lock or state file fails the step, naming the uid and modes."""
    if os.geteuid() == 0:
        pytest.skip("root ignores file modes, so the open cannot fail")
    result = _run_guard(tmp_path, unwritable)
    assert result.returncode == 1
    assert "GUARD_PASSED" not in result.stdout
    assert "ERROR: cannot write to" in result.stdout
    assert "uid=" in result.stdout
    assert "group-writable" in result.stdout
    if _gnu_stat():
        # owner:group and mode per path, e.g. "  1000:0 -r--r--r-- .../.llmbpull.lock"
        assert re.search(r"^  \d+:\d+ -r--r--r-- .*llmbpull", result.stdout, re.M)
