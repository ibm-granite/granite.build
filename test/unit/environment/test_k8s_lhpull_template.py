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


def _run_guard(tmp_path: Path, writable: bool) -> subprocess.CompletedProcess:
    lhpull_dir = tmp_path / "cache"
    lhpull_dir.mkdir()
    lock = lhpull_dir / ".llmbpull.lock"
    lock.touch()
    if not writable:
        lock.chmod(0o444)
    script = (
        f'LHPULL_DIR="{lhpull_dir}"\nLOCK_FILE="{lock}"\n'
        f"{_lock_guard()}\necho LOCK_OPENED"
    )
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=False
    )


def test_lock_guard_opens_writable_lock(tmp_path):
    result = _run_guard(tmp_path, writable=True)
    assert result.returncode == 0, result.stderr
    assert "LOCK_OPENED" in result.stdout


def test_lock_guard_explains_permission_failure(tmp_path):
    """An unwritable lock fails the step and names the uid and the file modes."""
    if os.geteuid() == 0:
        pytest.skip("root ignores file modes, so the open cannot fail")
    result = _run_guard(tmp_path, writable=False)
    assert result.returncode == 1
    assert "LOCK_OPENED" not in result.stdout
    assert "ERROR: cannot open" in result.stdout
    assert "uid=" in result.stdout
    assert "group-writable" in result.stdout
