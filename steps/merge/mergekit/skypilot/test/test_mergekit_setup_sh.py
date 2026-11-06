# Copyright LLM.build Authors
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

"""Tests for src/setup.sh: the uv install, and that it can run again in the same workdir.

SkyPilot's retry relaunches a failed step in the same per-run workdir, so setup must
cope with the venv a previous attempt left behind. The stub tests replace pip and uv
with recorders; the real-uv test (skipped without uv on PATH) creates a real venv twice.
"""

import os
import pathlib
import shutil
import subprocess

import pytest

_SETUP = pathlib.Path(__file__).resolve().parents[1] / "src" / "setup.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash not available"
)

# pip (for the uv bootstrap) and uv: record argv. The stub `uv venv` behaves like an
# older uv that a node may already have (`pip install uv` doesn't upgrade it): it rejects
# --clear and --force (added in uv 0.8 and 0.12) and refuses a path that exists. It
# creates pyvenv.cfg and bin/activate so setup.sh can source it.
_UV = """#!/usr/bin/env bash
echo "uv $*" >> "$STUB_LOG"
if [ "$1" = venv ]; then
  shift; dir=""
  while [ $# -gt 0 ]; do
    case $1 in
      --clear|--force) echo "error: unexpected argument '$1' found" >&2; exit 2 ;;
      --python) shift ;;
      *) dir=$1 ;;
    esac; shift
  done
  if [ -e "$dir" ]; then
    echo "error: A virtual environment already exists at: $dir" >&2; exit 2
  fi
  mkdir -p "$dir/bin"; : > "$dir/bin/activate"; : > "$dir/pyvenv.cfg"
fi
exit 0
"""
_PIP = '#!/usr/bin/env bash\necho "pip $*" >> "$STUB_LOG"\nexit 0\n'


def _stub_env(tmp_path: pathlib.Path) -> dict:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, body in (("uv", _UV), ("pip", _PIP)):
        (bindir / name).write_text(body)
        (bindir / name).chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "STUB_LOG": str(tmp_path / "calls.log"),
        "GB_SHARED_WORKDIR": str(tmp_path / "shared"),
    }


def _setup(cwd: pathlib.Path, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "bash",
            str(_SETUP),
            "--venv",
            "./venv",
            "--index-url",
            "https://pypi.org/simple",
            "--",
            "mergekit==0.1.5",
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_installs_the_requested_package(tmp_path):
    env = _stub_env(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    proc = _setup(work, env)
    assert proc.returncode == 0, proc.stderr
    calls = (tmp_path / "calls.log").read_text()
    assert (
        "uv pip install --quiet --index-url https://pypi.org/simple mergekit==0.1.5"
        in calls
    )


def test_second_run_in_the_same_workdir_succeeds(tmp_path):
    """A retry finds the previous attempt's ./venv; setup must not refuse it."""
    env = _stub_env(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    assert _setup(work, env).returncode == 0
    (work / "venv" / "half-finished-install").write_text("x")
    proc = _setup(work, env)
    assert proc.returncode == 0, proc.stderr
    # Started fresh, so a half-finished install from the failed attempt is gone.
    assert not (work / "venv" / "half-finished-install").exists()


def test_retry_after_an_attempt_killed_mid_venv_succeeds(tmp_path):
    """An attempt killed while uv created the venv leaves a dir without pyvenv.cfg."""
    env = _stub_env(tmp_path)
    work = tmp_path / "work"
    (work / "venv" / "bin").mkdir(parents=True)
    proc = _setup(work, env)
    assert proc.returncode == 0, proc.stderr
    assert (work / "venv" / "pyvenv.cfg").is_file()


def test_uv_cache_is_on_the_shared_workdir(tmp_path):
    """The cache is under GB_SHARED_WORKDIR, so torch downloads once per cluster."""
    env = _stub_env(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    probe = tmp_path / "bin" / "uv"
    probe.write_text(
        probe.read_text().replace(
            'echo "uv $*" >> "$STUB_LOG"',
            'echo "uv $* cache=$UV_CACHE_DIR python=$UV_PYTHON_INSTALL_DIR" >> "$STUB_LOG"',
        )
    )
    assert _setup(work, env).returncode == 0
    calls = (tmp_path / "calls.log").read_text()
    assert f"cache={tmp_path}/shared/.uv-cache" in calls
    assert f"python={tmp_path}/shared/.uv-python" in calls


def test_real_uv_second_run_succeeds(tmp_path):
    """With real uv: run setup's venv step twice in one workdir (no package install)."""
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")
    work = tmp_path / "work"
    work.mkdir()
    env = {**os.environ, "GB_SHARED_WORKDIR": str(tmp_path / "shared")}
    pip = (
        tmp_path / "bin" / "pip"
    )  # uv is already installed: make the bootstrap a no-op
    pip.parent.mkdir()
    pip.write_text("#!/usr/bin/env bash\nexit 0\n")
    pip.chmod(0o755)
    env["PATH"] = f"{pip.parent}:{env['PATH']}"
    # setup.sh asks for 3.12 with its managed-Python dir set to the (empty) shared one, so
    # `uv venv` can only use a system 3.12 (a virtualenv's interpreter on PATH doesn't
    # count). Without one it would download CPython: never let it, and skip instead.
    env["UV_PYTHON_DOWNLOADS"] = "never"
    found = subprocess.run(
        ["uv", "python", "find", "--system", "--no-python-downloads", "3.12"],
        env={**env, "UV_PYTHON_INSTALL_DIR": str(tmp_path / "shared" / ".uv-python")},
        capture_output=True,
        check=False,
    )
    if found.returncode != 0:
        pytest.skip("no system Python 3.12 for uv to use offline")
    for _ in range(2):
        proc = subprocess.run(
            [
                "bash",
                str(_SETUP),
                "--venv",
                "./venv",
                "--index-url",
                "https://pypi.org/simple",
            ],
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
    assert (work / "venv" / "bin" / "python").exists()
